import html
import logging
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
TELEGRAM_API_BASE = "https://api.telegram.org"
YOUTUBE_TIMEDTEXT_BASE = "https://www.youtube.com/api/timedtext"
logger = logging.getLogger("youtube_digest")


@dataclass
class Video:
    video_id: str
    title: str
    description: str
    published_at: str
    channel_id: str
    channel_title: str
    url: str
    duration_seconds: int = 0
    chapters: List[str] = field(default_factory=list)
    transcript: str = ""
    summary_source: str = "metadata"


class ModelOutputError(RuntimeError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_csv(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


class GeminiRateLimiter:
    """Pace Gemini calls to stay under requests-per-minute (rolling window approximation)."""

    def __init__(self, rpm: int) -> None:
        self.rpm = max(1, rpm)
        self.min_interval_s = 60.0 / self.rpm
        self._last_call_at = 0.0

    @property
    def interval_ms(self) -> int:
        return int(self.min_interval_s * 1000)

    def wait(self, context: str = "") -> None:
        now = time.monotonic()
        if self._last_call_at > 0:
            elapsed = now - self._last_call_at
            if elapsed < self.min_interval_s:
                delay = self.min_interval_s - elapsed
                logger.info(
                    "gemini_rate_limit wait_s=%.1f rpm=%s context=%s",
                    delay,
                    self.rpm,
                    context or "-",
                )
                time.sleep(delay)
        self._last_call_at = time.monotonic()


def configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def redact_secrets(text: str) -> str:
    if not text:
        return text
    redacted = re.sub(r"([?&]key=)[^&\s]+", r"\1[REDACTED]", text, flags=re.IGNORECASE)
    redacted = re.sub(r"\bAIza[0-9A-Za-z_-]{20,}\b", "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"\bAQ\.[A-Za-z0-9._-]{20,}\b", "[REDACTED_API_KEY]", redacted)
    redacted = re.sub(r"\b\d{7,12}:[A-Za-z0-9_-]{20,}\b", "[REDACTED_BOT_TOKEN]", redacted)
    return redacted


def classify_error(exc: Exception) -> str:
    if isinstance(exc, ModelOutputError):
        return exc.code
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        if status == 429:
            return "RATE_LIMIT"
        if status in (500, 502, 503, 504):
            return "UPSTREAM_UNAVAILABLE"
        if status in (400, 401, 403):
            return "AUTH_OR_REQUEST_ERROR"
    if isinstance(exc, requests.Timeout):
        return "TIMEOUT"
    msg = str(exc)
    if "429" in msg and "Too Many Requests" in msg:
        return "RATE_LIMIT"
    return "UNKNOWN"


def video_label(video: Video) -> str:
    return f"{video.channel_title} | {video.title} | {video.video_id}"


def first_last_words(text: str, words: int = 6) -> Tuple[str, str]:
    tokens = [t for t in text.split() if t]
    if not tokens:
        return "-", "-"
    return " ".join(tokens[:words]), " ".join(tokens[-words:])


def looks_incomplete_sentence(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith((":", "-", "(", "/", "is", "are", "was", "were", "the", "and", "or")):
        return True
    return stripped[-1] not in ".!?)”\""


def count_bullets(text: str) -> int:
    count = 0
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("-", "*")):
            count += 1
            continue
        if re.match(r"^\d+[.)]\s+", s):
            count += 1
    return count


def load_config() -> Dict:
    load_dotenv(override=True)
    required = [
        "YOUTUBE_API_KEY",
        "GEMINI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID",
        "CHANNEL_IDS",
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

    return {
        "youtube_api_key": os.getenv("YOUTUBE_API_KEY", "").strip(),
        "gemini_api_key": os.getenv("GEMINI_API_KEY", "").strip(),
        "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        "channel_ids": [c.strip() for c in os.getenv("CHANNEL_IDS", "").split(",") if c.strip()],
        "lookback_hours": env_int("LOOKBACK_HOURS", 24),
        "max_videos_per_channel": env_int("MAX_VIDEOS_PER_CHANNEL", 2),
        "max_total_videos": env_int("MAX_TOTAL_VIDEOS", 4),
        "min_duration_seconds": env_int("MIN_DURATION_SECONDS", 180),
        "max_description_chars": env_int("MAX_DESCRIPTION_CHARS", 1200),
        "max_transcript_chars": env_int("MAX_TRANSCRIPT_CHARS", 8000),
        "youtube_channel_delay_ms": env_int("YOUTUBE_CHANNEL_DELAY_MS", 1500),
        "youtube_max_retries": env_int("YOUTUBE_MAX_RETRIES", 5),
        "gemini_rpm": env_int("GEMINI_RPM", 4),
        "gemini_call_delay_ms": env_int("GEMINI_CALL_DELAY_MS", 0),
        "gemini_final_delay_ms": env_int("GEMINI_FINAL_DELAY_MS", 0),
        "llm_max_retries": env_int("LLM_MAX_RETRIES", 3),
        "llm_backoff_base_seconds": env_int("LLM_BACKOFF_BASE_SECONDS", 3),
        "video_max_output_tokens": env_int("VIDEO_MAX_OUTPUT_TOKENS", 1000),
        "final_max_output_tokens": env_int("FINAL_MAX_OUTPUT_TOKENS", 1400),
        "min_bullets_per_video_summary": env_int("MIN_BULLETS_PER_VIDEO_SUMMARY", 3),
        "min_bullets_for_final_digest": env_int("MIN_BULLETS_FOR_FINAL_DIGEST", 5),
        "noise_title_keywords": env_csv(
            "NOISE_TITLE_KEYWORDS",
            "shorts,relationship,motivation,vlog,daily routine,tested positive",
        ),
        "transcript_languages": env_csv("TRANSCRIPT_LANGUAGES", "en,hi,en-US,en-GB,hi-IN"),
        "video_summary_model": os.getenv("VIDEO_SUMMARY_MODEL", "gemini-2.5-flash-lite").strip(),
        "video_summary_fallback_model": os.getenv(
            "VIDEO_SUMMARY_FALLBACK_MODEL", "gemini-2.5-flash-lite"
        ).strip(),
        "digest_model": os.getenv("DIGEST_MODEL", "gemini-2.5-flash-lite").strip(),
        "digest_fallback_model": os.getenv("DIGEST_FALLBACK_MODEL", "gemini-2.5-flash-lite").strip(),
    }


def youtube_get_with_retry(
    url: str,
    params: Dict,
    max_retries: int,
    backoff_base_seconds: int,
    context_label: str,
) -> Dict:
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
                delay = min(backoff_base_seconds ** (attempt + 1), 20) + random.uniform(0, 0.5)
                logger.warning(
                    "youtube_retry context=%s attempt=%s status=%s delay_s=%.2f",
                    context_label,
                    attempt + 1,
                    status,
                    delay,
                )
                time.sleep(delay)
                continue
            raise
        except requests.RequestException as exc:
            last_exc = exc
            if attempt + 1 < max_retries:
                delay = min(backoff_base_seconds ** (attempt + 1), 10) + random.uniform(0, 0.4)
                time.sleep(delay)
                continue
            raise
    if last_exc:
        raise RuntimeError(redact_secrets(str(last_exc)))
    raise RuntimeError(f"YOUTUBE_REQUEST_FAILED ({context_label})")


def fetch_channel_titles(
    youtube_api_key: str,
    channel_ids: List[str],
    youtube_max_retries: int,
    llm_backoff_base_seconds: int,
) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i : i + 50]
        data = youtube_get_with_retry(
            url=f"{YOUTUBE_API_BASE}/channels",
            params={
                "part": "snippet",
                "id": ",".join(chunk),
                "key": youtube_api_key,
            },
            max_retries=youtube_max_retries,
            backoff_base_seconds=llm_backoff_base_seconds,
            context_label="channel_titles",
        )
        for item in data.get("items", []):
            cid = item.get("id", "")
            title = item.get("snippet", {}).get("title", "Unknown Channel")
            titles[cid] = title
    return titles


def parse_iso8601_duration(value: str) -> int:
    if not value or not value.startswith("PT"):
        return 0
    pattern = r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$"
    match = re.match(pattern, value)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def extract_chapters(description: str) -> List[str]:
    matches = re.findall(r"(?m)^\s*\d{1,2}:\d{2}(?::\d{2})?\s+(.+)$", description or "")
    clean = [m.strip(" -–|") for m in matches if m.strip()]
    return clean[:12]


def fetch_video_durations(
    youtube_api_key: str,
    video_ids: List[str],
    youtube_max_retries: int,
    llm_backoff_base_seconds: int,
) -> Dict[str, int]:
    durations: Dict[str, int] = {}
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i : i + 50]
        data = youtube_get_with_retry(
            url=f"{YOUTUBE_API_BASE}/videos",
            params={
                "part": "contentDetails",
                "id": ",".join(chunk),
                "key": youtube_api_key,
            },
            max_retries=youtube_max_retries,
            backoff_base_seconds=llm_backoff_base_seconds,
            context_label="video_durations",
        )
        for item in data.get("items", []):
            vid = item.get("id", "")
            iso_duration = item.get("contentDetails", {}).get("duration", "")
            durations[vid] = parse_iso8601_duration(iso_duration)
    return durations


def parse_timedtext_xml(xml_text: str) -> str:
    if not xml_text.strip():
        return ""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return ""
    parts: List[str] = []
    for node in root.findall("text"):
        raw = "".join(node.itertext())
        clean = html.unescape(raw).replace("\n", " ").strip()
        if clean:
            parts.append(clean)
    return " ".join(parts).strip()


def fetch_transcript(
    video_id: str, preferred_languages: List[str], timeout: int = 20
) -> Tuple[str, str, str, str]:
    params = {"type": "list", "v": video_id}
    try:
        resp = requests.get(YOUTUBE_TIMEDTEXT_BASE, params=params, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        return "", "LIST_FETCH_FAILED", "", redact_secrets(str(exc))

    langs: List[str] = []
    try:
        root = ET.fromstring(resp.text)
        for track in root.findall("track"):
            lang = track.attrib.get("lang_code", "").strip()
            if lang:
                langs.append(lang)
    except ET.ParseError:
        langs = []

    if not langs:
        return "", "NO_TRACKS", "", ""

    ordered = [lang for lang in preferred_languages if lang in langs]
    if not ordered:
        ordered = langs[:1]

    last_error = ""
    for lang in ordered:
        try:
            cap_resp = requests.get(
                YOUTUBE_TIMEDTEXT_BASE,
                params={"v": video_id, "lang": lang, "fmt": "srv3"},
                timeout=timeout,
            )
            cap_resp.raise_for_status()
            transcript = parse_timedtext_xml(cap_resp.text)
            if transcript:
                return transcript, "OK", lang, ""
        except requests.RequestException as exc:
            last_error = redact_secrets(str(exc))
            continue
    if last_error:
        return "", "CAPTION_FETCH_FAILED", ordered[0], last_error
    return "", "EMPTY_CAPTION_TEXT", ordered[0], ""


def is_noise_video(video: Video, min_duration_seconds: int, blocked_keywords: List[str]) -> bool:
    title_lower = video.title.lower()
    description_lower = video.description.lower()
    if "#shorts" in title_lower or "#shorts" in description_lower:
        return True
    if video.duration_seconds and video.duration_seconds < min_duration_seconds:
        return True
    for keyword in blocked_keywords:
        token = keyword.lower()
        if token in title_lower:
            return True
    return False


def noise_reason(video: Video, min_duration_seconds: int, blocked_keywords: List[str]) -> str:
    title_lower = video.title.lower()
    description_lower = video.description.lower()
    if "#shorts" in title_lower or "#shorts" in description_lower:
        return "HAS_SHORTS_TAG"
    if video.duration_seconds and video.duration_seconds < min_duration_seconds:
        return f"DURATION_LT_{min_duration_seconds}"
    for keyword in blocked_keywords:
        token = keyword.lower()
        if token in title_lower:
            return f"TITLE_MATCH_{token}"
    return ""


def split_videos_by_signal(
    videos: List[Video], min_duration_seconds: int, blocked_keywords: List[str]
) -> Tuple[List[Video], List[Video]]:
    kept: List[Video] = []
    dropped: List[Video] = []
    for video in videos:
        reason = noise_reason(video, min_duration_seconds, blocked_keywords)
        if reason:
            dropped.append(video)
            logger.info("video_filtered reason=%s video=%s", reason, video_label(video))
        else:
            kept.append(video)
    return kept, dropped


def fetch_recent_videos(
    youtube_api_key: str,
    channel_ids: List[str],
    lookback_hours: int,
    max_videos_per_channel: int,
    youtube_channel_delay_ms: int,
    youtube_max_retries: int,
    llm_backoff_base_seconds: int,
) -> List[Video]:
    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    channel_titles: Dict[str, str] = {}
    try:
        channel_titles = fetch_channel_titles(
            youtube_api_key,
            channel_ids,
            youtube_max_retries,
            llm_backoff_base_seconds,
        )
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 429:
            logger.warning("channel_titles_skipped reason=rate_limit")
        else:
            raise

    videos: List[Video] = []
    skipped_channels = 0

    for idx, channel_id in enumerate(channel_ids):
        try:
            data = youtube_get_with_retry(
                url=f"{YOUTUBE_API_BASE}/search",
                params={
                    "part": "snippet",
                    "channelId": channel_id,
                    "type": "video",
                    "order": "date",
                    "publishedAfter": published_after,
                    "maxResults": max_videos_per_channel,
                    "key": youtube_api_key,
                },
                max_retries=youtube_max_retries,
                backoff_base_seconds=llm_backoff_base_seconds,
                context_label=f"channel_search:{channel_id}",
            )
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                skipped_channels += 1
                logger.warning("channel_skipped reason=rate_limit channel=%s", channel_id)
                if idx + 1 < len(channel_ids):
                    sleep_ms(youtube_channel_delay_ms * 2)
                continue
            raise

        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if not vid:
                continue
            snippet = item.get("snippet", {})
            videos.append(
                Video(
                    video_id=vid,
                    title=html.unescape(snippet.get("title", "").strip()),
                    description=html.unescape(snippet.get("description", "").strip()),
                    published_at=snippet.get("publishedAt", ""),
                    channel_id=channel_id,
                    channel_title=channel_titles.get(channel_id, snippet.get("channelTitle", "Unknown Channel")),
                    url=f"https://www.youtube.com/watch?v={vid}",
                )
            )

        if idx + 1 < len(channel_ids):
            sleep_ms(youtube_channel_delay_ms)

    if skipped_channels:
        logger.warning("channels_rate_limited skipped=%s fetched_videos=%s", skipped_channels, len(videos))

    # Deduplicate by video_id and sort latest first
    unique: Dict[str, Video] = {v.video_id: v for v in videos}
    deduped = list(unique.values())
    deduped.sort(key=lambda v: v.published_at, reverse=True)

    durations = fetch_video_durations(
        youtube_api_key,
        [v.video_id for v in deduped],
        youtube_max_retries,
        llm_backoff_base_seconds,
    )
    for video in deduped:
        video.duration_seconds = durations.get(video.video_id, 0)
        video.chapters = extract_chapters(video.description)

    return deduped


def call_gemini(
    model: str,
    api_key: str,
    prompt: str,
    max_output_tokens: int,
) -> Tuple[str, str]:
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_output_tokens,
        },
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        candidate = data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"].strip()
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        return text, finish_reason
    except (KeyError, IndexError, TypeError):
        raise ModelOutputError("INVALID_MODEL_RESPONSE")


def call_gemini_with_retry(
    prompt: str,
    primary_model: str,
    fallback_model: str,
    api_key: str,
    max_retries: int,
    backoff_base_seconds: int,
    base_max_output_tokens: int,
    min_bullets: int,
    context_label: str,
    rate_limiter: Optional[GeminiRateLimiter] = None,
) -> str:
    models = [primary_model]
    if fallback_model and fallback_model != primary_model:
        models.append(fallback_model)

    last_exc: Optional[Exception] = None

    for model in models:
        for attempt in range(max_retries):
            max_tokens = min(base_max_output_tokens + (attempt * 250), base_max_output_tokens * 2)
            try:
                if rate_limiter:
                    rate_limiter.wait(context_label)
                text, finish_reason = call_gemini(
                    model=model,
                    api_key=api_key,
                    prompt=prompt,
                    max_output_tokens=max_tokens,
                )
                bullet_count = count_bullets(text)
                truncated = finish_reason == "MAX_TOKENS" or looks_incomplete_sentence(text)
                insufficient_structure = bullet_count < min_bullets
                if truncated or insufficient_structure:
                    first, last = first_last_words(text)
                    logger.warning(
                        "llm_output_issue context=%s model=%s attempt=%s finish_reason=%s "
                        "bullet_count=%s min_bullets=%s max_tokens=%s first_words=%s last_words=%s",
                        context_label,
                        model,
                        attempt + 1,
                        finish_reason,
                        bullet_count,
                        min_bullets,
                        max_tokens,
                        first,
                        last,
                    )
                    if attempt + 1 < max_retries:
                        delay = min(backoff_base_seconds ** (attempt + 1), 10) + random.uniform(0, 0.4)
                        time.sleep(delay)
                        continue
                    raise ModelOutputError(
                        "TRUNCATED_OR_INCOMPLETE_OUTPUT",
                        (
                            f"finish_reason={finish_reason}, "
                            f"bullet_count={bullet_count}, first='{first}', last='{last}'"
                        ),
                    )
                return text
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                if status in (429, 500, 502, 503, 504):
                    # On 429, wait at least one full RPM window before retrying.
                    if status == 429 and rate_limiter:
                        rate_limiter.wait(f"{context_label}:429_retry")
                    delay = min(backoff_base_seconds ** (attempt + 1), 20) + random.uniform(0, 0.5)
                    time.sleep(delay)
                    continue
                break
            except requests.RequestException as exc:
                last_exc = exc
                delay = min(backoff_base_seconds ** (attempt + 1), 10) + random.uniform(0, 0.5)
                time.sleep(delay)
                continue
            except Exception as exc:
                last_exc = exc
                break

    if last_exc:
        raise RuntimeError(redact_secrets(str(last_exc)))
    raise RuntimeError(f"MODEL_CALL_FAILED ({context_label})")


def summarize_video(
    video: Video,
    model: str,
    fallback_model: str,
    gemini_api_key: str,
    transcript_languages: List[str],
    max_description_chars: int,
    max_transcript_chars: int,
    llm_max_retries: int,
    llm_backoff_base_seconds: int,
    video_max_output_tokens: int,
    min_bullets_per_video_summary: int,
    rate_limiter: Optional[GeminiRateLimiter] = None,
) -> str:
    if not video.transcript:
        transcript, transcript_status, transcript_lang, transcript_error = fetch_transcript(
            video.video_id, transcript_languages
        )
        video.transcript = transcript
        if transcript:
            logger.info(
                "transcript_loaded status=%s lang=%s chars=%s video=%s",
                transcript_status,
                transcript_lang or "-",
                len(transcript),
                video_label(video),
            )
        else:
            logger.warning(
                "transcript_unavailable status=%s lang=%s detail=%s video=%s",
                transcript_status,
                transcript_lang or "-",
                transcript_error or "-",
                video_label(video),
            )

    chapters_text = "\n".join(f"- {chapter}" for chapter in video.chapters) or "- Not available"
    transcript_excerpt = video.transcript[:max_transcript_chars].strip()
    if transcript_excerpt:
        source_text = transcript_excerpt
        video.summary_source = "transcript"
        source_label = "Transcript"
    else:
        source_text = video.description[:max_description_chars].strip()
        video.summary_source = "metadata"
        source_label = "Description"
    logger.info(
        "summary_input source=%s chars=%s chapters=%s duration=%s video=%s",
        video.summary_source,
        len(source_text),
        len(video.chapters),
        video.duration_seconds,
        video_label(video),
    )

    prompt = f"""
You are an analyst making concise daily intel notes.

Summarize this YouTube video in max 5 bullets:
- What happened / key topic
- Most useful insight
- Why it matters (AI/tech/startup/market/dev trends)
- Top subtopics (if multiple themes are discussed)
- One watchout (optional)

Return plain text bullets only.

Video metadata:
Channel: {video.channel_title}
Title: {video.title}
Published At: {video.published_at}
Duration Seconds: {video.duration_seconds}
Chapters:
{chapters_text}
Source Used: {source_label}
Content:
{source_text if source_text else "No transcript/description content available."}
""".strip()
    return call_gemini_with_retry(
        prompt=prompt,
        primary_model=model,
        fallback_model=fallback_model,
        api_key=gemini_api_key,
        max_retries=llm_max_retries,
        backoff_base_seconds=llm_backoff_base_seconds,
        base_max_output_tokens=video_max_output_tokens,
        min_bullets=min_bullets_per_video_summary,
        context_label=f"video_summary:{video.video_id}",
        rate_limiter=rate_limiter,
    )


def build_digest(videos: List[Video], per_video_summaries: Dict[str, str], lookback_hours: int) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = [
        f"📺 YouTube Signal Digest ({lookback_hours}h)",
        f"Generated: {now_utc}",
        f"Videos covered: {len(videos)}",
        "",
    ]

    lines = []
    for i, v in enumerate(videos, start=1):
        lines.append(f"{i}. {v.channel_title} — {v.title}")
        lines.append(f"   {v.url}")
        lines.append(f"   Source: {v.summary_source}")
        summary = per_video_summaries.get(v.video_id, "No summary.")
        for sline in summary.splitlines():
            sline = sline.strip()
            if not sline:
                continue
            lines.append(f"   {sline}")
        lines.append("")

    return "\n".join(header + lines).strip()


def build_final_editorial_digest(
    digest_text: str,
    model: str,
    fallback_model: str,
    gemini_api_key: str,
    llm_max_retries: int,
    llm_backoff_base_seconds: int,
    final_max_output_tokens: int,
    min_bullets_for_final_digest: int,
    rate_limiter: Optional[GeminiRateLimiter] = None,
) -> str:
    prompt = f"""
You are writing a concise intelligence briefing for one user.

Input is raw per-video notes. Create final digest in this format:

1) Top 5 signals today
2) AI & Tech
3) Startup & Business
4) Market / Macro
5) What to watch next 24h

Rules:
- Keep it factual and concise.
- Use bullet points.
- Mention source channel names where useful.
- No fluff.

Raw notes:
{digest_text[:20000]}
""".strip()
    return call_gemini_with_retry(
        prompt=prompt,
        primary_model=model,
        fallback_model=fallback_model,
        api_key=gemini_api_key,
        max_retries=llm_max_retries,
        backoff_base_seconds=llm_backoff_base_seconds,
        base_max_output_tokens=final_max_output_tokens,
        min_bullets=min_bullets_for_final_digest,
        context_label="final_editorial_digest",
        rate_limiter=rate_limiter,
    )


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
    # Defense-in-depth: never send raw provider URLs/keys to Telegram.
    text = redact_secrets(text)
    # Telegram limit is ~4096 chars per message
    chunks = []
    current = []
    current_len = 0
    max_len = 3800

    for line in text.splitlines():
        line_len = len(line) + 1
        if current_len + line_len > max_len:
            chunks.append("\n".join(current))
            current = [line]
            current_len = line_len
        else:
            current.append(line)
            current_len += line_len

    if current:
        chunks.append("\n".join(current))

    for idx, chunk in enumerate(chunks, start=1):
        prefix = f"[Part {idx}/{len(chunks)}]\n" if len(chunks) > 1 else ""
        payload = {"chat_id": chat_id, "text": prefix + chunk}
        resp = requests.post(
            f"{TELEGRAM_API_BASE}/bot{bot_token}/sendMessage",
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()


def main() -> None:
    configure_logging()
    cfg = load_config()
    gemini_limiter = GeminiRateLimiter(cfg["gemini_rpm"])
    logger.info(
        "run_start channels=%s lookback_hours=%s max_total_videos=%s yt_delay_ms=%s "
        "gemini_rpm=%s gemini_interval_s=%.1f video_model=%s digest_model=%s",
        len(cfg["channel_ids"]),
        cfg["lookback_hours"],
        cfg["max_total_videos"],
        cfg["youtube_channel_delay_ms"],
        cfg["gemini_rpm"],
        gemini_limiter.min_interval_s,
        cfg["video_summary_model"],
        cfg["digest_model"],
    )

    videos = fetch_recent_videos(
        youtube_api_key=cfg["youtube_api_key"],
        channel_ids=cfg["channel_ids"],
        lookback_hours=cfg["lookback_hours"],
        max_videos_per_channel=cfg["max_videos_per_channel"],
        youtube_channel_delay_ms=cfg["youtube_channel_delay_ms"],
        youtube_max_retries=cfg["youtube_max_retries"],
        llm_backoff_base_seconds=cfg["llm_backoff_base_seconds"],
    )

    if not videos:
        send_telegram_message(
            cfg["telegram_bot_token"],
            cfg["telegram_chat_id"],
            f"No new videos found in last {cfg['lookback_hours']}h for configured channels.",
        )
        print("No videos found; sent no-update message.")
        logger.info("run_end status=no_videos")
        return

    logger.info("videos_fetched total=%s", len(videos))

    signal_videos, dropped_videos = split_videos_by_signal(
        videos,
        min_duration_seconds=cfg["min_duration_seconds"],
        blocked_keywords=cfg["noise_title_keywords"],
    )
    videos = signal_videos[: cfg["max_total_videos"]]

    if not videos:
        send_telegram_message(
            cfg["telegram_bot_token"],
            cfg["telegram_chat_id"],
            "All recent videos were filtered as low-signal/noise by current rules.",
        )
        print("All videos dropped after noise filtering.")
        logger.info("run_end status=all_filtered")
        return

    logger.info("videos_selected kept=%s dropped=%s", len(videos), len(dropped_videos))

    per_video = {}
    for v in videos:
        try:
            per_video[v.video_id] = summarize_video(
                video=v,
                model=cfg["video_summary_model"],
                fallback_model=cfg["video_summary_fallback_model"],
                gemini_api_key=cfg["gemini_api_key"],
                transcript_languages=cfg["transcript_languages"],
                max_description_chars=cfg["max_description_chars"],
                max_transcript_chars=cfg["max_transcript_chars"],
                llm_max_retries=cfg["llm_max_retries"],
                llm_backoff_base_seconds=cfg["llm_backoff_base_seconds"],
                video_max_output_tokens=cfg["video_max_output_tokens"],
                min_bullets_per_video_summary=cfg["min_bullets_per_video_summary"],
                rate_limiter=gemini_limiter,
            )
        except Exception as exc:
            safe_error = redact_secrets(str(exc))
            code = classify_error(exc)
            per_video[v.video_id] = f"- Summary unavailable ({code}): {safe_error[:180]}"
            logger.error("summary_failed code=%s detail=%s video=%s", code, safe_error[:220], video_label(v))
        sleep_ms(cfg["gemini_call_delay_ms"])

    raw_digest = build_digest(videos, per_video, cfg["lookback_hours"])

    try:
        sleep_ms(cfg["gemini_final_delay_ms"])
        final_digest = build_final_editorial_digest(
            digest_text=raw_digest,
            model=cfg["digest_model"],
            fallback_model=cfg["digest_fallback_model"],
            gemini_api_key=cfg["gemini_api_key"],
            llm_max_retries=cfg["llm_max_retries"],
            llm_backoff_base_seconds=cfg["llm_backoff_base_seconds"],
            final_max_output_tokens=cfg["final_max_output_tokens"],
            min_bullets_for_final_digest=cfg["min_bullets_for_final_digest"],
            rate_limiter=gemini_limiter,
        )
        dropped_line = (
            f"\n\nFiltered out low-signal videos: {len(dropped_videos)}"
            if dropped_videos
            else ""
        )
        final_text = (
            "🧠 Final Curated Briefing\n\n"
            + final_digest
            + dropped_line
            + "\n\n---\nRaw per-video notes below:\n\n"
            + raw_digest
        )
    except Exception as exc:
        safe_error = redact_secrets(str(exc))
        code = classify_error(exc)
        logger.error("editorial_failed code=%s detail=%s", code, safe_error[:220])
        final_text = (
            f"⚠️ Final editorial synthesis skipped ({code}).\n"
            f"Reason: {safe_error[:220]}\n\n"
            "Sending per-video digest instead:\n\n"
            + raw_digest
        )

    send_telegram_message(
        cfg["telegram_bot_token"],
        cfg["telegram_chat_id"],
        final_text,
    )
    transcript_backed = sum(1 for video in videos if video.summary_source == "transcript")
    print(
        f"Success. Sent digest for {len(videos)} videos. "
        f"Transcript-backed summaries: {transcript_backed}. "
        f"Dropped noise videos: {len(dropped_videos)}."
    )
    logger.info(
        "run_end status=success sent=%s transcript_backed=%s dropped=%s",
        len(videos),
        transcript_backed,
        len(dropped_videos),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        configure_logging()
        safe_error = redact_secrets(str(exc))
        code = classify_error(exc)
        logger.exception("run_failed code=%s detail=%s", code, safe_error[:220])
        print(f"Run failed ({code}): {safe_error[:220]}")
        raise SystemExit(1) from None
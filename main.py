import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import requests
from dotenv import load_dotenv


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
TELEGRAM_API_BASE = "https://api.telegram.org"


@dataclass
class Video:
    video_id: str
    title: str
    description: str
    published_at: str
    channel_id: str
    channel_title: str
    url: str


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_config() -> Dict:
    load_dotenv()
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
        "max_videos_per_channel": env_int("MAX_VIDEOS_PER_CHANNEL", 5),
        "max_total_videos": env_int("MAX_TOTAL_VIDEOS", 20),
        "video_summary_model": os.getenv("VIDEO_SUMMARY_MODEL", "gemini-2.5-flash").strip(),
        "digest_model": os.getenv("DIGEST_MODEL", "gemini-2.5-pro").strip(),
    }


def fetch_channel_titles(youtube_api_key: str, channel_ids: List[str]) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for i in range(0, len(channel_ids), 50):
        chunk = channel_ids[i : i + 50]
        resp = requests.get(
            f"{YOUTUBE_API_BASE}/channels",
            params={
                "part": "snippet",
                "id": ",".join(chunk),
                "key": youtube_api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for item in data.get("items", []):
            cid = item.get("id", "")
            title = item.get("snippet", {}).get("title", "Unknown Channel")
            titles[cid] = title
    return titles


def fetch_recent_videos(
    youtube_api_key: str,
    channel_ids: List[str],
    lookback_hours: int,
    max_videos_per_channel: int,
) -> List[Video]:
    published_after = (
        datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    channel_titles = fetch_channel_titles(youtube_api_key, channel_ids)
    videos: List[Video] = []

    for channel_id in channel_ids:
        resp = requests.get(
            f"{YOUTUBE_API_BASE}/search",
            params={
                "part": "snippet",
                "channelId": channel_id,
                "type": "video",
                "order": "date",
                "publishedAfter": published_after,
                "maxResults": max_videos_per_channel,
                "key": youtube_api_key,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if not vid:
                continue
            snippet = item.get("snippet", {})
            videos.append(
                Video(
                    video_id=vid,
                    title=snippet.get("title", "").strip(),
                    description=snippet.get("description", "").strip(),
                    published_at=snippet.get("publishedAt", ""),
                    channel_id=channel_id,
                    channel_title=channel_titles.get(channel_id, snippet.get("channelTitle", "Unknown Channel")),
                    url=f"https://www.youtube.com/watch?v={vid}",
                )
            )

    # Deduplicate by video_id and sort latest first
    unique: Dict[str, Video] = {v.video_id: v for v in videos}
    deduped = list(unique.values())
    deduped.sort(key=lambda v: v.published_at, reverse=True)
    return deduped


def call_gemini(model: str, api_key: str, prompt: str) -> str:
    url = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 700,
        },
    }
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        return "No summary returned."


def summarize_video(video: Video, model: str, gemini_api_key: str) -> str:
    prompt = f"""
You are an analyst making concise daily intel notes.

Summarize this YouTube video in max 4 bullets:
- What happened / key topic
- Most useful insight
- Why it matters (AI/tech/startup/market/dev trends)
- One watchout (optional)

Return plain text bullets only.

Video metadata:
Channel: {video.channel_title}
Title: {video.title}
Published At: {video.published_at}
Description:
{video.description[:3000]}
""".strip()
    return call_gemini(model, gemini_api_key, prompt)


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
        summary = per_video_summaries.get(v.video_id, "No summary.")
        for sline in summary.splitlines():
            sline = sline.strip()
            if not sline:
                continue
            lines.append(f"   {sline}")
        lines.append("")

    return "\n".join(header + lines).strip()


def build_final_editorial_digest(
    digest_text: str, model: str, gemini_api_key: str
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
    return call_gemini(model, gemini_api_key, prompt)


def send_telegram_message(bot_token: str, chat_id: str, text: str) -> None:
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
    cfg = load_config()

    videos = fetch_recent_videos(
        youtube_api_key=cfg["youtube_api_key"],
        channel_ids=cfg["channel_ids"],
        lookback_hours=cfg["lookback_hours"],
        max_videos_per_channel=cfg["max_videos_per_channel"],
    )

    if not videos:
        send_telegram_message(
            cfg["telegram_bot_token"],
            cfg["telegram_chat_id"],
            f"No new videos found in last {cfg['lookback_hours']}h for configured channels.",
        )
        print("No videos found; sent no-update message.")
        return

    videos = videos[: cfg["max_total_videos"]]

    per_video = {}
    for v in videos:
        try:
            per_video[v.video_id] = summarize_video(
                v, cfg["video_summary_model"], cfg["gemini_api_key"]
            )
        except Exception as exc:
            per_video[v.video_id] = f"- Summary failed: {exc}"

    raw_digest = build_digest(videos, per_video, cfg["lookback_hours"])

    try:
        final_digest = build_final_editorial_digest(
            raw_digest, cfg["digest_model"], cfg["gemini_api_key"]
        )
        final_text = (
            "🧠 Final Curated Briefing\n\n"
            + final_digest
            + "\n\n---\nRaw per-video notes below:\n\n"
            + raw_digest
        )
    except Exception as exc:
        final_text = (
            f"⚠️ Final editorial synthesis failed: {exc}\n\n"
            "Sending raw per-video digest instead:\n\n"
            + raw_digest
        )

    send_telegram_message(
        cfg["telegram_bot_token"],
        cfg["telegram_chat_id"],
        final_text,
    )
    print(f"Success. Sent digest for {len(videos)} videos.")


if __name__ == "__main__":
    main()
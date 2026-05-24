# YouTube Digest Agent (Gemini + Telegram + GitHub Actions)

Personal AI agent that:
1. Tracks selected YouTube channels
2. Fetches videos from last 24 hours
3. Summarizes each video with Gemini
4. Sends one digest to Telegram

## 1) Setup locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
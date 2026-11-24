#!/usr/bin/env python3
"""
Blot Outdoors Show - Whaskey & Tasting Review Tracker
Fetches YouTube video metadata & transcripts, caches raw transcripts locally,
and uses Gemini (google-genai SDK Interactions API) to extract structured
tasting notes, ratings, categories, tags, and an editable Markdown transcript excerpt.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi

ROOT_DIR = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = ROOT_DIR / "data" / "transcripts"
REVIEWS_DIR = ROOT_DIR / "src" / "content" / "reviews"

DEFAULT_CHANNEL_URL = "https://www.youtube.com/@BlotOutdoorsShow/videos"
DEFAULT_MODEL = "gemini-4-hydrogen"


class RatedItem(BaseModel):
    name: str = Field(
        description=(
            "Full corrected name of the whisky, beer, drink, food, or item being tasted/reviewed. "
            "Fix YouTube auto-caption phonetic errors of Scottish distilleries (e.g., Ledaig, "
            "Bunnahabhain, Laphroaig, Bruichladdich, Glenfarclas, Auchentoshan, Ardnamurchan)."
        )
    )
    category: Literal[
        "Whisky", "Beer & Ale", "Spirits & Liqueur", "Food & Scran", "Other"
    ] = Field(
        description="Category of the rated item. Use 'Whisky' for all Scotch/whisky/whiskey/bourbon."
    )
    brand_or_distillery: str = Field(
        description="Distillery, brewery, or brand name (e.g., 'Tobermory', 'Ardbeg', 'BrewDog', 'Macsween'). Leave empty string if unknown."
    )
    region_or_style: str = Field(
        description="Whisky region or style (e.g., 'Islay Single Malt', 'Speyside', 'Highland', 'Blended Malt', 'IPA', 'Campfire Scran')."
    )
    abv: str = Field(
        description="ABV percentage if mentioned or standard for that bottle (e.g., '46.3%', '50%'), or empty string if unknown."
    )
    timestamp_seconds: int = Field(
        description="Start timestamp in seconds in the video where he introduces or tastes this item."
    )
    timestamp_display: str = Field(
        description="Human-readable timestamp in MM:SS or HH:MM:SS format matching timestamp_seconds."
    )
    rating_score: Optional[float] = Field(
        default=None,
        description="Numeric score normalized to a 0-10 scale (e.g. 8.5). Use null if no numeric rating can be inferred.",
    )
    rating_raw: str = Field(
        description="How he stated the score or verdict (e.g., '8.5 out of 10', '9/10 - absolute belter')."
    )
    tags: List[str] = Field(
        description=(
            "4 to 8 lowercase flavor or descriptive tags (e.g., 'peaty', 'smoky', 'sherry cask', "
            "'vanilla', 'coastal', 'sweet', 'spicy', 'smooth', 'oily', 'fruity', 'budget')."
        )
    )
    tasting_notes: str = Field(
        description="1 to 3 sentences summarizing the nose, palate, taste descriptions, and finish he gets from the item."
    )
    verdict_quote: str = Field(
        description="A short, authentic quote from the transcript capturing his reaction and verdict."
    )


class VideoReviewExtraction(BaseModel):
    has_reviews: bool = Field(
        description="True if the host tastes, drinks, or rates any whisky, drink, or food item in the video."
    )
    episode_summary: str = Field(
        description="1-2 sentence summary of where he is camping/exploring and what he drinks/eats."
    )
    items: List[RatedItem] = Field(
        description="List of all drinks, whiskies ('whaskey'), beers, or food items tasted and reviewed in this video."
    )
    cleaned_transcript_excerpt: str = Field(
        description=(
            "Markdown-formatted transcript excerpt covering the tasting & rating segments. "
            "Include **[MM:SS]** bold timestamps at the start of paragraphs. Fix obvious auto-caption "
            "garbling of whisky names while preserving his authentic Scottish phrasing ('whaskey', 'wee dram', 'aye')."
        )
    )


def format_seconds(seconds: float) -> str:
    total = int(seconds)
    hrs, rem = divmod(total, 3600)
    mins, secs = divmod(rem, 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def run_yt_dlp_json(args: List[str]) -> List[dict]:
    yt_dlp_bin = str(ROOT_DIR / ".venv" / "bin" / "yt-dlp")
    if not Path(yt_dlp_bin).exists():
        yt_dlp_bin = "yt-dlp"
    cmd = [yt_dlp_bin, "--no-warnings"] + args
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    items = []
    for line in res.stdout.strip().splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def discover_videos(channel_url: str, limit: int) -> List[str]:
    print(f"🔎 Fetching latest {limit} videos from {channel_url} ...")
    entries = run_yt_dlp_json(
        ["--flat-playlist", "--dump-json", "--playlist-end", str(limit), channel_url]
    )
    video_ids = [e["id"] for e in entries if e.get("id")]
    print(f"   Found {len(video_ids)} videos: {', '.join(video_ids)}")
    return video_ids


def fetch_video_metadata(video_id: str) -> dict:
    meta_cache = TRANSCRIPTS_DIR / f"{video_id}.meta.json"
    if meta_cache.exists():
        return json.loads(meta_cache.read_text(encoding="utf-8"))

    url = f"https://www.youtube.com/watch?v={video_id}"
    entries = run_yt_dlp_json(["--dump-json", "--skip-download", url])
    if not entries:
        raise RuntimeError(f"Could not fetch metadata for {video_id}")
    raw = entries[0]
    upload_date = raw.get("upload_date", "")
    if len(upload_date) == 8:
        publish_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"
    else:
        publish_date = datetime.date.today().isoformat()

    meta = {
        "video_id": video_id,
        "title": raw.get("title", f"Video {video_id}"),
        "publish_date": publish_date,
        "duration": raw.get("duration", 0),
        "description": raw.get("description", ""),
        "thumbnail": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
        "channel": raw.get("uploader", "Blot Outdoors Show"),
    }
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    meta_cache.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def fetch_or_load_transcript(video_id: str) -> str:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    json_path = TRANSCRIPTS_DIR / f"{video_id}.json"

    if txt_path.exists():
        print(f"   📄 Loaded cached transcript: {txt_path.relative_to(ROOT_DIR)}")
        return txt_path.read_text(encoding="utf-8")

    print(f"   ⏬ Downloading transcript for {video_id} ...")
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=["en", "en-GB", "en-US"])
        snippets = [
            {"start": s.start, "duration": s.duration, "text": s.text}
            for s in fetched
        ]
    except Exception as exc:
        print(f"   ⚠️ Primary transcript fetch failed ({exc}), listing available transcripts...")
        t_list = api.list(video_id)
        transcript_obj = next(iter(t_list))
        fetched = transcript_obj.fetch()
        snippets = [
            {"start": s.start, "duration": s.duration, "text": s.text}
            for s in fetched
        ]

    json_path.write_text(json.dumps(snippets, indent=2), encoding="utf-8")

    # Format into readable timestamped lines grouped roughly every ~15 seconds
    lines = []
    bucket_text = []
    bucket_start = None
    for seg in snippets:
        text_clean = seg["text"].replace("\n", " ").strip()
        if not text_clean:
            continue
        if bucket_start is None:
            bucket_start = seg["start"]
        bucket_text.append(text_clean)
        if seg["start"] - bucket_start >= 15:
            ts = format_seconds(bucket_start)
            lines.append(f"[{ts} ({int(bucket_start)}s)] {' '.join(bucket_text)}")
            bucket_text = []
            bucket_start = None

    if bucket_text and bucket_start is not None:
        ts = format_seconds(bucket_start)
        lines.append(f"[{ts} ({int(bucket_start)}s)] {' '.join(bucket_text)}")

    formatted = "\n".join(lines)
    txt_path.write_text(formatted, encoding="utf-8")
    print(f"   💾 Saved editable raw transcript to {txt_path.relative_to(ROOT_DIR)}")
    return formatted


def is_manually_edited(md_path: Path) -> bool:
    if not md_path.exists():
        return False
    content = md_path.read_text(encoding="utf-8")
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
                return bool(frontmatter and frontmatter.get("manually_edited") is True)
            except Exception:
                return False
    return False


def extract_reviews_with_gemini(
    client: genai.Client, model: str, meta: dict, transcript_text: str
) -> VideoReviewExtraction:
    prompt = f"""You are analyzing an episode of the 'Blot Outdoors Show' on YouTube.
The Scottish host goes wild camping, bothy staying, or exploring outdoors, and almost always tastes and rates 'whaskey' (whisky), beers/ales, spirits, or camp food ('scran'), giving flavour descriptions and a score (usually out of 10).

Note: YouTube's automatic speech-to-text often struggles with thick Scottish accents and Scottish distillery names (e.g., 'whaskey' = whisky, 'Ledaig', 'Bunnahabhain', 'Bruichladdich', 'Laphroaig', 'Ardnamurchan', 'Glen Scotia', 'Springbank', 'Kilkerran', 'Craigellachie', 'Tamdhu', 'Edradour', 'Deanston', 'Arran').
Use context from the Video Title, Description, and phonetic clues in the transcript to identify the exact bottle, brewery, or item name.

VIDEO METADATA:
- Video ID: {meta['video_id']}
- Title: {meta['title']}
- Publish Date: {meta['publish_date']}
- Description:
{meta.get('description', '')[:3000]}

TIMESTAMPED TRANSCRIPT:
{transcript_text}

Extract every whisky, drink, or food item that the host tastes, describes, or rates in this episode, along with a cleaned-up Markdown transcript excerpt of the tasting segment(s)."""

    interaction = client.interactions.create(
        model=model,
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": VideoReviewExtraction.model_json_schema(),
        },
    )
    return VideoReviewExtraction.model_validate_json(interaction.output_text)


def write_review_markdown(
    meta: dict, extraction: VideoReviewExtraction, md_path: Path
) -> None:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    frontmatter = {
        "video_id": meta["video_id"],
        "video_title": meta["title"],
        "publish_date": meta["publish_date"],
        "thumbnail": meta["thumbnail"],
        "has_reviews": extraction.has_reviews,
        "manually_edited": False,
        "episode_summary": extraction.episode_summary,
        "items": [item.model_dump() for item in extraction.items],
    }
    yaml_str = yaml.dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()

    body = extraction.cleaned_transcript_excerpt.strip()
    if not body:
        body = "_No tasting transcript excerpt extracted for this episode._"

    md_content = f"---\n{yaml_str}\n---\n\n{body}\n"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"   ✅ Wrote review markdown: {md_path.relative_to(ROOT_DIR)}")


def extract_video_id(raw: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", raw)
    if m:
        return m.group(1)
    return raw.strip()


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    parser = argparse.ArgumentParser(
        description="Fetch Blot Outdoors Show transcripts and extract whisky/item ratings with Gemini."
    )
    parser.add_argument(
        "--channel",
        default=DEFAULT_CHANNEL_URL,
        help="YouTube channel videos URL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of latest videos to process from the channel (default: 5)",
    )
    parser.add_argument(
        "--video",
        nargs="*",
        help="Specific YouTube video ID(s) or URL(s) to process instead of scanning the channel",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Gemini model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run Gemini extraction even if the markdown file already exists (still skips manually_edited: true)",
    )
    parser.add_argument(
        "--transcripts-only",
        action="store_true",
        help="Only download metadata and raw transcripts without calling Gemini API",
    )
    args = parser.parse_args()

    if args.video:
        video_ids = [extract_video_id(v) for v in args.video]
    else:
        video_ids = discover_videos(args.channel, args.limit)

    client = None
    if not args.transcripts_only:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print(
                "\n❌ GEMINI_API_KEY is not set! Add it to .env or export GEMINI_API_KEY='your_key'.\n"
                "   Tip: You can still run with --transcripts-only to test fetching YouTube transcripts.",
                file=sys.stderr,
            )
            sys.exit(1)
        client = genai.Client(api_key=api_key)

    for idx, vid in enumerate(video_ids, 1):
        print(f"\n[{idx}/{len(video_ids)}] Processing video: {vid}")
        md_path = REVIEWS_DIR / f"{vid}.md"

        if is_manually_edited(md_path):
            print(f"   🔒 Skipping {vid}: marked as manually_edited: true")
            continue

        if md_path.exists() and not args.force and not args.transcripts_only:
            print(f"   ⏭️  Skipping {vid}: {md_path.relative_to(ROOT_DIR)} already exists (use --force to re-extract)")
            continue

        try:
            meta = fetch_video_metadata(vid)
            print(f"   🎬 Title: {meta['title']} ({meta['publish_date']})")
            transcript_text = fetch_or_load_transcript(vid)

            if args.transcripts_only:
                continue

            assert client is not None
            print(f"   🤖 Extracting tasting notes & ratings with {args.model} ...")
            extraction = extract_reviews_with_gemini(
                client, args.model, meta, transcript_text
            )
            print(
                f"   🥃 Extracted {len(extraction.items)} rated item(s): "
                + (", ".join(f"{i.name} ({i.rating_raw})" for i in extraction.items) or "None")
            )
            write_review_markdown(meta, extraction, md_path)
        except Exception as err:
            print(f"   ❌ Error processing {vid}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()

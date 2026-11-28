#!/usr/bin/env python3
"""
Verify and correct garbled drink/brand names in existing review markdown files
using Gemini multimodal inspection (video frames + transcript clues).

Examples:
    .venv/bin/python -u scripts/verify_names.py --video C3HUpzvFK0s --dry-run
    .venv/bin/python -u scripts/verify_names.py --limit 25 --delay 2 --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
REVIEWS_DIR = ROOT_DIR / "src" / "content" / "reviews"
TRANSCRIPTS_DIR = ROOT_DIR / "data" / "transcripts"

DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODEL = "gemini-3.5-flash-lite"
DEFAULT_FALLBACK_PROXY = "socks5://127.0.0.1:1080"
GEMINI_CALL_TIMEOUT_SEC = 120
DEFAULT_CATEGORIES = ["Whisky", "Beer & Ale", "Spirits & Liqueur"]


class ItemVerification(BaseModel):
    index: int = Field(description="0-based index of the item in the input list.")
    changed: bool = Field(
        description="True only if the existing name or brand_or_distillery is a speech-to-text error or wrong spelling and needs correction."
    )
    corrected_name: str = Field(
        description="Verified full bottle/drink name (keep original if no change needed)."
    )
    corrected_brand_or_distillery: Optional[str] = Field(
        default=None,
        description="Verified brand, distillery, or brewery name (keep original if no change needed).",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the correction based on video frames and transcript clues."
    )
    reason: str = Field(
        description="Brief 1-sentence explanation citing label text seen in frames or transcript clues."
    )


class EpisodeVerificationResult(BaseModel):
    items: list[ItemVerification] = Field(default_factory=list)


class BottleWindow(BaseModel):
    index: int = Field(description="0-based index of the item in the input list.")
    start_seconds: int = Field(
        description="Start timestamp (seconds) where the host first shows, introduces, or reads the label of the bottle/can."
    )
    end_seconds: int = Field(
        description="End timestamp (seconds), typically 30-45 seconds after start_seconds."
    )
    reason: str = Field(description="Brief note on which chapter or transcript line indicates the bottle is shown.")


class BottleWindowResult(BaseModel):
    windows: list[BottleWindow] = Field(default_factory=list)


def is_proxy_reachable(proxy_url: Optional[str]) -> bool:
    if not proxy_url:
        return False
    parsed = urlparse(proxy_url)
    host = parsed.hostname
    port = parsed.port
    if not host or not port:
        return False
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def get_yt_dlp_bin() -> str:
    venv_bin = ROOT_DIR / ".venv" / "bin" / "yt-dlp"
    return str(venv_bin) if venv_bin.exists() else "yt-dlp"


def load_video_meta(video_id: str) -> dict:
    meta_path = TRANSCRIPTS_DIR / f"{video_id}.meta.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def load_or_fetch_chapters(video_id: str, proxy: Optional[str] = None) -> list[dict]:
    """Return YouTube chapters for video_id, caching them in .meta.json if available."""
    meta_path = TRANSCRIPTS_DIR / f"{video_id}.meta.json"
    meta = load_video_meta(video_id)
    if "chapters" in meta:
        return meta["chapters"] or []

    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [get_yt_dlp_bin(), "--no-warnings", "--dump-json", "--skip-download"]
    if proxy:
        cmd.extend(["--proxy", proxy])
    cmd.append(url)

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        if res.returncode == 0 and res.stdout.strip():
            info = json.loads(res.stdout.splitlines()[0])
            chapters = info.get("chapters") or []
            if meta_path.exists():
                meta["chapters"] = chapters
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return chapters
    except Exception as err:
        print(f"   ⚠️  Could not fetch chapters for {video_id}: {err}")
    return []


def compute_capture_window(
    timestamp_seconds: int, chapters: list[dict], fallback_before: int = 120, fallback_after: int = 30
) -> tuple[int, int]:
    """Pick [start, end] window around the review timestamp using chapters when available."""
    ts = max(0, int(timestamp_seconds or 0))
    for ch in chapters:
        start = int(ch.get("start_time", 0))
        end = int(ch.get("end_time", 0))
        if start <= ts <= end:
            # Bottle is often shown near chapter start; cover from chapter start through ts+20s (capped at 180s span).
            win_start = max(0, start)
            win_end = min(end, max(win_start + 90, ts + 20))
            if win_end - win_start > 180:
                win_start = max(win_start, ts - 120)
                win_end = min(end, ts + 30)
            return win_start, max(win_start + 15, win_end)
    return max(0, ts - fallback_before), ts + fallback_after


def extract_frames_for_window(
    video_id: str,
    start_sec: int,
    end_sec: int,
    max_frames: int = 8,
    proxy: Optional[str] = None,
) -> list[bytes]:
    """Download only the [start_sec, end_sec] clip via yt-dlp and extract evenly spaced JPEG frames."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    duration = max(10, end_sec - start_sec)
    step = max(3, duration // max_frames)

    with tempfile.TemporaryDirectory(prefix=f"blot_frames_{video_id}_") as tmpdir:
        tmp = Path(tmpdir)
        clip_tmpl = str(tmp / "clip.%(ext)s")
        cmd = [
            get_yt_dlp_bin(),
            "--no-warnings",
            "--quiet",
            "--download-sections",
            f"*{start_sec}-{end_sec}",
            "-f",
            "best[height<=480][ext=mp4]/best[height<=480]/bestvideo[height<=480]",
            "-o",
            clip_tmpl,
        ]
        if proxy:
            cmd.extend(["--proxy", proxy])
        cmd.append(url)

        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        except subprocess.TimeoutExpired:
            print(f"   ⚠️  yt-dlp clip download timed out for {video_id} ({start_sec}s-{end_sec}s)")
            return []
        clips = list(tmp.glob("clip.*"))
        if res.returncode != 0 or not clips:
            return []

        clip_path = clips[0]
        frame_tmpl = str(tmp / "frame_%02d.jpg")
        ff_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(clip_path),
            "-vf",
            f"fps=1/{step},scale=640:-1",
            "-frames:v",
            str(max_frames),
            "-q:v",
            "4",
            frame_tmpl,
        ]
        subprocess.run(ff_cmd, capture_output=True, timeout=45)
        frames = sorted(tmp.glob("frame_*.jpg"))
        return [f.read_bytes() for f in frames[:max_frames]]


def load_full_transcript(video_id: str, fallback_body: str) -> str:
    """Load full timestamped transcript if cached, else fall back to review excerpt."""
    txt_path = TRANSCRIPTS_DIR / f"{video_id}.txt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8")
    return fallback_body


def locate_bottle_windows(
    client: genai.Client,
    model: str,
    video_id: str,
    chapters: list[dict],
    transcript_text: str,
    items: list[dict],
    target_indices: list[int],
) -> dict[int, tuple[int, int, str]]:
    """Separate pass: use chapters + full transcript to pinpoint when the host shows each bottle."""
    chapters_str = (
        "\n".join(
            f"- [{int(c.get('start_time', 0))}s - {int(c.get('end_time', 0))}s] {c.get('title', '')}"
            for c in chapters
        )
        or "(no chapters)"
    )
    items_str = "\n".join(
        f"- index {i}: {items[i].get('name')} (tasting timestamp: {items[i].get('timestamp_seconds')}s / {items[i].get('timestamp_display')})"
        for i in target_indices
    )
    prompt = f"""You are locating the exact video timestamps where the host of 'Blot Outdoors Show' FIRST shows or holds up each drink bottle/can to the camera.
Note: The tasting/rating timestamp is often AFTER he first showed the bottle (he may introduce the bottle at the start of a chapter, when unpacking, or when reading the ABV/label).

Find a tight 30-40 second window [start_seconds, end_seconds] for each item where he is most likely holding the bottle up to the camera.
IMPORTANT: Transcript lines look like `[39:43 (2383s)]`. Always use the integer seconds in parentheses (e.g. `2383`), NEVER concatenate `MM:SS` into `3943`.

CHAPTERS:
{chapters_str}

ITEMS:
{items_str}

FULL TIMESTAMPED TRANSCRIPT:
{transcript_text[:28000]}
"""
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=BottleWindowResult,
    )
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        parsed = BottleWindowResult.model_validate_json(response.text)
        out: dict[int, tuple[int, int, str]] = {}
        for w in parsed.windows:
            start = max(0, int(w.start_seconds))
            end = max(start + 25, min(start + 45, int(w.end_seconds)))
            out[w.index] = (start, end, w.reason)
        return out
    except Exception as err:
        print(f"   ⚠️  Bottle-window locator failed for {video_id}: {err}")
        return {}


def parse_review_markdown(md_path: Path) -> tuple[dict, str]:
    """Split a review markdown file into YAML frontmatter dict and body string."""
    text = md_path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid frontmatter in {md_path}")
    frontmatter = yaml.safe_load(parts[1]) or {}
    body = parts[2].strip()
    return frontmatter, body


def write_review_markdown_raw(md_path: Path, frontmatter: dict, body: str) -> None:
    """Write updated frontmatter and body back to a review markdown file."""
    yaml_str = yaml.dump(
        frontmatter,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    ).strip()
    body_out = body if body else "_No tasting transcript excerpt extracted for this episode._"
    md_path.write_text(f"---\n{yaml_str}\n---\n\n{body_out}\n", encoding="utf-8")


def verify_episode_items(
    client: genai.Client,
    model: str,
    frontmatter: dict,
    body: str,
    target_indices: list[int],
    frames_by_item: dict[int, list[bytes]],
) -> EpisodeVerificationResult:
    """Ask Gemini (with optional video frames) to verify bottle/brand names for selected items."""
    items = frontmatter.get("items") or []
    items_block = []
    for idx in target_indices:
        it = items[idx]
        frame_count = len(frames_by_item.get(idx, []))
        items_block.append(
            f"- index: {idx}\n"
            f"  name: {it.get('name')}\n"
            f"  category: {it.get('category')}\n"
            f"  brand_or_distillery: {it.get('brand_or_distillery')}\n"
            f"  region_or_style: {it.get('region_or_style')}\n"
            f"  abv: {it.get('abv')}\n"
            f"  timestamp_display: {it.get('timestamp_display')}\n"
            f"  tasting_notes: {it.get('tasting_notes')}\n"
            f"  attached_frames: {frame_count}"
        )

    prompt = f"""You are fact-checking drink/bottle names extracted from an episode of the 'Blot Outdoors Show' (UK YouTube channel).
The host has a strong Geordie accent, and YouTube auto-captions often mishear distillery, brewery, or supermarket own-brand names phonetically (for example, 'Inverglen' transcribed as 'In the Glen').

Inspect the attached video screenshots (if any) for bottle/can labels, distillery names, and age statements, and cross-check with the transcript excerpt clues (age statement, style, ABV, or retailer like Marks & Spencer, Aldi, Lidl, Tesco, etc.).

Rules:
- Prefer text visibly printed on the bottle/can label in the screenshots over speech-to-text transcripts.
- Only set changed=true if you are confident there is a transcription/spelling/identification error and you can name the exact real product.
- Preserve existing age statements, bottle expressions, or descriptors unless they are clearly wrong.
- If the existing name is already accurate or the frames/transcript do not give a confident correction, set changed=false and return the original values.

EPISODE METADATA:
- Video ID: {frontmatter.get('video_id')}
- Video Title: {frontmatter.get('video_title')}
- Publish Date: {frontmatter.get('publish_date')}

ITEMS TO VERIFY:
{chr(10).join(items_block)}

TRANSCRIPT EXCERPT:
{body[:4000]}
"""

    contents: list = [prompt]
    for idx in target_indices:
        frames = frames_by_item.get(idx, [])
        if frames:
            contents.append(f"Screenshots for item index {idx} ({items[idx].get('name')}):")
            for jpg_bytes in frames:
                contents.append(types.Part.from_bytes(data=jpg_bytes, mime_type="image/jpeg"))

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=EpisodeVerificationResult,
    )

    max_retries = 4
    for attempt in range(1, max_retries + 1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=config,
            )
            response = future.result(timeout=GEMINI_CALL_TIMEOUT_SEC)
            return EpisodeVerificationResult.model_validate_json(response.text)
        except Exception as err:
            err_str = str(err)
            is_timeout = isinstance(err, (TimeoutError, concurrent.futures.TimeoutError))
            daily_quota = "GenerateRequestsPerDay" in err_str or "per day" in err_str.lower()
            transient = (
                is_timeout
                or any(
                    s in err_str
                    for s in (
                        "429",
                        "503",
                        "504",
                        "RESOURCE_EXHAUSTED",
                        "service_unavailable",
                        "high demand",
                        "DEADLINE_EXCEEDED",
                    )
                )
                or "quota" in err_str.lower()
            )
            if daily_quota and model != FALLBACK_MODEL:
                print(f"   🔁 Daily quota hit on {model}, switching to {FALLBACK_MODEL}")
                model = FALLBACK_MODEL
                continue
            if transient and attempt < max_retries:
                wait_sec = 20 * attempt
                label = f"Timed out after {GEMINI_CALL_TIMEOUT_SEC}s" if is_timeout else "Transient API error"
                print(f"   ⏳ {label} (attempt {attempt}/{max_retries}), sleeping {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    raise RuntimeError("Unreachable")


def main() -> int:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found in .env or environment.", file=sys.stderr)
        return 1

    parser = argparse.ArgumentParser(
        description="Verify drink/brand names in existing review markdown files using Gemini + video screenshots."
    )
    parser.add_argument("--video", action="append", help="Specific video ID to verify (repeatable).")
    parser.add_argument(
        "--category",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help=f"Categories to verify (default: {DEFAULT_CATEGORIES}).",
    )
    parser.add_argument("--all-categories", action="store_true", help="Verify items across all categories.")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N review files.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N review files.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds to sleep between API calls.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Gemini model (default: {DEFAULT_MODEL}).")
    parser.add_argument(
        "--min-confidence",
        choices=["high", "medium"],
        default="high",
        help="Minimum confidence required to apply a rename (default: high).",
    )
    parser.add_argument("--proxy", default=None, help="Optional SOCKS5/HTTP proxy for yt-dlp (e.g. socks5://127.0.0.1:1080).")
    parser.add_argument("--no-screenshots", action="store_true", help="Skip downloading video frames (text-only mode).")
    parser.add_argument("--max-frames", type=int, default=8, help="Max screenshots per item (default: 8).")
    parser.add_argument("--dry-run", action="store_true", help="Print proposed changes without writing files.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-verify files already marked names_verified: true (still skips manually_edited: true).",
    )
    args = parser.parse_args()

    proxy = args.proxy
    if not proxy and is_proxy_reachable(DEFAULT_FALLBACK_PROXY):
        proxy = DEFAULT_FALLBACK_PROXY

    if args.video:
        files = [REVIEWS_DIR / f"{vid}.md" for vid in args.video]
    else:
        files = sorted(REVIEWS_DIR.glob("*.md"))
        files = files[args.offset :]
        if args.limit is not None:
            files = files[: args.limit]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=GEMINI_CALL_TIMEOUT_SEC * 1000),
    )

    allowed_conf = {"high"} if args.min_confidence == "high" else {"high", "medium"}
    checked = 0
    updated_files = 0
    updated_items = 0

    for idx, md_path in enumerate(files, 1):
        if not md_path.exists():
            print(f"[{idx}/{len(files)}] ⚠️  Missing file: {md_path.name}")
            continue

        frontmatter, body = parse_review_markdown(md_path)
        vid = frontmatter.get("video_id", md_path.stem)

        if frontmatter.get("manually_edited") is True:
            print(f"[{idx}/{len(files)}] 🔒 Skipping {vid}: manually_edited: true")
            continue
        if not frontmatter.get("has_reviews"):
            continue
        if not args.force and frontmatter.get("names_verified") is True:
            continue

        items = frontmatter.get("items") or []
        target_indices = [
            i
            for i, it in enumerate(items)
            if args.all_categories or it.get("category") in args.category
        ]
        if not target_indices:
            continue

        frames_by_item: dict[int, list[bytes]] = {}
        if not args.no_screenshots:
            chapters = load_or_fetch_chapters(vid, proxy=proxy)
            duration_sec = int(load_video_meta(vid).get("duration") or 0)
            full_transcript = load_full_transcript(vid, body)
            located = locate_bottle_windows(
                client, args.model, vid, chapters, full_transcript, items, target_indices
            )
            for i in target_indices:
                ts = int(items[i].get("timestamp_seconds") or 0)
                used_fallback = False
                if i in located:
                    win_start, win_end, loc_reason = located[i]
                    if duration_sec and win_start >= duration_sec:
                        win_start, win_end = compute_capture_window(ts, chapters)
                        loc_note = " (fallback window: locator past video end)"
                        used_fallback = True
                    else:
                        loc_note = f" ({loc_reason})"
                else:
                    win_start, win_end = compute_capture_window(ts, chapters)
                    loc_note = " (fallback window)"
                    used_fallback = True
                frames = extract_frames_for_window(
                    vid, win_start, win_end, max_frames=args.max_frames, proxy=proxy
                )
                if not frames and not used_fallback:
                    win_start, win_end = compute_capture_window(ts, chapters)
                    frames = extract_frames_for_window(
                        vid, win_start, win_end, max_frames=args.max_frames, proxy=proxy
                    )
                    loc_note = " (fallback window: locator returned 0 frames)"
                frames_by_item[i] = frames
                print(
                    f"   📸 {vid} item {i} (tasting {items[i].get('timestamp_display', ts)}): "
                    f"bottle window {win_start}s-{win_end}s -> {len(frames)} frame(s){loc_note}"
                )

        print(f"[{idx}/{len(files)}] 🔍 Verifying {vid} ({len(target_indices)} item(s))...")
        result = verify_episode_items(client, args.model, frontmatter, body, target_indices, frames_by_item)
        checked += 1

        file_changed = False
        for ver in result.items:
            if not (0 <= ver.index < len(items)):
                continue
            orig = items[ver.index]
            if ver.changed and ver.confidence in allowed_conf:
                old_name = orig.get("name")
                old_brand = orig.get("brand_or_distillery")
                new_brand = ver.corrected_brand_or_distillery or old_brand
                if ver.corrected_name != old_name or new_brand != old_brand:
                    print(
                        f"   ✏️  [{ver.confidence}] {old_name!r} -> {ver.corrected_name!r}"
                        f" (brand: {old_brand!r} -> {new_brand!r})\n"
                        f"       reason: {ver.reason}"
                    )
                    orig["name"] = ver.corrected_name
                    if ver.corrected_brand_or_distillery:
                        orig["brand_or_distillery"] = ver.corrected_brand_or_distillery
                    file_changed = True
                    updated_items += 1
            elif ver.changed:
                print(
                    f"   ⏭️  Skipped low-confidence ({ver.confidence}) suggestion for {orig.get('name')!r}: "
                    f"{ver.corrected_name!r} ({ver.reason})"
                )

        if not args.dry_run:
            frontmatter["names_verified"] = True
            write_review_markdown_raw(md_path, frontmatter, body)
            if file_changed:
                updated_files += 1
        elif file_changed:
            updated_files += 1

        if idx < len(files) and args.delay > 0:
            time.sleep(args.delay)

    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"\n✅ {mode}: checked {checked} file(s), {updated_items} item(s) updated across {updated_files} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

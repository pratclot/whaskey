# AGENTS.md — Blot Outdoors Show Tracker (`pratclot/whaskey`)

## Overview
Automated tracker and Astro static site for the **Blot Outdoors Show** (`@BlotOutdoorsShow`) on YouTube. Extracts rated whiskies ("whaskey"), beers/ales, spirits, and food ("scran") with tasting notes, scores, flavour tags, timestamps, and editable transcript excerpts.
- **Repo / Live site**: `pratclot/whaskey` → `https://pratclot.github.io/whaskey/`
- **GitHub workflow**: `.github/workflows/deploy.yml` auto-deploys `main` to GitHub Pages.

## Key Files
- `scripts/tracker.py` — YouTube catalog/transcript fetcher (`yt-dlp` auto-subs → `youtube-transcript-api` → local `faster-whisper` CUDA fallback) + Gemini structured extraction → `src/content/reviews/<video_id>.md`.
- `data/channel_videos.jsonl` — Full channel catalog (641 videos).
- `data/transcripts/<video_id>.txt` & `.meta.json` — Cached raw transcripts and metadata.
- `src/content/reviews/*.md` — Astro Content Collection markdown files (`manually_edited: true` protects manual GitHub edits from being overwritten).
- `src/content.config.ts` — Content collection schema.
- `src/pages/index.astro` — Frontend UI (tabs: `Whisky` default, `All`, `Beer & Ale`, `Spirits & Liqueur`, `Food & Scran`, `Other`; search, sort, flavour tag cloud, transcript modal, "Edit on GitHub" links).
- `.env` — Contains `GEMINI_API_KEY` (gitignored; never print or expose).

## Commands & Environment Quirks
- **Python (requires `BypassSandbox: true` for YouTube + Gemini API)**:
  ```bash
  .venv/bin/python -u scripts/tracker.py --offset 18 --limit 10 --model gemini-3.5-flash-lite
  ```
- **Astro build (use `/usr/bin/node` v26; default `node` in PATH is v18.19.0)**:
  ```bash
  XDG_CONFIG_HOME=/tmp ASTRO_TELEMETRY_DISABLED=1 /usr/bin/node ./node_modules/.bin/astro build
  ```
- **GitHub CLI (`gh`)**: Prefix with `DBUS_SESSION_BUS_ADDRESS=unix:path=/dev/null` to avoid D-Bus keyring hangs.
- **Gemini models & rate limits**: Use `gemini-4-hydrogen` (default) and `gemini-3.5-flash-lite` (fallback) via `client.interactions.create` (`google-genai>=3.5.0`, 120s client timeout). Never use deprecated `gemini-2.5-*`. On free tier, `gemini-4-hydrogen` is capped at 20 req/day (and can return `503` high demand); `tracker.py` switches immediately to `gemini-3.5-flash-lite` on daily quota errors, or pass `--model gemini-3.5-flash-lite` directly.
- **YouTube captions & local `faster-whisper` fallback**: YouTube `/api/timedtext` and `youtube-transcript-api` are rate-limited/IP-blocked (`HTTP 429`). `tracker.py` fetches `en-orig` first (falling back to `en` only if missing) with `--sleep-subtitles 2` to halve subtitle requests and prefer original audio captions. When both fail, `tracker.py` downloads English original audio (`wa[language^=en]...` to avoid YouTube auto-dubs), decodes to 16kHz mono `pcm_f32le` via `ffmpeg` (bypassing `av>=19` `metadata_errors` error), preloads `.venv` `nvidia-cublas-cu12`/`nvidia-cudnn-cu12` libs, and transcribes with `faster-whisper` (`turbo` model, `beam_size=2` + `initial_prompt`).
- **YouTube 429 SOCKS5 proxy workaround**: `remote-host` (`aarch64`) is bot-walled by YouTube (`Sign in to confirm`), but `user@proxy-host` (`x86_64`) serves captions. Open a local SOCKS5 tunnel and pass `--proxy` to `tracker.py` (routes YouTube traffic only; Gemini stays direct):
  ```bash
  ssh -fN -D 1080 user@proxy-host
  .venv/bin/python -u scripts/tracker.py --offset 26 --limit 10 --proxy socks5://127.0.0.1:1080
  ```
  (Note: `yt-dlp` supports `socks5://` natively; `youtube-transcript-api` needs `pysocks` installed in `.venv` for `socks5://`.)
- **Laptop performance mode & Whisper quality**:
  - **NEVER use `small.en`** (or smaller Whisper models): accuracy on Geordie accents and whisky/beer brand names is unacceptable. Always use **`turbo`** (`large-v3-turbo`) with `initial_prompt`.
  - **Always remind the user / ensure the laptop is plugged into AC power and switched to performance mode** before running local `faster-whisper` batches — the GTX 1650 Max-Q (4 GiB VRAM) and CPU (`ffmpeg` decode) throttle clocks/TDP on battery or balanced/power-saver profiles (affects speed, not accuracy).
- **Resetting GTX 1650 Max-Q without reboot** (when `Xid 119` / GSP hang puts GPU into `[GPU requires reset]`; `nvidia-smi --gpu-reset` is unsupported on GeForce, but `i915` drives display so `nvidia` modules have 0 external users):
  ```bash
  sudo modprobe -r nvidia_uvm nvidia_modeset nvidia
  echo 1 | sudo tee /sys/bus/pci/devices/0000:02:00.0/reset
  sudo modprobe nvidia nvidia_uvm
  nvidia-smi
  ```
- **Automation architecture**: Prefer a local laptop `systemd` user timer / `cron` (residential IP + GTX 1650) over GitHub Actions for fetching/transcribing new episodes, then `git push` to trigger `.github/workflows/deploy.yml`.

## Current State
- **Done**: First 126 episodes (`offset=0..125`) extracted and verified with `astro build`.
- **Remaining**: 515 videos (`offset=126..640`).

## Next Steps
1. Process remaining videos in paced batches (e.g. `--offset 126 --limit 50 --delay 5 --model gemini-3.5-flash-lite --no-whisper`) with `BypassSandbox: true` (`tracker.py` auto-falls back to `socks5://127.0.0.1:1080` if reachable and exits with code `2` on unrecoverable `429`).
2. Verify `astro build` passes, then commit and `git push` `src/content/reviews/*.md` to `origin/main`.
3. Optionally set up a local `systemd` user timer/cron to poll new `@BlotOutdoorsShow` uploads, run `tracker.py`, and push to `main`.

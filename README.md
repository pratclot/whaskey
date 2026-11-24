# 🥃 Blot Outdoors Show — Whaskey & Scran Tracker

An automated YouTube transcript scraper, Gemini LLM tasting-notes extractor, and Astro static website for tracking every **whaskey** (whisky), beer/ale, spirit, and campfire/takeaway scran tasted and rated on the **Blot Outdoors Show**.

---

## ✨ Features

- **🥃 "Whaskey Only" & Multi-Category Filtering**: Defaults to **Whaskey Only**, with instant tabs for **All Rated Items**, **Beer & Ale**, **Spirits & Liqueur**, and **Food & Scran**.
- **🔍 Search, Sort & Flavour Tag Cloud**:
  - Sort by **Highest Rating First**, **Lowest Rating First**, **Newest/Oldest Video**, or **Name (A → Z)**.
  - Filter by clickable flavour tags (`#marzipan`, `#sherry`, `#vanilla`, `#cinnamon`, `#toffee`, `#oaky`, `#spicy`, `#hoppy`, etc.).
  - Full-text search across bottle names, distilleries, tasting notes, quotes, and episode titles.
- **▶ Timestamped Video & Editable Transcript Modal**:
  - Every card links directly to the exact second (`timestamp_seconds`) in the YouTube episode where the tasting starts.
  - Includes a cleaned-up, timestamped **Tasting Transcript Excerpt** for the episode.
- **✏️ Direct "Edit on GitHub" Workflow**:
  - Every review is stored as a human-readable Markdown file in `src/content/reviews/<video_id>.md`.
  - Clicking **✏️ Edit** opens GitHub's built-in file editor for that review. When you (the repo owner) click **Commit changes**, it commits straight to `main` and GitHub Actions automatically rebuilds the live site.
  - Setting `manually_edited: true` in a file's YAML frontmatter locks it so future Python scraper runs never overwrite your manual edits.

---

## 🚀 Quick Start

### 1. Run the Astro Static Website Locally
```bash
# Use Node 20+ (or /usr/bin/node)
npm run dev
```
Open `http://localhost:4321` in your browser.

To build the production static site into `dist/`:
```bash
npm run build
```

---

### 2. Fetch New Videos & Extract Reviews with Gemini

The Python pipeline in [scripts/tracker.py](scripts/tracker.py) uses `yt-dlp` + `youtube-transcript-api` to cache raw timestamped transcripts in `data/transcripts/<video_id>.txt`, and uses the `google-genai` SDK (`gemini-4-hydrogen` via the Interactions API) with structured JSON schema output to generate `src/content/reviews/<video_id>.md`.

1. Copy `.env.example` to `.env` and add your free-tier Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey):
   ```bash
   cp .env.example .env
   ```
2. Run the extractor on the latest `N` videos from the channel:
   ```bash
   # Process the latest 5 videos from @BlotOutdoorsShow
   .venv/bin/python scripts/tracker.py --limit 5

   # Or process specific YouTube video URLs or IDs:
   .venv/bin/python scripts/tracker.py --video https://www.youtube.com/watch?v=6Az_N9EEzHU

   # Or download only raw transcripts to data/transcripts/ without calling Gemini:
   .venv/bin/python scripts/tracker.py --limit 10 --transcripts-only
   ```

---

### 3. Editing Transcripts & Reviews

You have **three ways** to edit any review or transcript:
1. **On the Live Website ("✏️ Edit on GitHub")**:
   - Click **⚙️ Set GitHub Repo for Editing** in the top-right header once (or set `PUBLIC_GITHUB_REPO="your-username/blot"` in `.env` / GitHub Actions).
   - Click **✏️ Edit** on any card to open `src/content/reviews/<video_id>.md` in GitHub's web editor and commit straight to `main`.
2. **Edit Markdown Directly (`src/content/reviews/<video_id>.md`)**:
   - Edit the YAML frontmatter (`name`, `rating_score`, `tags`, `tasting_notes`) or the Markdown transcript excerpt at the bottom.
   - Set `manually_edited: true` so `scripts/tracker.py --force` will never overwrite your tweaks.
3. **Edit Raw Cached Transcript (`data/transcripts/<video_id>.txt`)**:
   - Fix any phonetic caption errors in `data/transcripts/<video_id>.txt` and re-run `.venv/bin/python scripts/tracker.py --video <video_id> --force`.

---

### 4. Deploy to GitHub Pages

A complete GitHub Actions workflow is ready in [.github/workflows/deploy.yml](.github/workflows/deploy.yml):
1. Push this repository to GitHub.
2. In your GitHub repository, go to **Settings → Pages → Build and deployment** and select **Source: GitHub Actions**.
3. Every push (or direct web edit committed to `main`) will automatically build and publish the static site!

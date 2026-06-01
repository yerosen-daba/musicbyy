---
title: Music Byy
emoji: 🎵
colorFrom: blue
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# Music Byy

A music discovery tool framed as a compatibility check.

Two people type a few of their favorite songs each. The backend downloads the 30-second previews, runs real Fourier-based audio analysis to fingerprint each song in a 53-dimensional feature space, and computes a compatibility score from the cosine similarity of the two listeners' average fingerprints. The actual product is the recommendation engine that sits underneath: it searches a cache of thousands of pre-analyzed songs and returns the ones whose audio fingerprints sit closest to the **geometric midpoint** of the two listeners — the sonic intersection of their tastes, almost always songs neither of them has heard.

Live at **[musicbyy.com](https://musicbyy.com)**.

## How it works

**Each song gets a real audio fingerprint.** When a song comes in for the first time, the backend hits Deezer for its 30-second preview MP3 (with iTunes Search API as a fallback for tracks Deezer doesn't have a preview for). librosa loads the audio at 22 kHz mono and extracts:

- **Tempo** via beat tracking
- **Energy** from RMS amplitude, spectral centroid, rolloff, bandwidth, and zero-crossing rate
- **Timbre** as 13 MFCC means and standard deviations
- **Harmonic content** as 12 chroma values and 6 tonnetz components
- **Mode** (major vs. minor) via Krumhansl-Kessler key profile correlation

These features are normalized and packed into a 53-dim vector that uniquely represents how the song sounds. The vector is written to a Supabase Postgres cache — so the song is never analyzed twice across the entire history of the deployment.

**Each user's "musical fingerprint" is the average of their songs' vectors.** Compatibility is the weighted blend of three subspace cosine similarities:

- **Energy** (cosine on the 7 intensity dims) — 30% weight
- **Tempo** (gaussian on raw BPM averages) — 20% weight  
- **Mood** (cosine on the 45 timbre/harmonic/mode dims) — 50% weight

Scaled to 0–100. A subscore breakdown is surfaced in the UI.

**Recommendations come from nearest-neighbor search in audio-feature space.** The recommender computes the midpoint vector between the two users, loads every cached song vector into a numpy matrix (refreshed every 5 minutes), and ranks the full pool by cosine similarity to the midpoint. The top 6 with unique artists, excluding the input songs, are returned. The bigger the cache, the richer the discovery — quality scales with the size of the pre-warmed pool.

## Why this is different from Spotify Blend

Existing two-user compatibility tools (Spotify Blend, MusicTaste.space, etc.) use **collaborative filtering** on listening history — "people whose play patterns look like yours also play X." That requires a streaming-platform account, weeks of listening data, and access to a recommendation engine trained on millions of users.

Music Byy uses **content-based audio analysis** — "song X sounds acoustically similar to your shared midpoint." It works for anyone who can name 5 songs, on any platform, with no listening history required. No login. Cross-platform. And because the fingerprints come from raw signal analysis, the algorithm can surface genuinely surprising recommendations across genres that collaborative-filter engines wouldn't connect.

## Tech stack

| Layer | What it is |
| --- | --- |
| Frontend | Vanilla HTML / CSS / JS, hosted on GitHub Pages |
| Backend | FastAPI on Render |
| Audio analysis | librosa + numpy + scipy |
| Cache / metadata store | Supabase Postgres via asyncpg |
| Music data sources | Deezer Public API (primary), iTunes Search API (fallback) |

## Project structure

```
app.py          FastAPI entry point, route handlers, lifespan
analyzer.py     librosa-based audio feature extraction + vector packing
deezer.py       Deezer search + cache-first track enrichment
itunes.py       iTunes Search API fallback for missing previews
match.py        Compatibility scoring + midpoint nearest-neighbor recommendations
cache.py        Supabase Postgres feature cache (async via asyncpg)
client.py       Shared httpx async client singleton
prewarm.py      One-shot script that walks Deezer's genres and fills the cache
index.html      The entire frontend in one file
Dockerfile      Render's build instructions (installs ffmpeg + librosa deps)
requirements.txt Python deps pinned to Python 3.11 + 3.13 compatible versions
CNAME           GitHub Pages custom domain (musicbyy.com)
```

## Running locally

Prerequisites: Python 3.11+ and a Supabase Postgres project for the feature cache.

```bash
git clone https://github.com/yerosen-daba/musicbyy.git
cd musicbyy
pip install -r requirements.txt
export DATABASE_URL="postgresql://postgres.<project-id>:<password>@aws-1-us-east-2.pooler.supabase.com:5432/postgres"
uvicorn app:app --reload --port 8000
```

Then open `index.html` in your browser. The frontend's `API` constant points at the live Render service by default — change it to `http://localhost:8000` for fully local development.

### Pre-warming the cache

The midpoint recommender draws candidates entirely from the cache, so a populated cache is what makes recommendations rich. Run once after deploying:

```bash
python3 prewarm.py --tracks 5000 --concurrency 4
```

This walks Deezer's full genre list, pulls top tracks per genre, and writes their feature vectors to Supabase. Roughly 1 second per track. Already-cached tracks are skipped on re-runs.

## Credits

The original prototype was built as a CS course project using metadata proxies (popularity rank, release year) as stand-ins for energy and mood. The current architecture — real Fourier-based audio analysis, Supabase-backed feature cache, midpoint nearest-neighbor recommendations, and the refreshed frontend — was developed in collaboration with Claude (Anthropic) over an extended pair-programming session.

Music data is provided by the Deezer Public API and the iTunes Search API.

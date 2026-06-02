"""
prewarm.py — One-shot script to populate the feature cache with a broad,
diverse pool of analyzed songs.

The midpoint-search recommender draws candidates exclusively from the cache,
so cache size + diversity directly determines how good recommendations are.
This script walks several Deezer discovery sources to maximize coverage:

  1. Genre charts          — top tracks per genre (~150 genres)
  2. Editorial sections    — Deezer's curated playlists per region/mood
  3. Radio stations        — algorithmic per-genre radio (~250 stations)
  4. Per-year searches     — historical catalog 1960–2025

With default settings it discovers ~10,000–15,000 unique tracks across
sources, deduped. Already-cached tracks are skipped on re-runs.

Usage:
    DATABASE_URL=... python3 prewarm.py
    DATABASE_URL=... python3 prewarm.py --tracks 8000
    DATABASE_URL=... python3 prewarm.py --concurrency 4
    DATABASE_URL=... python3 prewarm.py --sources genres,radio   # subset
"""

import argparse
import asyncio
import os
import sys
import time

import httpx

# Make our modules importable when run from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cache
import client
import deezer


# ─── Defaults ───────────────────────────────────────────────────────────────

PER_GENRE_LIMIT        = 100   # Deezer max per chart-tracks request
PER_RADIO_LIMIT        = 40    # Radio endpoints return ~40 tracks max
PER_EDITORIAL_LIMIT    = 100
PER_YEAR_LIMIT         = 100

# Year range for the per-year search source.
YEAR_RANGE = (1960, 2025)

DEFAULT_TRACK_CAP   = 10000
DEFAULT_CONCURRENCY = 2

# Discovery sources the script can run. Each name maps to a builder function
# down below; --sources lets the user run subsets.
ALL_SOURCES = ["genres", "editorial", "radio", "years"]


# ─── HTTP helper ─────────────────────────────────────────────────────────────

async def _get(url: str, params: dict | None = None) -> dict | None:
    """Hit a Deezer endpoint, return decoded JSON or None on any failure."""
    try:
        r = await client.http_client.get(url, params=params or {}, timeout=10.0)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  fetch error ({url}): {e}")
    return None


# ─── Source 1: Genre charts ──────────────────────────────────────────────────

async def discover_from_genres() -> list[dict]:
    """Walk every Deezer genre and pull its top tracks."""
    print(f"\n[1/4] Discovering from genre charts...")

    data = await _get("https://api.deezer.com/genre")
    genres = (data or {}).get("data", []) or []
    print(f"      Got {len(genres)} genres.")

    # Include the "All" chart too (genre id 0) for general top tracks.
    entries = [{"id": 0, "name": "All"}] + genres

    tracks: list[dict] = []
    for entry in entries:
        gid, gname = entry["id"], entry["name"]
        data = await _get(
            f"https://api.deezer.com/chart/{gid}/tracks",
            params={"limit": PER_GENRE_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        print(f"      {gname[:22]:22s} (genre {gid:4d}): +{len(batch):3d} tracks")

    return tracks


# ─── Source 2: Editorial sections ────────────────────────────────────────────

async def discover_from_editorial() -> list[dict]:
    """Pull tracks from each editorial section's charts."""
    print(f"\n[2/4] Discovering from editorial sections...")

    data = await _get("https://api.deezer.com/editorial")
    editorials = (data or {}).get("data", []) or []
    print(f"      Got {len(editorials)} editorial sections.")

    tracks: list[dict] = []
    for ed in editorials:
        eid, ename = ed["id"], ed.get("name", "?")
        data = await _get(
            f"https://api.deezer.com/editorial/{eid}/charts/tracks",
            params={"limit": PER_EDITORIAL_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        print(f"      {ename[:22]:22s} (ed {eid:4d}):    +{len(batch):3d} tracks")

    return tracks


# ─── Source 3: Radio stations ────────────────────────────────────────────────

async def discover_from_radio() -> list[dict]:
    """Pull tracks from every Deezer radio station's seed list."""
    print(f"\n[3/4] Discovering from radio stations...")

    data = await _get("https://api.deezer.com/radio")
    radios = (data or {}).get("data", []) or []
    print(f"      Got {len(radios)} radio stations.")

    tracks: list[dict] = []
    for radio in radios:
        rid, rname = radio["id"], radio.get("title", "?")
        data = await _get(
            f"https://api.deezer.com/radio/{rid}/tracks",
            params={"limit": PER_RADIO_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        # Radios are numerous; only log every ~10th to keep output readable.
        if (rid % 10) == 0 or len(batch) > 30:
            print(f"      {rname[:22]:22s} (radio {rid:5d}): +{len(batch):3d} tracks")

    return tracks


# ─── Source 4: Per-year searches ─────────────────────────────────────────────

async def discover_from_years() -> list[dict]:
    """Pull top tracks for each year in YEAR_RANGE via the search endpoint."""
    start, end = YEAR_RANGE
    print(f"\n[4/4] Discovering by year ({start}–{end})...")

    tracks: list[dict] = []
    for year in range(start, end + 1):
        data = await _get(
            "https://api.deezer.com/search",
            params={"q": f"year:{year}", "limit": PER_YEAR_LIMIT},
        )
        batch = (data or {}).get("data", []) or []
        tracks.extend(batch)
        if batch:
            print(f"      year {year}: +{len(batch):3d} tracks")

    return tracks


# ─── Discovery orchestrator ──────────────────────────────────────────────────

SOURCE_FUNCS = {
    "genres":    discover_from_genres,
    "editorial": discover_from_editorial,
    "radio":     discover_from_radio,
    "years":     discover_from_years,
}


def deezer_track_to_dict(t: dict) -> dict:
    artist = t.get("artist", {}) or {}
    album  = t.get("album",  {}) or {}
    return {
        "name":      t.get("title", ""),
        "artist":    artist.get("name", ""),
        "artist_id": str(artist.get("id", "")),
        "track_id":  str(t.get("id", "")),
        "deezer_id": t.get("id"),
        "image":     album.get("cover_medium", "") or album.get("cover_big", ""),
        "url":       t.get("link", ""),
        "preview":   t.get("preview", ""),
    }


async def discover_candidates(sources: list[str], target: int) -> list[dict]:
    """Run each requested source, dedup by track ID, cap at target."""
    candidates: dict[str, dict] = {}

    for source_name in sources:
        if source_name not in SOURCE_FUNCS:
            print(f"  ! unknown source: {source_name} — skipping")
            continue
        if len(candidates) >= target:
            print(f"  Reached target of {target} unique tracks, stopping.")
            break

        raw_tracks = await SOURCE_FUNCS[source_name]()

        added = 0
        for t in raw_tracks:
            tid = str(t.get("id", ""))
            if tid and tid not in candidates:
                candidates[tid] = deezer_track_to_dict(t)
                added += 1
        print(f"      → {source_name}: +{added} unique  (running pool: {len(candidates)})")

    return list(candidates.values())[:target]


# ─── The actual analysis loop ───────────────────────────────────────────────

async def prewarm(sources: list[str], target: int, concurrency: int) -> None:
    candidates = await discover_candidates(sources=sources, target=target)
    print(f"\n{'='*60}")
    print(f"Discovered {len(candidates)} unique candidate tracks across "
          f"{len(sources)} sources.")

    if not candidates:
        return

    # Skip anything already in the cache.
    existing = await cache.get_many_cached([c["deezer_id"] for c in candidates])
    fresh = [c for c in candidates if str(c["deezer_id"]) not in existing]
    print(f"Already cached: {len(existing)}")
    print(f"To analyze:     {len(fresh)}\n")

    if not fresh:
        print("Nothing to do — the cache is fully warmed for this pool.")
        return

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    failed    = 0
    start     = time.time()

    async def warm_one(track: dict, idx: int):
        nonlocal completed, failed
        async with sem:
            t0 = time.time()
            try:
                result = await deezer.enrich_track(track)
                if result.get("vector") is None:
                    failed += 1
                    status = "FAIL"
                else:
                    completed += 1
                    status = "OK  "
            except Exception as e:
                failed += 1
                status = f"ERR ({type(e).__name__})"
            elapsed = time.time() - t0
            print(
                f"  [{idx + 1:5d}/{len(fresh)}] {status} "
                f"{elapsed:5.2f}s — {track['artist'][:25]:25s} — {track['name'][:50]}"
            )

    await asyncio.gather(*[warm_one(t, i) for i, t in enumerate(fresh)])

    total_elapsed = time.time() - start
    final_count   = await cache.count_cached()
    print(f"\n{'='*60}")
    print(f"Done in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Newly analyzed: {completed}")
    print(f"  Failed:         {failed}")
    print(f"  Cache total:    {final_count}")
    if completed > 0:
        print(f"  Avg time/track: {total_elapsed / completed:.2f}s")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracks", type=int, default=DEFAULT_TRACK_CAP,
        help=f"Cap on unique tracks to pre-warm. Default {DEFAULT_TRACK_CAP}.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=(
            "Parallel librosa analyses. Laptop can handle 4-8; "
            "leave at 2 for shared/free hosting."
        ),
    )
    parser.add_argument(
        "--sources", type=str, default=",".join(ALL_SOURCES),
        help=(
            f"Comma-separated discovery sources. Available: {','.join(ALL_SOURCES)}. "
            f"Default runs all of them in order."
        ),
    )
    args = parser.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    client.http_client = httpx.AsyncClient(timeout=30.0)
    await cache.init_pool()
    try:
        await prewarm(
            sources=sources,
            target=args.tracks,
            concurrency=args.concurrency,
        )
    finally:
        await client.http_client.aclose()
        await cache.close_pool()


if __name__ == "__main__":
    asyncio.run(main())

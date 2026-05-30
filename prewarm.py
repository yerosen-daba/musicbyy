"""
prewarm.py — One-shot script to populate the feature cache with a broad pool
of analyzed songs.

The midpoint-search recommender draws candidates exclusively from the cache,
so the cache size directly determines how rich the recommendations can be.
This script seeds the cache by walking every Deezer genre (including
sub-genres) and analyzing each genre's top tracks. With default settings it
produces a pool of roughly 5,000–10,000 unique songs.

Usage:
    DATABASE_URL=... python3 prewarm.py
    DATABASE_URL=... python3 prewarm.py --tracks 5000          # cap total
    DATABASE_URL=... python3 prewarm.py --per-genre 100        # depth per genre
    DATABASE_URL=... python3 prewarm.py --concurrency 4        # parallelism

Already-cached tracks are skipped on re-runs, so this is safe to re-run any
time you want to extend coverage (e.g. after Deezer's charts shift over a
season).
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


PER_GENRE_DEFAULT  = 100   # Deezer's max per chart-tracks request
DEFAULT_TRACK_CAP  = 8000  # rough target — enough for solid midpoint search
DEFAULT_CONCURRENCY = 2    # parallel librosa workers (CPU-bound)


# ─── Genre + track discovery ─────────────────────────────────────────────────

async def fetch_all_genres() -> list[dict]:
    """Fetch Deezer's full genre list. ~150 entries including sub-genres."""
    try:
        r = await client.http_client.get(
            "https://api.deezer.com/genre", timeout=10.0,
        )
        if r.status_code != 200:
            return []
        return r.json().get("data", []) or []
    except Exception as e:
        print(f"  fetch_all_genres error: {e}")
        return []


async def fetch_genre_top_tracks(genre_id: int, limit: int) -> list[dict]:
    """Pull the top tracks for one Deezer genre."""
    try:
        r = await client.http_client.get(
            f"https://api.deezer.com/chart/{genre_id}/tracks",
            params={"limit": limit},
            timeout=10.0,
        )
        if r.status_code != 200:
            return []
        return r.json().get("data", []) or []
    except Exception as e:
        print(f"  [genre {genre_id}] fetch error: {e}")
        return []


def deezer_track_to_dict(t: dict) -> dict:
    """Convert raw Deezer search payload into the shape enrich_track expects."""
    artist = t.get("artist", {})
    album  = t.get("album",  {})
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


# ─── Discovery: build the candidate set ──────────────────────────────────────

async def discover_candidates(per_genre: int, target: int) -> list[dict]:
    """Walk every Deezer genre and collect top tracks until we hit `target`."""
    print(f"Fetching Deezer's full genre list...")
    genres = await fetch_all_genres()
    if not genres:
        print("  Failed to fetch genres. Exiting.")
        return []

    # The "All" genre (id=0) plus everything Deezer returns. We include the
    # All chart because it surfaces tracks that are popular across all genres
    # (a good safety net for catching obvious hits regardless of genre tags).
    genre_entries = [{"id": 0, "name": "All"}] + genres
    print(f"  Got {len(genres)} genres, total to walk: {len(genre_entries)}")
    print()

    candidates: dict[str, dict] = {}

    for entry in genre_entries:
        if len(candidates) >= target:
            print(f"  Reached target of {target} unique tracks, stopping discovery.")
            break

        gid   = entry["id"]
        gname = entry["name"]
        raw_tracks = await fetch_genre_top_tracks(gid, per_genre)
        added = 0
        for t in raw_tracks:
            tid = str(t.get("id", ""))
            if tid and tid not in candidates:
                candidates[tid] = deezer_track_to_dict(t)
                added += 1
        print(
            f"  {gname[:22]:22s} (genre {gid:4d}): +{added:3d} new   "
            f"(pool: {len(candidates)})"
        )

    return list(candidates.values())[:target]


# ─── The actual analysis loop ───────────────────────────────────────────────

async def prewarm(per_genre: int, target: int, concurrency: int) -> None:
    candidates = await discover_candidates(per_genre=per_genre, target=target)
    print(f"\nDiscovered {len(candidates)} unique candidate tracks.\n")

    if not candidates:
        return

    # Skip anything already in the cache — running this script multiple
    # times should be cheap, not redo all the work.
    existing = await cache.get_many_cached([c["deezer_id"] for c in candidates])
    fresh = [c for c in candidates if str(c["deezer_id"]) not in existing]
    print(f"Already cached: {len(existing)}")
    print(f"To analyze:     {len(fresh)}\n")

    if not fresh:
        print("Nothing to do — the cache is already fully warmed for this pool.")
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
        "--per-genre", type=int, default=PER_GENRE_DEFAULT,
        help=f"How many top tracks to pull per Deezer genre. Default {PER_GENRE_DEFAULT}.",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=(
            "Parallel librosa analyses. Local laptop can probably handle 4-8; "
            "leave at 2 if running this on the Render Starter dyno."
        ),
    )
    args = parser.parse_args()

    # Init the global HTTP client and DB pool the same way app.py does.
    client.http_client = httpx.AsyncClient(timeout=30.0)
    await cache.init_pool()
    try:
        await prewarm(
            per_genre=args.per_genre,
            target=args.tracks,
            concurrency=args.concurrency,
        )
    finally:
        await client.http_client.aclose()
        await cache.close_pool()


if __name__ == "__main__":
    asyncio.run(main())

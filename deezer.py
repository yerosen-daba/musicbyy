"""
deezer.py — Deezer search + real audio feature enrichment.

Replaces the old metadata-proxy approach (popularity rank as "energy",
release year as "valence") with actual Fourier-based audio analysis via
analyzer.py, backed by a persistent Postgres feature cache (cache.py).

The cache makes this fast: every track is analyzed exactly once across
the entire history of the deployment. First user to query a song eats
~1 second of librosa. Every subsequent user (across sessions, restarts,
deploys) gets a sub-100ms cache hit.
"""

import asyncio

import analyzer
import cache
import client


DEEZER_SEARCH = "https://api.deezer.com/search"
DEEZER_TRACK  = "https://api.deezer.com/track"

# Cap concurrent audio analyses so we don't OOM the 512MB Render dyno.
# librosa peaks around ~150MB per analysis; 3 concurrent leaves headroom
# for the FastAPI process itself and the Postgres pool.
_analysis_semaphore = asyncio.Semaphore(3)


# ─── Search ──────────────────────────────────────────────────────────────────

async def search_track(query: str) -> dict | None:
    """Search Deezer for a track. Returns basic metadata or None.

    Doesn't run audio analysis — that's enrich_track's job.
    """
    try:
        r = await client.http_client.get(
            DEEZER_SEARCH, params={"q": query, "limit": 1},
        )
        data = r.json().get("data", [])
    except Exception:
        data = []

    if not data:
        return None

    t = data[0]
    artist = t.get("artist", {})
    album  = t.get("album",  {})
    track_id = t.get("id")

    return {
        "name":      t.get("title", ""),
        "artist":    artist.get("name", ""),
        "artist_id": str(artist.get("id", "")),
        "track_id":  str(track_id),
        "deezer_id": track_id,
        "image":     album.get("cover_medium", "") or album.get("cover_big", ""),
        "url":       t.get("link", ""),
        "preview":   t.get("preview", ""),
    }


# ─── Enrichment ──────────────────────────────────────────────────────────────

async def enrich_track(track: dict) -> dict:
    """Attach real audio features to a track dict.

    Returns the original track plus two new fields:
        "features": full feature dict from analyzer (or None on failure)
        "vector":   53-dim float list ready for cosine math (or None)

    Cache-first: hits Postgres before doing any audio download/analysis.
    On a cache miss, analyzes the preview clip, then writes the result
    back so future requests skip the work.

    Concurrency-limited so parallel enrichment of many tracks doesn't
    spin up too many simultaneous librosa workers.
    """
    deezer_id = track.get("deezer_id") or track.get("track_id")

    # ── Cache hit path (fast path) ──────────────────────────────────────
    if deezer_id:
        cached = await cache.get_cached(deezer_id)
        if cached is not None:
            return {
                **track,
                "features": cached["features"],
                "vector":   cached["vector"],
            }

    # ── Cache miss: live analysis (slow path, limited concurrency) ──────
    preview_url = track.get("preview", "") or ""

    async with _analysis_semaphore:
        result = await analyzer.analyze_track_with_fallback(
            primary_url=preview_url,
            track_name=track.get("name", ""),
            artist_name=track.get("artist", ""),
        )

    if result is None:
        # Audio download or analysis failed. Return the track with empty
        # features; downstream code (match.py) will skip it in calculations.
        return {**track, "features": None, "vector": None}

    # ── Write through to the cache ──────────────────────────────────────
    # We persist artist_id, URL, and cover image alongside the vector so
    # the midpoint-search recommender can return tracks straight from the
    # cache without round-tripping Deezer for metadata.
    if deezer_id:
        try:
            await cache.set_cached(
                deezer_id=deezer_id,
                features=result["features"],
                vector=result["vector"],
                track_name=track.get("name"),
                artist_name=track.get("artist"),
                artist_id=track.get("artist_id"),
                track_url=track.get("url"),
                track_image=track.get("image"),
            )
        except Exception:
            # Cache write failure shouldn't tank the user-facing request.
            pass

    return {
        **track,
        "features": result["features"],
        "vector":   result["vector"],
    }


# ─── Composite helpers ───────────────────────────────────────────────────────

async def search_and_enrich(query: str) -> dict | None:
    """Convenience: search Deezer for a query, then run enrichment."""
    track = await search_track(query)
    if not track:
        return None
    return await enrich_track(track)


async def search_many_tracks(queries: list[str]) -> list[dict]:
    """Search and enrich a list of queries in parallel.

    Tracks that fail to even *search* on Deezer are dropped. Tracks that
    search OK but fail audio analysis are kept (with vector=None) so the
    user still sees their pick on the results page; match.py filters them
    out of the math.
    """
    results = await asyncio.gather(*[search_and_enrich(q) for q in queries])
    return [r for r in results if r is not None]

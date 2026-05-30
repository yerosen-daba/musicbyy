"""
cache.py — Persistent feature cache backed by Supabase Postgres.

Replaces the old in-memory _feature_cache dict in deezer.py (which died on
every server restart) with a real database. Once a song's audio features
have been computed, they're stored here forever and served sub-100ms on
every subsequent request — across users, across restarts, across deploys.

Schema (created on app startup if missing):

    feature_cache
      deezer_id    TEXT       PRIMARY KEY    -- Deezer track ID
      features     JSONB      NOT NULL       -- full feature dict
      vector       JSONB      NOT NULL       -- 53-dim feature vector
      track_name   TEXT                      -- for debugging / browsing
      artist_name  TEXT                      -- for debugging / browsing
      created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
      updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()

Connection details come from the DATABASE_URL environment variable, which
must be the Supabase Session Pooler URI (IPv4-compatible, port 5432).
"""

import json
import os

import asyncpg


# ─── Module state ────────────────────────────────────────────────────────────

# Connection pool, initialized via init_pool() during app startup.
_pool: asyncpg.Pool | None = None


# ─── Schema migration ────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feature_cache (
    deezer_id    TEXT PRIMARY KEY,
    features     JSONB NOT NULL,
    vector       JSONB NOT NULL,
    track_name   TEXT,
    artist_name  TEXT,
    artist_id    TEXT,
    track_url    TEXT,
    track_image  TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotent column adds, in case this table was created by an earlier
-- version of the schema. Safe to leave in forever.
ALTER TABLE feature_cache ADD COLUMN IF NOT EXISTS artist_id   TEXT;
ALTER TABLE feature_cache ADD COLUMN IF NOT EXISTS track_url   TEXT;
ALTER TABLE feature_cache ADD COLUMN IF NOT EXISTS track_image TEXT;

CREATE INDEX IF NOT EXISTS idx_feature_cache_artist
  ON feature_cache (artist_name);
"""


# ─── Lifecycle ───────────────────────────────────────────────────────────────

async def init_pool(dsn: str | None = None) -> None:
    """Create the connection pool and apply the schema.

    Idempotent. Call once from the FastAPI lifespan on startup.

    Args:
        dsn: Postgres connection string. If not provided, reads from the
             DATABASE_URL environment variable.
    """
    global _pool

    if _pool is not None:
        return  # already initialized

    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL environment variable not set. "
            "Expected the Supabase Session Pooler URI."
        )

    # Supabase requires TLS. Pass ssl='require' so asyncpg negotiates it.
    # min_size kept low so the app starts fast even on cold boot;
    # max_size kept modest because Supabase's free tier has connection limits.
    _pool = await asyncpg.create_pool(
        dsn=dsn,
        min_size=1,
        max_size=5,
        ssl="require",
        command_timeout=10.0,
        # Disable prepared statement caching to play nicely with poolers.
        statement_cache_size=0,
    )

    # Apply schema. Safe to re-run on every startup.
    async with _pool.acquire() as conn:
        await conn.execute(_SCHEMA_SQL)


async def close_pool() -> None:
    """Tear down the pool. Call from FastAPI lifespan shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# ─── Read paths ──────────────────────────────────────────────────────────────

async def get_cached(deezer_id: str | int) -> dict | None:
    """Look up cached features for one track. Returns the analyzer-shape dict
    {"features": ..., "vector": [...]} or None if not in cache.
    """
    if _pool is None or deezer_id is None:
        return None

    row = await _pool.fetchrow(
        "SELECT features, vector FROM feature_cache WHERE deezer_id = $1",
        str(deezer_id),
    )
    if row is None:
        return None

    return {
        "features": json.loads(row["features"]) if isinstance(row["features"], str) else row["features"],
        "vector":   json.loads(row["vector"])   if isinstance(row["vector"], str)   else row["vector"],
    }


async def get_many_cached(deezer_ids: list[str | int]) -> dict[str, dict]:
    """Batch lookup. Returns a dict mapping deezer_id (string) → cached entry.
    Missing IDs are simply not present in the returned dict.

    This is the hot path for the recommendation engine: pull 100 candidate
    track IDs, hit the DB once, get back whatever's already analyzed.
    """
    if _pool is None or not deezer_ids:
        return {}

    str_ids = [str(d) for d in deezer_ids if d is not None]
    if not str_ids:
        return {}

    rows = await _pool.fetch(
        "SELECT deezer_id, features, vector FROM feature_cache "
        "WHERE deezer_id = ANY($1::text[])",
        str_ids,
    )

    result: dict[str, dict] = {}
    for row in rows:
        result[row["deezer_id"]] = {
            "features": json.loads(row["features"]) if isinstance(row["features"], str) else row["features"],
            "vector":   json.loads(row["vector"])   if isinstance(row["vector"], str)   else row["vector"],
        }
    return result


# ─── Write path ──────────────────────────────────────────────────────────────

async def set_cached(
    deezer_id: str | int,
    features: dict,
    vector: list[float],
    track_name: str | None = None,
    artist_name: str | None = None,
    artist_id: str | None = None,
    track_url: str | None = None,
    track_image: str | None = None,
) -> None:
    """Upsert a track's analysis into the cache.

    Stores the audio features + vector alongside enough metadata that the
    midpoint-search recommender can return tracks straight out of the cache
    without having to round-trip to Deezer for image/URL/artist info.

    Uses ON CONFLICT so re-analyzing a track (e.g. after we improve the
    feature pipeline) overwrites the old entry. updated_at gets bumped.
    """
    if _pool is None or deezer_id is None:
        return

    await _pool.execute(
        """
        INSERT INTO feature_cache
            (deezer_id, features, vector, track_name, artist_name,
             artist_id, track_url, track_image)
        VALUES ($1, $2::jsonb, $3::jsonb, $4, $5, $6, $7, $8)
        ON CONFLICT (deezer_id) DO UPDATE
            SET features    = EXCLUDED.features,
                vector      = EXCLUDED.vector,
                track_name  = COALESCE(EXCLUDED.track_name,  feature_cache.track_name),
                artist_name = COALESCE(EXCLUDED.artist_name, feature_cache.artist_name),
                artist_id   = COALESCE(EXCLUDED.artist_id,   feature_cache.artist_id),
                track_url   = COALESCE(EXCLUDED.track_url,   feature_cache.track_url),
                track_image = COALESCE(EXCLUDED.track_image, feature_cache.track_image),
                updated_at  = NOW()
        """,
        str(deezer_id),
        json.dumps(features),
        json.dumps(vector),
        track_name,
        artist_name,
        artist_id,
        track_url,
        track_image,
    )


async def get_all_cached() -> list[dict]:
    """Return every cached track's vector + metadata.

    Used by the midpoint-search recommender. The expectation is that callers
    cache the result in-process for the lifetime of a few requests rather
    than calling this every time — at scale this pulls all rows from the
    table, which is a lot of bytes over the wire.

    Each returned dict has:
        deezer_id, vector (list[float]), track_name, artist_name,
        artist_id, track_url, track_image
    """
    if _pool is None:
        return []

    rows = await _pool.fetch(
        "SELECT deezer_id, vector, track_name, artist_name, "
        "       artist_id, track_url, track_image "
        "FROM feature_cache"
    )

    result: list[dict] = []
    for row in rows:
        vec = row["vector"]
        if isinstance(vec, str):
            vec = json.loads(vec)
        result.append({
            "deezer_id":   row["deezer_id"],
            "vector":      vec,
            "track_name":  row["track_name"],
            "artist_name": row["artist_name"],
            "artist_id":   row["artist_id"],
            "track_url":   row["track_url"],
            "track_image": row["track_image"],
        })
    return result


# ─── Observability ───────────────────────────────────────────────────────────

async def count_cached() -> int:
    """Return total number of cached tracks. Useful for health checks and
    monitoring the pre-warm progress."""
    if _pool is None:
        return 0
    row = await _pool.fetchrow("SELECT COUNT(*) AS n FROM feature_cache")
    return int(row["n"]) if row else 0

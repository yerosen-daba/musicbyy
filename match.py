"""
match.py — Compatibility scoring + nearest-neighbor recommendations.

Compatibility scoring:
  * Each user is represented by the mean of their songs' 53-dim feature
    vectors (their "musical fingerprint" in audio-feature space).
  * Overall similarity is a weighted blend of three subspace scores:
      - energy  — RMS, spectral centroid/rolloff/bandwidth, ZCR
      - tempo   — measured BPM (gaussian on raw BPM; cosine on 1-D degenerates)
      - mood    — MFCC timbre + chroma + tonnetz + major/minor mode

Recommendations:
  * Compute the midpoint vector between the two users.
  * Search the ENTIRE cached feature pool for the songs whose vectors are
    closest to that midpoint (cosine similarity, vectorized in numpy).
  * Return the top 6 with unique artists, excluding the input songs/artists.
  * Quality scales with cache size — a 5k-song cache produces dramatically
    richer discovery than a 500-song cache. The prewarm script is what
    seeds this pool.
"""

import asyncio
import math
import time

import numpy as np

import analyzer
import cache


# ─── Subscore configuration ──────────────────────────────────────────────────
#
# Indices into the 53-dim analyzer vector. We pull three semantic groups out
# for the per-category scores shown in the UI. Note that the "mood" slice is
# everything beyond the scalar features and tempo — it covers timbre (MFCCs),
# harmonic content (chroma + tonnetz), and major/minor mode together because
# users intuitively read all of those as part of "what a song feels like."

SUBSCORE_INDICES = {
    "tempo":  slice(0, 1),    # special-cased — gaussian on raw BPM
    "energy": slice(1, 8),    # 7 dims of intensity/brightness features
    "mood":   slice(8, 53),   # 45 dims: MFCCs + chroma + tonnetz + mode
}

# Weights for the overall_similarity blend. Must sum to 1.0.
OVERALL_WEIGHTS = {
    "tempo":  0.20,
    "energy": 0.30,
    "mood":   0.50,
}

# Standard deviation for the tempo gaussian (in BPM). 5 BPM is strict:
# two songs at 118 and 124 BPM score ~0.49 tempo similarity, two songs
# at 100 and 130 BPM score ~0.0 (basically zero).
TEMPO_SIGMA = 5.0

# Empirical baseline cosine between two random user-mean vectors.
# When we average a user's songs into a single vector, the mean tends to
# land near the center of feature space — so two random users start
# at a cosine baseline closer to ~0.85 than 0 due to convergence.
# We rescale relative to this baseline so the percentage feels right:
#   cosine == baseline → 0% similar (random pair, not "85% similar")
#   cosine == 1.0      → 100% similar (identical fingerprint)
COSINE_BASELINE = 0.88

# Power exponent applied after baseline rescaling. >1 compresses high
# similarities downward, making "looks 90% the same" require genuinely
# being 90% the same instead of just sharing baseline structure.
SIMILARITY_POWER = 2.0


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _valid_songs(songs: list[dict]) -> list[dict]:
    """Drop songs whose audio analysis didn't produce a vector."""
    return [s for s in songs if s.get("vector") is not None and s.get("features") is not None]


def _user_vector(valid_songs: list[dict]) -> np.ndarray | None:
    """The mean of all the user's song vectors. None if no valid songs."""
    if not valid_songs:
        return None
    return np.mean(np.array([s["vector"] for s in valid_songs]), axis=0)


def _user_tempo(valid_songs: list[dict]) -> float | None:
    """Average BPM across the user's valid songs."""
    if not valid_songs:
        return None
    return float(np.mean([s["features"]["tempo"] for s in valid_songs]))


def _cosine(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]. Returns 0 for degenerate (zero-norm) cases."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _gaussian_sim(diff: float, sigma: float) -> float:
    """1.0 when diff=0, decays smoothly to 0 as diff grows."""
    return math.exp(-(diff * diff) / (2 * sigma * sigma))


def _to_unit(cos: float) -> float:
    """Map cosine similarity to a [0, 1] perceptual-similarity value.

    Two stages:
      1. Linear rescale relative to the empirical baseline (cosine 0.88 → 0,
         cosine 1.0 → 1). Two random user vectors end up near 0% instead of
         85%, which is what cosine on convergent mean vectors does naturally.
      2. Power transform: raise to SIMILARITY_POWER. This compresses high
         values down, so "looks 90% the same" requires genuinely being 90%
         the same instead of just sharing baseline music structure.

    Net effect: scoring is sharply discriminating. Mainstream-pop vs.
    hip-hop should land in the 50–70 range, not the 90s.
    """
    if cos <= COSINE_BASELINE:
        return 0.0
    rescaled = (cos - COSINE_BASELINE) / (1.0 - COSINE_BASELINE)
    return min(1.0, rescaled) ** SIMILARITY_POWER


# ─── Compatibility scoring ───────────────────────────────────────────────────

def compute_compatibility(songs1: list[dict], songs2: list[dict]) -> dict:
    """Cosine-based compatibility with energy/tempo/mood breakdown."""
    v1_songs = _valid_songs(songs1)
    v2_songs = _valid_songs(songs2)

    u1 = _user_vector(v1_songs)
    u2 = _user_vector(v2_songs)

    if u1 is None or u2 is None:
        return _empty_compatibility()

    # Subspace similarities
    energy_cos = _cosine(u1[SUBSCORE_INDICES["energy"]],
                         u2[SUBSCORE_INDICES["energy"]])
    mood_cos   = _cosine(u1[SUBSCORE_INDICES["mood"]],
                         u2[SUBSCORE_INDICES["mood"]])

    energy_sim = _to_unit(energy_cos)
    mood_sim   = _to_unit(mood_cos)

    # Tempo: gaussian on actual BPM averages
    t1 = _user_tempo(v1_songs)
    t2 = _user_tempo(v2_songs)
    tempo_sim = _gaussian_sim(abs(t1 - t2), TEMPO_SIGMA)

    overall = (
        OVERALL_WEIGHTS["energy"] * energy_sim
        + OVERALL_WEIGHTS["tempo"]  * tempo_sim
        + OVERALL_WEIGHTS["mood"]   * mood_sim
    )
    overall_score = round(overall * 100)

    energy_score = round(energy_sim * 40)
    tempo_score  = round(tempo_sim  * 30)
    mood_score   = round(mood_sim   * 30)

    return {
        "total":              overall_score,
        "overall_similarity": overall_score,
        "energy_score":       energy_score,
        "tempo_score":        tempo_score,
        "mood_score":         mood_score,
        "vibe1":              _vibe_label(v1_songs),
        "vibe2":              _vibe_label(v2_songs),
        "details": {
            "tempo1_bpm":     round(t1, 1),
            "tempo2_bpm":     round(t2, 1),
            "energy_cosine":  round(energy_cos, 3),
            "mood_cosine":    round(mood_cos,   3),
            "valid_count_1":  len(v1_songs),
            "valid_count_2":  len(v2_songs),
        },
    }


def _empty_compatibility() -> dict:
    return {
        "total":              0,
        "overall_similarity": 0,
        "energy_score":       0,
        "tempo_score":        0,
        "mood_score":         0,
        "vibe1":              "Unknown",
        "vibe2":              "Unknown",
        "details":            {},
    }


def compatibility_message(score: int) -> str:
    if score >= 85: return "Soulmates 🎵 You were made for the same playlist."
    if score >= 70: return "Great match! You'd have an amazing road trip together."
    if score >= 55: return "Solid overlap. You'd agree on at least half the aux cord."
    if score >= 40: return "Some common ground. You'd negotiate the playlist."
    if score >= 25: return "Different worlds. But opposites attract, right?"
    return "Complete opposites. Take turns on the aux — no fighting."


# ─── Vibe labels ─────────────────────────────────────────────────────────────

def _vibe_label(valid_songs: list[dict]) -> str:
    """Generate a short descriptive label from real audio averages."""
    if not valid_songs:
        return "Unknown"

    feats = [s["features"] for s in valid_songs]
    avg_tempo    = float(np.mean([f["tempo"]         for f in feats]))
    avg_rms      = float(np.mean([f["rms_mean"]      for f in feats]))
    avg_centroid = float(np.mean([f["centroid_mean"] for f in feats]))
    avg_mode     = float(np.mean([f["mode"]          for f in feats]))

    parts = []

    if   avg_rms > 0.15: parts.append("High Energy")
    elif avg_rms < 0.06: parts.append("Chill")
    else:                parts.append("Moderate")

    if   avg_tempo > 130: parts.append("Fast-Paced")
    elif avg_tempo < 90:  parts.append("Slow-Paced")

    if   avg_centroid > 3500: parts.append("Bright")
    elif avg_centroid < 1800: parts.append("Warm")

    if   avg_mode > 0.66: parts.append("Upbeat")
    elif avg_mode < 0.34: parts.append("Moody")
    else:                 parts.append("Balanced")

    return " / ".join(parts)


# ─── Vector-pool matrix (in-process cache of all cached song vectors) ───────
#
# Recommendations work by computing cosine similarity between the midpoint
# vector and EVERY cached song's vector. Pulling thousands of rows from
# Postgres on every request would be wasteful, so we load them once into a
# numpy matrix kept in memory, refresh on a TTL, and do the cosine
# computation as one vectorized matrix multiply.

_vector_matrix: np.ndarray | None = None  # shape (N, 53), float32
_vector_norms:  np.ndarray | None = None  # shape (N,), precomputed for cosine
_vector_meta:   list[dict] = []           # parallel row metadata
_matrix_loaded_at: float = 0.0
_matrix_lock = asyncio.Lock()

# How long the in-memory matrix is considered fresh. After this, the next
# recommendation request triggers a reload from Postgres. The pool grows
# in the background (via prewarm and live user queries), so a few-minute
# TTL is enough to keep the recommender working with recent additions.
MATRIX_TTL_SECONDS = 300  # 5 minutes


async def _ensure_vector_matrix() -> None:
    """Load (or refresh) the in-memory matrix of all cached vectors."""
    global _vector_matrix, _vector_norms, _vector_meta, _matrix_loaded_at

    now = time.time()
    if _vector_matrix is not None and (now - _matrix_loaded_at) < MATRIX_TTL_SECONDS:
        return

    async with _matrix_lock:
        # Re-check inside the lock — another caller may have just loaded.
        now = time.time()
        if _vector_matrix is not None and (now - _matrix_loaded_at) < MATRIX_TTL_SECONDS:
            return

        entries = await cache.get_all_cached()

        if not entries:
            _vector_matrix = np.zeros((0, analyzer.VECTOR_DIM), dtype=np.float32)
            _vector_norms  = np.zeros((0,), dtype=np.float32)
            _vector_meta   = []
        else:
            _vector_matrix = np.array(
                [e["vector"] for e in entries], dtype=np.float32,
            )
            _vector_norms = np.linalg.norm(_vector_matrix, axis=1)
            _vector_meta  = entries

        _matrix_loaded_at = now


# ─── Midpoint nearest-neighbor recommendations ──────────────────────────────

async def get_recommendations(songs1: list[dict], songs2: list[dict]) -> list[dict]:
    """Recommend songs whose audio fingerprint sits near the midpoint of the
    two users' fingerprints, drawn from the entire cached pool.

    Returns up to 6 unique-artist recommendations (track name, artist, image,
    Deezer URL). Excludes any song or artist that appeared in the user inputs.

    Returns an empty list if the cache is empty or if neither user has any
    analyzable songs.
    """
    v1_songs = _valid_songs(songs1)
    v2_songs = _valid_songs(songs2)
    u1 = _user_vector(v1_songs)
    u2 = _user_vector(v2_songs)
    if u1 is None or u2 is None:
        return []

    # The midpoint = the geometric center between two users' tastes. Songs
    # near here are the "sonic intersection" of what they each individually
    # like — by construction, they should appeal to both.
    midpoint = ((u1 + u2) / 2.0).astype(np.float32)
    midpoint_norm = float(np.linalg.norm(midpoint))
    if midpoint_norm == 0.0:
        return []

    # Load (or refresh) the in-memory matrix.
    await _ensure_vector_matrix()
    if _vector_matrix is None or _vector_matrix.shape[0] == 0:
        return []

    # Vectorized cosine similarity:
    #   sim_i = (matrix[i] · midpoint) / (||matrix[i]|| · ||midpoint||)
    # One matrix-vector multiply, one elementwise divide, done.
    dots = _vector_matrix @ midpoint
    denom = (_vector_norms * midpoint_norm) + 1e-9
    similarities = dots / denom

    # Exclusion sets — don't recommend songs/artists the user already typed.
    exclude_track_ids: set[str] = set()
    exclude_artist_ids: set[str] = set()
    for s in songs1 + songs2:
        if s.get("track_id"):  exclude_track_ids.add(str(s["track_id"]))
        if s.get("deezer_id"): exclude_track_ids.add(str(s["deezer_id"]))
        if s.get("artist_id"): exclude_artist_ids.add(str(s["artist_id"]))

    # Walk sorted indices (highest similarity first), pick first 6 that
    # pass the filters AND have a unique artist.
    order = np.argsort(-similarities)

    final: list[dict] = []
    used_artists: set[str] = set()
    for idx in order:
        sim = float(similarities[idx])
        if sim <= 0.0:
            break  # cosine ≤ 0 means the song is no more similar than random

        entry     = _vector_meta[int(idx)]
        deezer_id = str(entry.get("deezer_id") or "")
        artist_id = str(entry.get("artist_id") or "")

        if deezer_id and deezer_id in exclude_track_ids:
            continue
        if artist_id and artist_id in exclude_artist_ids:
            continue
        if artist_id and artist_id in used_artists:
            continue

        final.append({
            "name":   entry.get("track_name")  or "",
            "artist": entry.get("artist_name") or "",
            "image":  entry.get("track_image") or "",
            "url":    entry.get("track_url")   or "",
        })
        if artist_id:
            used_artists.add(artist_id)

        if len(final) >= 6:
            break

    return final

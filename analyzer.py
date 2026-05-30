"""
analyzer.py — Real audio analysis for Music Byy

Replaces the metadata proxies (popularity rank, release year) with actual
Fourier-based signal analysis of each song's audio.

Pipeline:
  1. Download a short audio clip (Deezer 30s preview)
  2. Load with librosa at 22050 Hz mono
  3. Extract spectral, rhythmic, harmonic, and timbral features
  4. Pack features into a flat vector for cosine similarity math

The vector layout is fixed (see VECTOR_LAYOUT below) so downstream code can
slice it into subspaces for the per-category subscores (energy/tempo/mood).
"""

import asyncio
import os
import tempfile

import numpy as np

# librosa is heavy — import lazily inside functions that need it so the rest
# of the app starts fast even on cold boot.

import client
import itunes


# ─── Configuration ────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050          # Hz — half of CD quality, plenty for feature analysis
TARGET_DURATION = 30         # seconds of audio we analyze
N_MFCC = 13                  # standard count of MFCC coefficients
MIN_AUDIO_SECONDS = 5        # bail if clip is shorter than this

PREVIEW_DOWNLOAD_TIMEOUT = 10.0  # seconds


# ─── Feature vector layout ───────────────────────────────────────────────────
#
# The vector concatenates these groups in this exact order. Indices are used by
# match.py to slice into subspaces for the energy/tempo/mood subscores.
#
#   index 0           : tempo (BPM, normalized)
#   indices 1-7       : energy block (rms_mean, rms_std, centroid_mean,
#                       centroid_std, rolloff_mean, bandwidth_mean, zcr_mean)
#   indices 8-33      : timbre block (13 MFCC means + 13 MFCC stds)
#   indices 34-51     : mood block (12 chroma means + 6 tonnetz means)
#   index 52          : mode (1=major, 0=minor)
#
# Total dimension: 53

VECTOR_LAYOUT = {
    "tempo":  slice(0, 1),
    "energy": slice(1, 8),
    "timbre": slice(8, 34),
    "mood":   slice(34, 53),  # chroma + tonnetz + mode
}

VECTOR_DIM = 53


# Z-score normalization scales for the scalar features (mean, std).
# These are hand-tuned to approximate typical-music statistics so that after
# normalization the values land in a similar range to MFCCs/chroma. Once we
# have a populated cache we can re-derive these from real data.
NORMALIZATION = {
    "tempo":         (120.0, 30.0),
    "rms_mean":      (0.10,  0.05),
    "rms_std":       (0.05,  0.025),
    "centroid_mean": (2500.0, 1000.0),
    "centroid_std":  (800.0,  300.0),
    "rolloff_mean":  (5000.0, 2000.0),
    "bandwidth_mean":(2000.0, 600.0),
    "zcr_mean":      (0.08,  0.04),
}


# ─── Audio download ──────────────────────────────────────────────────────────

async def download_audio(url: str) -> bytes | None:
    """Fetch the preview MP3. Returns raw bytes or None on failure."""
    if not url:
        return None
    try:
        r = await client.http_client.get(url, timeout=PREVIEW_DOWNLOAD_TIMEOUT)
        if r.status_code == 200 and r.content:
            return r.content
    except Exception:
        pass
    return None


# ─── Mode estimation (major vs minor) ────────────────────────────────────────

# Krumhansl-Kessler key profiles. These are perceptual weights for each
# chromatic scale degree in major vs. minor keys, derived from listener
# experiments. Correlating a song's chroma distribution against these tells us
# whether it leans major (brighter/happier) or minor (darker/sadder).
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_mode(chroma_means: np.ndarray) -> int:
    """Return 1 for major, 0 for minor.

    Tries all 12 possible tonic positions and picks the rotation+mode combo
    with the strongest correlation to the Krumhansl-Kessler profile.
    """
    if chroma_means.sum() == 0:
        return 1  # default to major for silence/garbage

    best_major = max(
        float(np.dot(np.roll(chroma_means, -i), KK_MAJOR)) for i in range(12)
    )
    best_minor = max(
        float(np.dot(np.roll(chroma_means, -i), KK_MINOR)) for i in range(12)
    )
    return 1 if best_major >= best_minor else 0


# ─── Core librosa analysis (sync, CPU-bound) ────────────────────────────────

def _analyze_audio_bytes_sync(audio_bytes: bytes) -> dict | None:
    """Run librosa on the given audio bytes. Returns a feature dict or None.

    This function is synchronous and CPU-bound — call it inside
    asyncio.run_in_executor to avoid blocking the event loop.
    """
    # Lazy import: librosa pulls in numpy, scipy, numba, etc. and takes
    # ~1 second to import. Keep it out of startup.
    import librosa

    if not audio_bytes:
        return None

    # librosa.load wants a file path. Write the bytes to a temp file.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        y, sr = librosa.load(
            tmp_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=TARGET_DURATION,
        )

        if len(y) < sr * MIN_AUDIO_SECONDS:
            return None

        # ── Rhythm ────────────────────────────────────────────────────────
        # beat_track returns (tempo, beat_frames). tempo may be a numpy
        # scalar; cast to float.
        tempo_est, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_est)

        # ── Energy / loudness ────────────────────────────────────────────
        rms = librosa.feature.rms(y=y)[0]

        # ── Spectral shape (brightness, sharpness, noisiness) ────────────
        centroid  = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        zcr       = librosa.feature.zero_crossing_rate(y)[0]

        # ── Timbre via MFCCs ─────────────────────────────────────────────
        # MFCCs capture the spectral envelope. Mean across time for the
        # "average tone color"; std for how much it varies in the clip.
        mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_stds  = np.std(mfcc, axis=1)

        # ── Harmonic content ─────────────────────────────────────────────
        # Chroma collapses the spectrum into the 12 pitch classes — a
        # fingerprint of which notes are most active. Tonnetz projects
        # chroma onto a 6D tonal centroid space (intervals/triads).
        chroma       = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_means = np.mean(chroma, axis=1)

        try:
            y_harmonic   = librosa.effects.harmonic(y)
            tonnetz      = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            tonnetz_means = np.mean(tonnetz, axis=1)
        except Exception:
            tonnetz_means = np.zeros(6)

        mode = estimate_mode(chroma_means)

        return {
            "tempo":          tempo,
            "rms_mean":       float(np.mean(rms)),
            "rms_std":        float(np.std(rms)),
            "centroid_mean":  float(np.mean(centroid)),
            "centroid_std":   float(np.std(centroid)),
            "rolloff_mean":   float(np.mean(rolloff)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "zcr_mean":       float(np.mean(zcr)),
            "mfcc_means":     mfcc_means.tolist(),
            "mfcc_stds":      mfcc_stds.tolist(),
            "chroma_means":   chroma_means.tolist(),
            "tonnetz_means":  tonnetz_means.tolist(),
            "mode":           int(mode),
        }

    except Exception as e:
        # Don't blow up the whole request on one bad MP3.
        print(f"[analyzer] librosa error: {e}")
        return None

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─── Vector packing ──────────────────────────────────────────────────────────

def features_to_vector(f: dict) -> np.ndarray:
    """Convert a feature dict into a fixed-layout numpy float32 vector.

    Order matters: matches VECTOR_LAYOUT exactly.
    """
    parts: list[float] = []

    # ── Tempo (index 0) ──────────────────────────────────────────────────
    m, s = NORMALIZATION["tempo"]
    parts.append((f["tempo"] - m) / s)

    # ── Energy block (indices 1-7) ───────────────────────────────────────
    for key in ("rms_mean", "rms_std",
                "centroid_mean", "centroid_std",
                "rolloff_mean", "bandwidth_mean", "zcr_mean"):
        m, s = NORMALIZATION[key]
        parts.append((f[key] - m) / s)

    # ── Timbre block (indices 8-33): MFCC means + stds ───────────────────
    parts.extend(f["mfcc_means"])
    parts.extend(f["mfcc_stds"])

    # ── Mood block (indices 34-51): chroma + tonnetz + mode ──────────────
    parts.extend(f["chroma_means"])    # 12 dims
    parts.extend(f["tonnetz_means"])   # 6 dims
    parts.append(float(f["mode"]))     # 1 dim

    v = np.array(parts, dtype=np.float32)

    if v.shape[0] != VECTOR_DIM:
        raise ValueError(
            f"Vector dim mismatch: got {v.shape[0]}, expected {VECTOR_DIM}. "
            f"Check VECTOR_LAYOUT vs features_to_vector."
        )

    return v


# ─── Async public entrypoint ─────────────────────────────────────────────────

async def analyze_track(preview_url: str) -> dict | None:
    """Download the preview clip at this URL and run audio analysis.

    Low-level entrypoint. For most callers, prefer analyze_track_with_fallback
    which handles the iTunes fallback for missing Deezer previews.

    Returns:
        {
          "features": {...},      # the raw feature dict (for caching / UI)
          "vector":   [...],      # length-53 list of floats (for similarity)
        }
        or None if download or analysis failed.
    """
    audio_bytes = await download_audio(preview_url)
    if not audio_bytes:
        return None

    # librosa is CPU-bound and synchronous. Run it in the default thread pool
    # so we don't block the FastAPI event loop while it crunches.
    loop = asyncio.get_event_loop()
    features = await loop.run_in_executor(
        None, _analyze_audio_bytes_sync, audio_bytes
    )
    if not features:
        return None

    vector = features_to_vector(features)

    return {
        "features": features,
        "vector":   vector.tolist(),
    }


async def analyze_track_with_fallback(
    primary_url: str | None,
    track_name: str | None = None,
    artist_name: str | None = None,
) -> dict | None:
    """Try the primary preview URL first; fall back to iTunes Search API.

    The waterfall:
        1. If primary_url (Deezer preview) is non-empty, attempt to analyze
           it. If that succeeds, return immediately.
        2. If step 1 failed AND we have a track name + artist, search iTunes
           for an alternative preview URL and try analyzing that.
        3. Otherwise return None.

    This keeps Deezer as the primary source (fastest, broadest coverage)
    while filling gaps for long-tail tracks Deezer hasn't indexed previews
    for. The whole pipeline stays TOS-clean.
    """
    if primary_url:
        result = await analyze_track(primary_url)
        if result is not None:
            return result

    # Primary failed or was empty. Try iTunes if we have enough info to search.
    if track_name:
        itunes_url = await itunes.find_preview_url(track_name, artist_name or "")
        if itunes_url:
            return await analyze_track(itunes_url)

    return None

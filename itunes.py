"""
itunes.py — iTunes Search API fallback for preview audio.

Used when Deezer doesn't have a 30-second preview URL for a track (which
happens for some long-tail or recently-released songs). The iTunes Search
API is free, requires no auth, and returns a 30-second preview URL hosted
on Apple's CDN.

API reference: https://performance-partners.apple.com/search-api
"""

import client

ITUNES_SEARCH = "https://itunes.apple.com/search"
ITUNES_TIMEOUT = 5.0  # seconds — keep tight; this is a fallback


async def find_preview_url(track_name: str, artist_name: str) -> str | None:
    """Search iTunes for a track and return its 30-second preview URL.

    Returns None if nothing found, the request fails, or the matched track
    has no preview (rare but possible).

    The preview is typically an m4a file ~30 seconds long on Apple's CDN.
    librosa handles m4a fine as long as ffmpeg is installed on the host
    (which we install via the Dockerfile).
    """
    if not track_name:
        return None

    # Build a query string. Including the artist disambiguates covers and
    # remixes. iTunes ranks by relevance so the first match is almost always
    # the correct one when artist is included.
    query = f"{track_name} {artist_name}".strip()

    try:
        r = await client.http_client.get(
            ITUNES_SEARCH,
            params={
                "term":   query,
                "media":  "music",
                "entity": "song",
                "limit":  1,
            },
            timeout=ITUNES_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None

    results = data.get("results", [])
    if not results:
        return None

    return results[0].get("previewUrl") or None

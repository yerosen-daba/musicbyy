FROM python:3.11-slim

# System-level deps:
#   ffmpeg  — librosa uses audioread which calls ffmpeg to decode MP3/M4A.
#             Without this, audio analysis fails on any non-WAV input.
#   libsndfile1 — soundfile dependency for native audio I/O.
#   build-essential — needed for compiling some librosa/scipy sub-deps if
#                     the wheel cache misses; safe to keep slim image small.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so Docker layer cache is reused when
# only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY . .

# Run the FastAPI server.
# Port 7860 is Hugging Face Spaces' default Docker port. Render uses this
# value via the $PORT env var, so the same image runs on either platform.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

# ---- build stage ----
FROM python:3.11-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serve.txt .
# torch: CPU-only wheel explicitly, via its own index - the default PyPI
# resolution pulls the full CUDA build (~10x larger) with zero benefit on
# a machine with no GPU passthrough into the container.
RUN pip install --no-cache-dir --prefix=/install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements-serve.txt

# ---- runtime stage ----
FROM python:3.11-slim
# libgl1/libglib2.0-0 are needed by OpenCV/MediaPipe at runtime even
# though they're not obviously "ML" packages - a silent
# "ImportError: libGL.so.1" if omitted. ffmpeg backs OpenCV's video
# decoding for the uploaded .mp4/.avi/.mov files.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /install /usr/local
WORKDIR /app
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
# The window=45 model, chosen in EXPERIMENTS.md - config lives inside the
# .pt file itself (see api/inference.py), so no separate sidecar file.
COPY checkpoints_window45/best_model.pt models/best_model.pt

ENV FALLDET_MODEL_PATH=models/best_model.pt
ENV FALLDET_THRESHOLD=0.45
ENV FALLDET_DEVICE=cpu
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

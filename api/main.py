"""FastAPI app exposing /predict for fall-detection inference on an
uploaded video file.

Model + threshold are configured via environment variables so the same
image/code can serve different checkpoints without a code change:
  FALLDET_MODEL_PATH        default: checkpoints_window45/best_model.pt
  FALLDET_MODEL_TAG         default: unset - if set (and FALLDET_DATABASE_URL is
                            also set), resolves the checkpoint via the model
                            registry (api/registry.py) instead of FALLDET_MODEL_PATH
  FALLDET_THRESHOLD         default: 0.45
  FALLDET_DEVICE            default: cpu
  FALLDET_API_KEY           default: unset (auth disabled) - see api/auth.py
  FALLDET_MAX_UPLOAD_BYTES  default: 200MB
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.auth import require_api_key
from api.errors import PayloadTooLargeError, UnsupportedFileTypeError, register_error_handlers
from api.inference import InferenceEngine, LiveInferenceEngine
from api.monitoring import log_inference
from api.registry import resolve_model_path
from api.schemas import PredictResponse
from api.ws import router as ws_router

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
UPLOAD_CHUNK_BYTES = 1024 * 1024  # 1MB - read in chunks so an oversized
# upload is rejected as soon as it crosses the limit, not after it's
# already been fully buffered into memory/disk.

MODEL_TAG = os.environ.get("FALLDET_MODEL_TAG")
MODEL_PATH = (
    (MODEL_TAG and resolve_model_path(MODEL_TAG))
    or os.environ.get("FALLDET_MODEL_PATH", "checkpoints_window45/best_model.pt")
)
THRESHOLD = float(os.environ.get("FALLDET_THRESHOLD", "0.45"))
DEVICE = os.environ.get("FALLDET_DEVICE", "cpu")
MAX_UPLOAD_BYTES = int(os.environ.get("FALLDET_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

app = FastAPI(title="Fall Detection API")
register_error_handlers(app)

# Loaded once at process startup, not per-request - model loading is
# expensive and this is the classic FastAPI mistake to avoid.
engine = InferenceEngine(MODEL_PATH, device=DEVICE, threshold=THRESHOLD)
app.state.live_engine = LiveInferenceEngine(MODEL_PATH, device=DEVICE, threshold=THRESHOLD)
app.include_router(ws_router)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/demo")
async def demo() -> FileResponse:
    return FileResponse(STATIC_DIR / "ws_client.html")


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_tag": MODEL_TAG,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_api_key)])
async def predict(file: UploadFile) -> PredictResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"unsupported file type '{suffix}' - expected one of {sorted(ALLOWED_EXTENSIONS)}"
        )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
            total_bytes = 0
            while True:
                chunk = await file.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise PayloadTooLargeError(
                        f"upload exceeds the {MAX_UPLOAD_BYTES}-byte limit"
                    )
                tmp.write(chunk)

        # InferenceEngine raises typed FallDetectionError subclasses
        # (VideoReadError, VideoTooShortError, NoPersonDetectedError) -
        # register_error_handlers() above turns any of those into the
        # right HTTP response, so nothing to catch here.
        result = engine.predict_video(tmp_path)
        log_inference(result, source="predict")
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

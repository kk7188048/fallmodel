"""WebSocket live-inference endpoint.

Protocol: client sends one raw JPEG frame (binary message) at a time. The
server:
  1. decodes it, runs it through `falldet.pose.estimate_frame()` using a
     Pose context kept open for the whole connection (recreating it per
     frame would be prohibitively expensive), and appends the resulting
     (33, 4) keypoints to a `deque(maxlen=window_size)` - sliding-window
     eviction for free as new frames arrive.
  2. while the buffer isn't full yet: replies
     `{"buffer_full": false, "frames_collected": n}` so the client can
     show a "warming up" state instead of nothing.
  3. once full: reruns `interpolate_missing_frames` -> `normalize_pose_
     sequence` -> `assemble_feature_vector` over the whole buffered window
     (the exact same functions `pipeline.py`/`api/inference.py` use - not
     re-derived here) and runs the model, replying
     `{"buffer_full": true, "prediction": "fall"|"adl", "confidence": ...}`.

**Constraint worth stating out loud, not hiding**: the model is
bidirectional, so it needs the FULL window before it can predict at all.
Live detection therefore has a built-in lag of `window_size` frames at
whatever rate the client sends them (e.g. 45 frames @ 15fps sent =~ 3
seconds before the first real prediction, and every prediction after that
reflects events up to ~1 window-length ago). This is a structural property
of the architecture, not a bug to optimize away here.

`FALLDET_WS_INFER_EVERY` (default 1) controls how many new frames
accumulate between inference runs, in case full-frame-rate inference is
too slow for the CPU it's running on - raise it to trade prediction
latency for throughput.
"""
from __future__ import annotations

import os
from collections import deque

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from falldet.pose import create_pose, estimate_frame

from api.inference import LiveInferenceEngine
from api.schemas import WebSocketState

router = APIRouter()

INFER_EVERY = int(os.environ.get("FALLDET_WS_INFER_EVERY", "1"))


@router.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket) -> None:
    await websocket.accept()

    # Loaded once at app startup (api/main.py), shared across every
    # connection - same "don't reload the model per request" principle
    # as InferenceEngine in api/inference.py.
    live_engine: LiveInferenceEngine = websocket.app.state.live_engine
    pose_cfg = live_engine.pose_cfg
    window_size = live_engine.window_size

    buffer: deque[np.ndarray] = deque(maxlen=window_size)
    frames_since_infer = 0

    with create_pose(
        pose_cfg["model_complexity"],
        pose_cfg["min_detection_confidence"],
        pose_cfg["min_tracking_confidence"],
    ) as pose:
        try:
            while True:
                data = await websocket.receive_bytes()

                frame_bytes = np.frombuffer(data, dtype=np.uint8)
                bgr_frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
                if bgr_frame is None:
                    await websocket.send_json({"error": "could not decode frame as an image"})
                    continue

                keypoints = estimate_frame(pose, bgr_frame)
                buffer.append(keypoints)

                if len(buffer) < window_size:
                    await websocket.send_json(
                        WebSocketState(buffer_full=False, frames_collected=len(buffer)).model_dump()
                    )
                    continue

                frames_since_infer += 1
                if frames_since_infer < INFER_EVERY:
                    continue
                frames_since_infer = 0

                raw_keypoints = np.stack(buffer, axis=0)
                prediction, confidence = live_engine.predict_window(raw_keypoints)

                await websocket.send_json(
                    WebSocketState(
                        buffer_full=True,
                        frames_collected=window_size,
                        prediction=prediction,
                        confidence=confidence,
                    ).model_dump()
                )
        except WebSocketDisconnect:
            pass

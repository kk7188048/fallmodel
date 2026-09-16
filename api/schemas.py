"""Pydantic request/response models for the fall-detection API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class PredictResponse(BaseModel):
    predicted_class: Literal["fall", "adl"]
    confidence: float
    latency_ms: float
    n_windows_evaluated: int  # a video can span multiple windows - see InferenceEngine


class WebSocketState(BaseModel):
    """Streamed to the client on every frame - both while the sliding
    buffer is still filling and once real predictions start.
    """
    buffer_full: bool
    frames_collected: int
    prediction: Literal["fall", "adl"] | None = None
    confidence: float | None = None

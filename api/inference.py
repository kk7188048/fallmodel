"""Thin inference wrapper around the trained model.

Reuses `falldet.pipeline.process_video()` completely unchanged from
training - this is deliberate and non-negotiable: any preprocessing
difference between training and serving (train/serve skew) silently tanks
accuracy in a way that's very hard to debug later, since nothing errors,
it just quietly predicts worse. Every pose/windowing parameter here comes
straight from the checkpoint's own saved config (see train.py - the full
merged config dict is saved alongside the weights), not from a separate
file that could drift out of sync with what the model actually trained on.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from falldet.features import assemble_feature_vector
from falldet.model import FallDetectionModel
from falldet.normalize import interpolate_missing_frames, normalize_pose_sequence
from falldet.pipeline import process_video

from api.errors import NoPersonDetectedError, VideoReadError, VideoTooShortError
from api.schemas import PredictResponse


def _load_checkpoint(checkpoint_path: str | Path, device: str) -> tuple[FallDetectionModel, dict]:
    """Shared by InferenceEngine and LiveInferenceEngine - both load the
    exact same checkpoint format (weights + the full training config
    saved alongside them), just consume it differently downstream (whole
    video vs. one accumulated window at a time).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    full_config = checkpoint["config"]

    model = FallDetectionModel.from_config(full_config, input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, full_config


class InferenceEngine:
    """Loads the model once at construction (not per-request - model
    loading is the classic FastAPI performance mistake to avoid).
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", threshold: float = 0.5):
        self.model, full_config = _load_checkpoint(checkpoint_path, device)
        self.device = device
        self.threshold = threshold
        self.pose_cfg = full_config["pose"]
        self.window_cfg = full_config["windowing"]

    def predict_video(self, video_path: str | Path) -> PredictResponse:
        start = time.perf_counter()

        try:
            result = process_video(
                str(video_path),
                model_complexity=self.pose_cfg["model_complexity"],
                min_detection_confidence=self.pose_cfg["min_detection_confidence"],
                min_tracking_confidence=self.pose_cfg["min_tracking_confidence"],
                window_size=self.window_cfg["window_size"],
                stride=self.window_cfg["stride"],
            )
        except IOError as exc:
            raise VideoReadError(f"could not read video: {exc}") from exc

        windows = result["windows"]

        if windows.shape[0] == 0:
            raise VideoTooShortError(
                f"video is shorter than the required {self.window_cfg['window_size']}-frame "
                f"window ({result['num_frames']} frames found)"
            )

        # A window can come out all-NaN if pose detection failed on every
        # frame it covers (e.g. no person visible anywhere in that span) -
        # interpolation only fills *gaps* between valid frames, it can't
        # invent a pose from nothing. Drop those rather than feed NaN into
        # the model, which would silently produce a NaN (garbage) prediction.
        finite_mask = np.isfinite(windows).all(axis=(1, 2))
        windows = windows[finite_mask]
        if windows.shape[0] == 0:
            raise NoPersonDetectedError("no person detected in any frame of this video")

        with torch.no_grad():
            x = torch.from_numpy(windows).float().to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        # Aggregation across windows: a video is rarely exactly one window
        # long. We favor recall - ANY window crossing the threshold flags
        # the whole clip as "fall" - consistent with M3's class-weighted
        # loss: a missed fall is worse than a false alarm, so one confident
        # window should be enough, not a majority vote across all windows.
        any_fall = bool((probs >= self.threshold).any())
        if any_fall:
            predicted_class = "fall"
            confidence = float(probs.max())
        else:
            predicted_class = "adl"
            confidence = float((1 - probs).mean())

        latency_ms = (time.perf_counter() - start) * 1000

        return PredictResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            latency_ms=latency_ms,
            n_windows_evaluated=int(windows.shape[0]),
        )


class LiveInferenceEngine:
    """Same loaded-once philosophy as InferenceEngine, but for the
    websocket path: takes a buffered raw-keypoint window (already
    extracted frame-by-frame by the caller via falldet.pose.estimate_frame)
    instead of a video file path, since a live stream never has "the
    whole video" to hand to process_video() at once.
    """

    def __init__(self, checkpoint_path: str | Path, device: str = "cpu", threshold: float = 0.5):
        self.model, full_config = _load_checkpoint(checkpoint_path, device)
        self.device = device
        self.threshold = threshold
        self.pose_cfg = full_config["pose"]
        self.window_size = full_config["windowing"]["window_size"]

    def predict_window(self, raw_keypoints: np.ndarray) -> tuple[str | None, float | None]:
        """raw_keypoints: (window_size, 33, 4) buffered raw pose output,
        one row per incoming frame in arrival order.

        Returns (None, None) - not an exception - if the window is
        unusable (all-NaN after interpolation, i.e. nobody was ever
        detected across the whole buffered window): a live stream should
        keep running and try the next frame, not error out and drop the
        connection over one bad window.
        """
        sequence = interpolate_missing_frames(raw_keypoints)
        sequence = normalize_pose_sequence(sequence)
        features = assemble_feature_vector(sequence)

        if not np.isfinite(features).all():
            return None, None

        with torch.no_grad():
            x = torch.from_numpy(features[None]).float().to(self.device)
            logits = self.model(x)
            prob = torch.softmax(logits, dim=-1)[0, 1].item()

        prediction = "fall" if prob >= self.threshold else "adl"
        confidence = prob if prediction == "fall" else 1 - prob
        return prediction, confidence

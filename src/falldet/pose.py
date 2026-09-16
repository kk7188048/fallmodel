"""M2 - MediaPipe pose estimation wrapper.

Uses the legacy `mediapipe.python.solutions.pose` API. mediapipe>=1.0
replaced it with a Tasks-based `PoseLandmarker`, but that API's
TensorsToDetectionsCalculator crashes on macOS ("Metal GraphService
unavailable") as of this writing - see requirements.txt, which pins
mediapipe==0.10.21 for this reason.
"""
from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 33
# (x, y, z, visibility) per landmark
NUM_CHANNELS = 4

# Indices into the 33 MediaPipe pose landmarks that normalize.py/features.py
# key off of. See https://developers.google.com/mediapipe for the full list.
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24


def create_pose(
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """Open a MediaPipe Pose context. Centralizes the mediapipe import and
    construction so callers (estimate_video, the live websocket handler)
    don't each import mediapipe.python.solutions directly - one place
    that knows how a Pose context gets built.
    """
    from mediapipe.python.solutions import pose as mp_pose

    return mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )


def estimate_frame(pose, bgr_frame: np.ndarray) -> np.ndarray:
    """Run pose estimation on a single BGR frame using an already-open
    Pose context (see create_pose). Returns (33, 4) keypoints, or all-NaN
    if no person was detected.

    Shared by estimate_video() (looping over a video file) and the live
    websocket handler (looping over incoming camera frames) - the actual
    MediaPipe call lives in exactly one place, not duplicated per caller.
    """
    import cv2

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)
    if result.pose_landmarks is None:
        return np.full((NUM_LANDMARKS, NUM_CHANNELS), np.nan)
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
    )


def estimate_video(
    video_path: str,
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> np.ndarray:
    """Run pose estimation over every frame of a video.

    Returns an (T, 33, 4) array of (x, y, z, visibility), x/y normalized to
    [0, 1] by MediaPipe relative to frame width/height. Frames where no
    person was detected are filled with NaN so downstream code can tell
    "missing" apart from a genuine (0, 0) landmark.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"could not open video: {video_path}")

    frames = []
    try:
        with create_pose(model_complexity, min_detection_confidence, min_tracking_confidence) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(estimate_frame(pose, frame))
    finally:
        cap.release()

    if not frames:
        return np.empty((0, NUM_LANDMARKS, NUM_CHANNELS))
    return np.stack(frames, axis=0)

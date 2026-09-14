#!/usr/bin/env python
"""CLI: live fall-detection demo on a video file or webcam.

Runs the exact same pose -> normalize -> feature pipeline used in
training, over a rolling window of the last `window_size` frames, and
overlays the trained model's fall/ADL prediction + probability on each
frame in real time.

Usage:
    python scripts/run_demo.py --source path/to/video.avi
    python scripts/run_demo.py --source webcam
    python scripts/run_demo.py --source path/to/video.avi --save_to out.mp4
    python scripts/run_demo.py --source webcam --threshold 0.45
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from mediapipe.python.solutions import pose as mp_pose

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from falldet.features import assemble_feature_vector
from falldet.model import FallDetectionModel
from falldet.normalize import interpolate_missing_frames, normalize_pose_sequence
from falldet.pose import NUM_CHANNELS, NUM_LANDMARKS


def load_model(checkpoint_path: str) -> FallDetectionModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FallDetectionModel.from_config(checkpoint["config"], input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="video file path, or 'webcam'")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="defaults to <checkpoints_dir>/best_model.pt")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="fall decision threshold - see scripts/threshold_sweep.py for tuning this",
    )
    parser.add_argument(
        "--save_to",
        default=None,
        help="write the annotated video to this path instead of opening a live window "
        "(use this over SSH / headless environments with no display)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint_path = args.checkpoint or str(Path(config["paths"]["checkpoints_dir"]) / "best_model.pt")
    model = load_model(checkpoint_path)

    window_size = config["windowing"]["window_size"]
    pose_cfg = config["pose"]

    source = 0 if args.source == "webcam" else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_to:
        writer = cv2.VideoWriter(args.save_to, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    keypoint_buffer: deque = deque(maxlen=window_size)
    label, prob = "warming up", 0.0
    frame_count = 0
    fall_frame_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=pose_cfg["model_complexity"],
        min_detection_confidence=pose_cfg["min_detection_confidence"],
        min_tracking_confidence=pose_cfg["min_tracking_confidence"],
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks is None:
                keypoints = np.full((NUM_LANDMARKS, NUM_CHANNELS), np.nan)
            else:
                keypoints = np.array(
                    [[lm.x, lm.y, lm.z, lm.visibility] for lm in result.pose_landmarks.landmark]
                )
            keypoint_buffer.append(keypoints)

            if len(keypoint_buffer) == window_size:
                sequence = np.stack(keypoint_buffer, axis=0)
                sequence = interpolate_missing_frames(sequence)
                sequence = normalize_pose_sequence(sequence)
                features = assemble_feature_vector(sequence)
                if np.isfinite(features).all():
                    x = torch.from_numpy(features[None]).float()
                    with torch.no_grad():
                        logits = model(x)
                        prob = torch.softmax(logits, dim=-1)[0, 1].item()
                    label = "FALL" if prob >= args.threshold else "ADL"
                    if label == "FALL":
                        fall_frame_count += 1

            color = (0, 0, 255) if label == "FALL" else (0, 200, 0)
            cv2.putText(frame, f"{label} ({prob:.2f})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            if writer is not None:
                writer.write(frame)
            else:
                cv2.imshow("fall-detection demo (press q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"saved annotated video to {args.save_to}")
    cv2.destroyAllWindows()

    print(f"processed {frame_count} frames, flagged FALL on {fall_frame_count} of them")


if __name__ == "__main__":
    main()

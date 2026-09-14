"""process_video() on a 2-second synthetic mp4."""
import numpy as np

from falldet.features import FEATURE_DIM
from falldet.pipeline import process_video


def test_process_video_on_synthetic_clip_no_crash(make_synthetic_video):
    # Solid-color synthetic frames contain no detectable person, so
    # detection_rate is 0 and there is nothing to interpolate keypoints
    # from - windows legitimately come out NaN. That is the pipeline being
    # honest about "no idea", not a bug; correctness of the numeric path
    # on frames that *do* get real detections is covered by test_features.py
    # and test_normalize.py, and was verified manually against real Le2i
    # footage during development.
    video_path = make_synthetic_video(seconds=2.0, fps=10)
    result = process_video(video_path, window_size=10, stride=5)

    assert result["num_frames"] == 20
    assert result["detection_rate"] == 0.0
    assert result["windows"].shape[1:] == (10, FEATURE_DIM)
    assert result["windows"].shape[0] == result["frame_ranges"].shape[0]
    assert result["labels"].shape[0] == result["windows"].shape[0]


def test_process_video_adl_labels_all_zero(make_synthetic_video):
    video_path = make_synthetic_video(seconds=2.0, fps=10)
    result = process_video(video_path, window_size=10, stride=5)
    assert np.all(result["labels"] == 0)


def test_process_video_too_short_clip_yields_no_windows(make_synthetic_video):
    video_path = make_synthetic_video(seconds=0.5, fps=10)  # 5 frames
    result = process_video(video_path, window_size=30, stride=15)
    assert result["windows"].shape[0] == 0
    assert result["num_frames"] == 5

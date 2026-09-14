import numpy as np

from falldet.windowing import label_windows, sliding_windows


def test_sliding_windows_shapes_and_ranges():
    features = np.arange(100).reshape(50, 2).astype(float)
    windows, frame_ranges = sliding_windows(features, window_size=10, stride=5)
    assert windows.shape == (9, 10, 2)
    assert frame_ranges.shape == (9, 2)
    assert list(frame_ranges[0]) == [0, 9]
    assert list(frame_ranges[1]) == [5, 14]
    assert list(frame_ranges[-1]) == [40, 49]


def test_sliding_windows_content_matches_source():
    features = np.arange(20).reshape(10, 2).astype(float)
    windows, frame_ranges = sliding_windows(features, window_size=4, stride=2)
    for w, (s, e) in zip(windows, frame_ranges):
        assert np.array_equal(w, features[s : e + 1])


def test_sliding_windows_too_short_video_yields_no_windows():
    features = np.zeros((5, 3))
    windows, frame_ranges = sliding_windows(features, window_size=10, stride=5)
    assert windows.shape == (0, 10, 3)
    assert frame_ranges.shape == (0, 2)


def test_label_windows_adl_video_all_zero():
    frame_ranges = np.array([[0, 9], [10, 19]])
    labels = label_windows(frame_ranges, None, None)
    assert np.array_equal(labels, [0, 0])

    labels_zero_header = label_windows(frame_ranges, 0, 0)
    assert np.array_equal(labels_zero_header, [0, 0])


def test_label_windows_marks_overlapping_windows():
    # fall spans annotation frames 54-86 (1-indexed) -> 0-indexed 53-85
    frame_ranges = np.array(
        [[0, 29], [15, 44], [30, 59], [45, 74], [60, 89], [75, 104], [90, 119]]
    )
    labels = label_windows(frame_ranges, fall_start_frame=54, fall_end_frame=86)
    assert list(labels) == [0, 0, 1, 1, 1, 1, 0]


def test_label_windows_exact_boundary_overlap():
    frame_ranges = np.array([[0, 9], [10, 19]])
    # fall is exactly frame 10 (1-indexed) -> 0-indexed frame 9, in the first window only
    labels = label_windows(frame_ranges, fall_start_frame=10, fall_end_frame=10)
    assert list(labels) == [1, 0]

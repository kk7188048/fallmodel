from api.monitoring import log_inference
from api.schemas import PredictResponse


def test_log_inference_creates_file_with_header(tmp_path):
    log_path = tmp_path / "log.csv"
    result = PredictResponse(predicted_class="fall", confidence=0.9, latency_ms=12.3, n_windows_evaluated=3)

    log_inference(result, source="predict", log_path=log_path)

    lines = log_path.read_text().strip().splitlines()
    assert lines[0] == "timestamp,source,predicted_class,confidence,latency_ms"
    assert len(lines) == 2
    assert "fall" in lines[1]


def test_log_inference_appends_without_duplicate_header(tmp_path):
    log_path = tmp_path / "log.csv"
    result = PredictResponse(predicted_class="adl", confidence=0.2, latency_ms=5.0, n_windows_evaluated=1)

    log_inference(result, source="predict", log_path=log_path)
    log_inference(result, source="websocket", log_path=log_path)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert lines[0].count("timestamp") == 1


def test_log_inference_creates_parent_dirs(tmp_path):
    log_path = tmp_path / "nested" / "dir" / "log.csv"
    result = PredictResponse(predicted_class="fall", confidence=0.8, latency_ms=1.0, n_windows_evaluated=1)

    log_inference(result, source="predict", log_path=log_path)

    assert log_path.exists()

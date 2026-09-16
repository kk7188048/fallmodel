import pandas as pd
import pytest

from scripts.plot_drift import check_drift_alert, latest_rolling_confidence, plot_confidence_drift


def _write_log(path, n=20):
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": 1000 + i,
                "source": "predict",
                "predicted_class": "fall" if i % 3 == 0 else "adl",
                "confidence": 0.5 + 0.01 * i,
                "latency_ms": 100.0,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_plot_confidence_drift_writes_file(tmp_path):
    log_path = tmp_path / "log.csv"
    _write_log(log_path)

    out_path = plot_confidence_drift(log_path, window=5, out_path=tmp_path / "drift.png")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_confidence_drift_missing_log_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_confidence_drift(tmp_path / "does_not_exist.csv")


def test_plot_confidence_drift_empty_log_raises(tmp_path):
    # An empty-but-existing CSV and a missing CSV are indistinguishable
    # from read_recent_logs()'s perspective (both -> no rows to plot), so
    # both now raise the same FileNotFoundError.
    log_path = tmp_path / "empty.csv"
    pd.DataFrame(columns=["timestamp", "source", "predicted_class", "confidence", "latency_ms"]).to_csv(
        log_path, index=False
    )
    with pytest.raises(FileNotFoundError):
        plot_confidence_drift(log_path)


def test_plot_confidence_drift_window_larger_than_data(tmp_path):
    log_path = tmp_path / "log.csv"
    _write_log(log_path, n=5)

    out_path = plot_confidence_drift(log_path, window=50, out_path=tmp_path / "drift.png")

    assert out_path.exists()


def test_check_drift_alert_below_threshold():
    assert check_drift_alert(0.4, threshold=0.5) is True


def test_check_drift_alert_at_or_above_threshold():
    assert check_drift_alert(0.5, threshold=0.5) is False
    assert check_drift_alert(0.9, threshold=0.5) is False


def test_latest_rolling_confidence_reflects_recent_data(tmp_path):
    log_path = tmp_path / "log.csv"
    # first 10 rows high confidence, last 10 rows low - the rolling mean
    # over a small window should reflect the recent (low) run, not the
    # overall average.
    rows = []
    for i in range(10):
        rows.append({"timestamp": 1000 + i, "source": "predict", "predicted_class": "adl", "confidence": 0.95, "latency_ms": 100.0})
    for i in range(10):
        rows.append({"timestamp": 1010 + i, "source": "predict", "predicted_class": "adl", "confidence": 0.3, "latency_ms": 100.0})
    pd.DataFrame(rows).to_csv(log_path, index=False)

    latest = latest_rolling_confidence(log_path, window=5)

    assert latest < 0.5


def test_cli_alert_exits_nonzero_below_threshold(tmp_path, capsys):
    import sys as _sys

    from scripts.plot_drift import main

    log_path = tmp_path / "log.csv"
    _write_log(log_path, n=10)
    pd.read_csv(log_path).assign(confidence=0.1).to_csv(log_path, index=False)

    _sys.argv = ["plot_drift.py", "--log", str(log_path), "--alert_threshold", "0.9"]
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert "ALERT" in capsys.readouterr().err


def test_cli_alert_no_exit_above_threshold(tmp_path, capsys):
    import sys as _sys

    from scripts.plot_drift import main

    log_path = tmp_path / "log.csv"
    _write_log(log_path, n=10)
    pd.read_csv(log_path).assign(confidence=0.95).to_csv(log_path, index=False)

    _sys.argv = ["plot_drift.py", "--log", str(log_path), "--alert_threshold", "0.5"]
    main()  # should not raise/exit

    assert "ok:" in capsys.readouterr().out

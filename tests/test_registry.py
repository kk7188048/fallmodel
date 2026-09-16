"""Unit tests for the parts of api/registry.py that don't need a real
database. The actual Postgres read/write path (register_model,
resolve_model_path, list_models against a real connection) is verified
live against a real database, the same way api/monitoring.py's Postgres
path was - see README.
"""
from api.registry import resolve_model_path, sha256_of_file


def test_sha256_of_file_is_deterministic(tmp_path):
    path = tmp_path / "data.bin"
    path.write_bytes(b"some model weights, pretend")

    first = sha256_of_file(path)
    second = sha256_of_file(path)

    assert first == second
    assert len(first) == 64  # hex-encoded sha256


def test_sha256_of_file_differs_for_different_content(tmp_path):
    path_a = tmp_path / "a.bin"
    path_b = tmp_path / "b.bin"
    path_a.write_bytes(b"content A")
    path_b.write_bytes(b"content B")

    assert sha256_of_file(path_a) != sha256_of_file(path_b)


def test_resolve_model_path_returns_none_without_database_url(monkeypatch):
    monkeypatch.delenv("FALLDET_DATABASE_URL", raising=False)

    result = resolve_model_path("production", database_url=None)

    assert result is None

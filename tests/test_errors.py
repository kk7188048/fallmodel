from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.errors import (
    FallDetectionError,
    NoPersonDetectedError,
    UnsupportedFileTypeError,
    VideoReadError,
    VideoTooShortError,
    register_error_handlers,
)


def test_exception_subclasses_carry_status_and_code():
    assert UnsupportedFileTypeError("x").status_code == 400
    assert UnsupportedFileTypeError("x").error_code == "unsupported_file_type"
    assert VideoReadError("x").status_code == 400
    assert VideoTooShortError("x").status_code == 422
    assert NoPersonDetectedError("x").status_code == 422


def test_exception_message_is_preserved():
    exc = NoPersonDetectedError("nobody visible")
    assert exc.message == "nobody visible"
    assert str(exc) == "nobody visible"


def _make_test_app():
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/boom/{kind}")
    async def boom(kind: str):
        errors = {
            "unsupported": UnsupportedFileTypeError("bad extension"),
            "read": VideoReadError("corrupt file"),
            "short": VideoTooShortError("too short"),
            "no_person": NoPersonDetectedError("nobody there"),
        }
        raise errors[kind]

    return app


def test_handler_formats_response_correctly():
    client = TestClient(_make_test_app())

    response = client.get("/boom/no_person")

    assert response.status_code == 422
    assert response.json() == {"error": "no_person_detected", "detail": "nobody there"}


def test_handler_maps_each_error_to_its_own_status():
    client = TestClient(_make_test_app())

    assert client.get("/boom/unsupported").status_code == 400
    assert client.get("/boom/read").status_code == 400
    assert client.get("/boom/short").status_code == 422
    assert client.get("/boom/no_person").status_code == 422


def test_base_class_is_caught_too():
    app = FastAPI()
    register_error_handlers(app)

    @app.get("/generic")
    async def generic():
        raise FallDetectionError("something generic")

    client = TestClient(app)
    response = client.get("/generic")

    assert response.status_code == 500
    assert response.json() == {"error": "internal_error", "detail": "something generic"}

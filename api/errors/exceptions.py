"""Domain-specific exceptions for the fall-detection application.

Centralized here rather than scattered as inline HTTPException raises in
each route, so every entry point - /predict, the websocket handler, any
future endpoint - maps the same underlying failure to the same HTTP
status and the same response shape, instead of each one reinventing its
own error handling.

To add a new error case anywhere in the app: subclass FallDetectionError,
set status_code + error_code, raise it. No other wiring needed - the
handler registered in handlers.py catches the base class and formats any
subclass automatically.
"""
from __future__ import annotations


class FallDetectionError(Exception):
    """Base class for all application-specific errors. Carries the HTTP
    status code it should map to, so one generic handler can translate
    any subclass into a response without a growing if/elif chain.
    """

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class UnsupportedFileTypeError(FallDetectionError):
    """Uploaded file extension isn't one we know how to read."""

    status_code = 400
    error_code = "unsupported_file_type"


class VideoReadError(FallDetectionError):
    """The video file couldn't be opened/decoded at all (corrupt, wrong
    codec, truncated upload, etc.).
    """

    status_code = 400
    error_code = "video_read_error"


class VideoTooShortError(FallDetectionError):
    """Video has fewer frames than the model's window_size - not enough
    context to build even one prediction window.
    """

    status_code = 422
    error_code = "video_too_short"


class NoPersonDetectedError(FallDetectionError):
    """Pose estimation found no person in any frame of the video (every
    window that could be built came out fully NaN after interpolation).
    """

    status_code = 422
    error_code = "no_person_detected"


class ModelNotReadyError(FallDetectionError):
    """The inference engine/model failed to load or isn't initialized yet."""

    status_code = 503
    error_code = "model_not_ready"


class PayloadTooLargeError(FallDetectionError):
    """Uploaded file exceeds the configured maximum size."""

    status_code = 413
    error_code = "payload_too_large"


class UnauthorizedError(FallDetectionError):
    """Missing or incorrect API key."""

    status_code = 401
    error_code = "unauthorized"

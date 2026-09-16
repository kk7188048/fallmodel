"""FastAPI exception handler wiring for api/errors/exceptions.py.

register_error_handlers(app) is called once from api/main.py, and should
be called the same way from any other FastAPI app entry point this
project grows (e.g. if the websocket server ever needs its own separate
app instance) - every FallDetectionError subclass then gets turned into
the same consistent JSON error shape automatically, instead of each route
handler writing its own try/except -> HTTPException translation.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.errors.exceptions import FallDetectionError


async def _fall_detection_error_handler(request: Request, exc: FallDetectionError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "detail": exc.message},
    )


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(FallDetectionError, _fall_detection_error_handler)

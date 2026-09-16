"""Optional API key auth.

If `FALLDET_API_KEY` is unset, the API stays open - matches the default
behavior every earlier part of this project already assumed (no auth is
the default for a dev/portfolio deployment, not something with real users
yet). Set `FALLDET_API_KEY` to require every request to send a matching
`X-API-Key` header.

This is header-based, so it only applies to `/predict` (a normal HTTP
request) - the browser `WebSocket` API cannot set custom headers at all,
so `/ws/predict` is intentionally left unauthenticated for now; it's
already documented elsewhere as a demo harness, not a product endpoint.
"""
from __future__ import annotations

import os

from fastapi import Header

from api.errors import UnauthorizedError


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("FALLDET_API_KEY")
    if expected is None:
        return  # auth disabled - no FALLDET_API_KEY configured
    if x_api_key != expected:
        raise UnauthorizedError("missing or incorrect X-API-Key header")

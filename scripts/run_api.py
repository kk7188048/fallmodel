#!/usr/bin/env python
"""CLI: run the fall-detection API locally for development.

Usage:
    python scripts/run_api.py
    FALLDET_MODEL_PATH=checkpoints/best_model.pt FALLDET_THRESHOLD=0.5 python scripts/run_api.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)

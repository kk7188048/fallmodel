import sys
from pathlib import Path

# Make src/falldet importable regardless of how this package is entered
# (uvicorn, scripts/run_api.py, pytest) - same trick the scripts/ CLIs use.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

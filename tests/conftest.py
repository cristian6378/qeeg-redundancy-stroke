"""Make the flat pipeline modules (config, features, ...) importable in tests."""
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1] / "pipeline"
sys.path.insert(0, str(PIPELINE))

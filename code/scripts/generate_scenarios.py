"""Generate SCEQ weather scenarios using the packaged configuration."""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT))

from sceq_lib.scenario_generator import ScenarioGenerator


if __name__ == "__main__":
    ScenarioGenerator().generate_all_paths()

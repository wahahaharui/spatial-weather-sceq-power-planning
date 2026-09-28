"""Central path helpers for the reproducible SCEQ code package.

Users normally only need to set environment variables documented in README.md:
SCEQ_DATA_ROOT, SCEQ_OUTPUT_DIR, SCEQ_DEVICE, SCEQ_N_WORKERS.
"""
from __future__ import annotations

import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SCEQ_DATA_ROOT", CODE_ROOT / "data")).resolve()
OUTPUT_DIR = Path(os.environ.get("SCEQ_OUTPUT_DIR", CODE_ROOT / "output")).resolve()
MODEL_ROOT = Path(os.environ.get("SCEQ_MODEL_ROOT", DATA_ROOT / "models")).resolve()


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).resolve()

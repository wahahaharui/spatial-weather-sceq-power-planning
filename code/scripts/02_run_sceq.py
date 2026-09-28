"""
SCEQ V2 full run for the reproducible code package.

The original model parameters and solve flow are retained. Machine-local paths are
provided by src/sceq_lib/config.py. If weather paths are not already present, this
runner generates them with ScenarioGenerator instead of copying from an old local
workspace.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT))

from sceq_lib.config import cfg_v2 as cfg
from sceq_lib.scenario_generator import ScenarioGenerator
from sceq_lib.ce_baseline import build_ce_baseline
from sceq_lib.sceq_orchestrator import SCEQOrchestrator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.FileHandler(cfg.LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def ensure_weather_paths():
    existing = sum(
        1
        for pid in range(cfg.N_PATHS)
        if (cfg.WEATHER_PATHS_DIR / f"path_{pid:03d}" / "period_2025.npz").exists()
    )
    if existing >= cfg.N_PATHS:
        logger.info("  All %s weather paths exist.", cfg.N_PATHS)
        return

    logger.info("  Found %s/%s weather paths; generating missing paths from models.", existing, cfg.N_PATHS)
    completed = {
        pid
        for pid in range(cfg.N_PATHS)
        if (cfg.WEATHER_PATHS_DIR / f"path_{pid:03d}" / "period_2025.npz").exists()
    }
    gen = ScenarioGenerator()
    gen.generate_all_paths(resume_from=completed)


def main():
    t_total = time.time()
    logger.info("=" * 70)
    logger.info("SCEQ V2 Full Run - reproducible package")
    logger.info("  gap_tol=%s, max_iter=%s", cfg.BENDERS_GAP_TOL, cfg.BENDERS_MAX_ITER)
    logger.info("  Warm-start: ON, N_WORKERS=%s", cfg.N_WORKERS)
    logger.info("  Paths: %s, Periods: %s", cfg.N_PATHS, cfg.PERIODS)
    logger.info("  Data root: %s", cfg.DATA_ROOT)
    logger.info("  Output dir: %s", cfg.OUTPUT_DIR)
    logger.info("=" * 70)

    logger.info("\n[Phase 1] Checking/generating weather paths...")
    ensure_weather_paths()
    t1 = time.time()

    logger.info("\n[Phase 2] Building CE baseline...")
    ce = build_ce_baseline()
    t2 = time.time()

    logger.info("\n[Phase 3] SCEQ rolling solve (%s workers)...", cfg.N_WORKERS)
    orch = SCEQOrchestrator()
    results = orch.run(ce)
    t3 = time.time()

    success = [r for r in results if "error" not in r and r.get("steps")]
    n_fail = len(results) - len(success)

    all_iters = []
    all_times = []
    for r in success:
        for step in r["steps"]:
            info = step.get("solve_info", {})
            all_iters.append(info.get("iterations", 0))
            all_times.append(info.get("solve_s", 0))

    summary = {
        "config": {
            "gap_tol": cfg.BENDERS_GAP_TOL,
            "max_iter": cfg.BENDERS_MAX_ITER,
            "n_workers": cfg.N_WORKERS,
            "method": "dual_simplex_v7",
            "warm_start": True,
        },
        "n_success": len(success),
        "n_failed": n_fail,
        "avg_iterations": float(np.mean(all_iters)) if all_iters else None,
        "avg_solve_time_s": float(np.mean(all_times)) if all_times else None,
        "phase1_min": (t1 - t_total) / 60,
        "phase2_s": t2 - t1,
        "phase3_h": (t3 - t2) / 3600,
        "total_wall_h": (time.time() - t_total) / 3600,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(cfg.SUMMARY_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n%s", "=" * 70)
    logger.info("RUN COMPLETE")
    logger.info("  Success: %s/%s | Failed: %s", len(success), cfg.N_PATHS, n_fail)
    logger.info("  Phase 1: %.0fmin", (t1 - t_total) / 60)
    logger.info("  Phase 2: %.0fs", t2 - t1)
    logger.info("  Phase 3: %.1fh", (t3 - t2) / 3600)
    logger.info("  Total wall time: %.1fh", (time.time() - t_total) / 3600)


if __name__ == "__main__":
    main()

"""
Certainty-equivalent baseline construction.
"""
import time
import numpy as np

from .config import cfg
from .time_aggregator import aggregate_8760_to_288
from .scenario_generator import ScenarioGenerator


def build_ce_baseline():
    print("[CE Baseline] building certainty-equivalent baseline...")
    t0 = time.time()

    ce_wind = None
    ce_solar = None
    ce_load = None

    n_valid = 0
    n_total = cfg.N_PATHS

    for path_id in range(n_total):
        try:
            path_data = ScenarioGenerator.load_path(path_id)
        except Exception as exc:
            print(f"  [CE] skip Path {path_id:03d}: {exc}")
            continue

        if n_valid == 0:
            ce_wind = np.zeros((4, 8760, cfg.N_NODES), dtype=np.float64)
            ce_solar = np.zeros((4, 8760, cfg.N_NODES), dtype=np.float64)
            ce_load = np.zeros((4, 8760), dtype=np.float64)

        for period_idx, period in enumerate(cfg.PERIODS):
            pd_data = path_data[period]
            k = n_valid
            ce_wind[period_idx] += (pd_data["wind_cf"] - ce_wind[period_idx]) / (k + 1)
            ce_solar[period_idx] += (pd_data["solar_cf"] - ce_solar[period_idx]) / (k + 1)
            ce_load[period_idx] += (pd_data["load"] - ce_load[period_idx]) / (k + 1)

        n_valid += 1
        if (path_id + 1) % 50 == 0:
            print(f"  [CE] processed {path_id + 1}/{n_total} paths...")

    if ce_wind is None:
        raise RuntimeError("no valid weather paths found for CE baseline")

    print(f"  [CE] valid paths: {n_valid}/{n_total}")

    ce_wind_288_list = []
    ce_solar_288_list = []
    ce_load_288_list = []
    for period_idx in range(4):
        ce_wind_288_list.append(aggregate_8760_to_288(ce_wind[period_idx]).astype(np.float32))
        ce_solar_288_list.append(aggregate_8760_to_288(ce_solar[period_idx]).astype(np.float32))
        ce_load_288_list.append(aggregate_8760_to_288(ce_load[period_idx]).astype(np.float32))

    ce_wind_1152 = np.concatenate(ce_wind_288_list, axis=0)
    ce_solar_1152 = np.concatenate(ce_solar_288_list, axis=0)
    ce_load_1152 = np.concatenate(ce_load_288_list, axis=0)

    np.save(cfg.CE_BASELINE_DIR / "ce_wind_cf_1152.npy", ce_wind_1152)
    np.save(cfg.CE_BASELINE_DIR / "ce_solar_cf_1152.npy", ce_solar_1152)
    np.save(cfg.CE_BASELINE_DIR / "ce_load_1152.npy", ce_load_1152)

    elapsed = time.time() - t0
    print(f"[CE Baseline] done in {elapsed:.0f}s")
    print(f"  CE Wind CF mean: {ce_wind_1152.mean():.3f}")
    print(f"  CE Solar CF mean: {ce_solar_1152.mean():.3f}")
    print(f"  CE Load mean: {ce_load_1152.mean():.0f} MW")

    return {
        "wind_cf": ce_wind_1152,
        "solar_cf": ce_solar_1152,
        "load": ce_load_1152,
    }


def load_ce_baseline():
    return {
        "wind_cf": np.load(cfg.CE_BASELINE_DIR / "ce_wind_cf_1152.npy"),
        "solar_cf": np.load(cfg.CE_BASELINE_DIR / "ce_solar_cf_1152.npy"),
        "load": np.load(cfg.CE_BASELINE_DIR / "ce_load_1152.npy"),
    }

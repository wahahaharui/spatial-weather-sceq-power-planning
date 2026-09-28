import argparse
import importlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SCEQ_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT_DEFAULT = Path(__file__).resolve().parent
if str(SCEQ_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(SCEQ_ROOT / "src"))
if str(SCEQ_ROOT) not in sys.path:
    sys.path.insert(0, str(SCEQ_ROOT))

from sceq_lib.time_aggregator import aggregate_8760_to_288
from sceq_lib.config import cfg_v2 as project_cfg


TECHS = ["wind", "solar"]
PERIODS_DEFAULT = [2025, 2030, 2035, 2040]
TOPK_LIST = [5, 10, 20, 50]
VARIANT_TO_TECH = {
    "wind_homogenized": "wind",
    "solar_homogenized": "solar",
}
STEP_OUTPUT_FILES = {
    "benders_cuts_history.csv",
    "dispatch.csv",
    "gen_build.csv",
    "gen_cap.csv",
    "summary.csv",
    "total_cost.txt",
}
REWRITTEN_DYNAMIC_FILES = {
    "gen_build_predetermined.csv",
    "loads.csv",
    "loads.parquet",
    "variable_capacity_factors.csv",
    "variable_capacity_factors.parquet",
}


@dataclass
class RunData:
    name: str
    path: Path
    df: pd.DataFrame


@dataclass
class SolverBackend:
    load_all: object
    run_benders: object
    label: str


def resolve_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_config(config_path):
    config_path = Path(config_path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    experiment_root = resolve_path(SCEQ_ROOT, payload.get("experiment_root", "mechanism_experiment"))
    run_order = list(payload["run_order"])
    baseline_runs = {
        run_name: resolve_path(SCEQ_ROOT, rel_path)
        for run_name, rel_path in payload["baseline_runs"].items()
    }
    sceq_cfg = payload.get("sceq", {})
    return {
        "config_path": config_path,
        "sceq_root": SCEQ_ROOT,
        "experiment_root": experiment_root,
        "run_order": run_order,
        "baseline_runs": baseline_runs,
        "periods": [int(x) for x in payload.get("periods", PERIODS_DEFAULT)],
        "paths_per_group": int(payload.get("paths_per_group", 20)),
        "paths_per_stratum": int(payload.get("paths_per_stratum", 5)),
        "selected_paths_file": experiment_root / "selected_paths.csv",
        "summary_dir": experiment_root / "summary",
        "sceq": {
            "n_workers": int(sceq_cfg.get("n_workers", 1)),
            "benders_max_iter": int(sceq_cfg.get("benders_max_iter", 200)),
            "benders_gap_tol": float(sceq_cfg.get("benders_gap_tol", 0.005)),
        },
    }


def ensure_experiment_dirs(config):
    dirs = [
        config["experiment_root"],
        config["experiment_root"] / "baseline",
        config["experiment_root"] / "baseline" / "results",
        config["experiment_root"] / "baseline" / "robustness",
        config["experiment_root"] / "wind_homogenized",
        config["experiment_root"] / "wind_homogenized" / "01_weather_paths",
        config["experiment_root"] / "wind_homogenized" / "results",
        config["experiment_root"] / "wind_homogenized" / "robustness",
        config["experiment_root"] / "solar_homogenized",
        config["experiment_root"] / "solar_homogenized" / "01_weather_paths",
        config["experiment_root"] / "solar_homogenized" / "results",
        config["experiment_root"] / "solar_homogenized" / "robustness",
        config["summary_dir"],
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def preflight_runtime_check(config, require_solver=False):
    issues = []
    for run_name in config["run_order"]:
        output_root = config["baseline_runs"].get(run_name)
        if output_root is None:
            issues.append(f"{run_name}: missing baseline_runs entry")
            continue
        if not output_root.exists():
            issues.append(f"{run_name}: baseline output root not found: {output_root}")
            continue
        required = [
            Path("checkpoint.json"),
            Path("01_weather_paths"),
            Path("02_ce_baseline"),
            Path("04_summary") / "02_grid_robustness_data.csv",
            Path("05_work_dirs"),
        ]
        for rel in required:
            if not (output_root / rel).exists():
                issues.append(f"{run_name}: missing required baseline artifact: {output_root / rel}")
    if require_solver:
        solver_check = ensure_solver_import_ready()
        if not solver_check["ok"]:
            issues.append(solver_check["message"])
    if issues:
        raise RuntimeError("Runtime dependency check failed.\n" + "\n".join(f"- {x}" for x in issues))


def clean_generated_outputs(config):
    root = config["experiment_root"]
    targets = [
        root / "baseline" / "robustness",
        root / "baseline" / "results" / "source_runs.json",
        root / "wind_homogenized" / "01_weather_paths",
        root / "wind_homogenized" / "results",
        root / "wind_homogenized" / "robustness",
        root / "solar_homogenized" / "01_weather_paths",
        root / "solar_homogenized" / "results",
        root / "solar_homogenized" / "robustness",
        root / "summary",
        root / "selected_paths.csv",
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        elif target.exists():
            target.unlink()
    ensure_experiment_dirs(config)


def candidate_path_ids(output_root: Path, periods):
    payload = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
    success_ids = []
    for row in payload.get("all_results", []):
        if "error" in row or not row.get("steps"):
            continue
        path_id = int(row["path_id"])
        weather_dir = output_root / "01_weather_paths" / f"path_{path_id:03d}"
        if all((weather_dir / f"period_{period}.npz").exists() for period in periods):
            success_ids.append(path_id)
    return sorted(success_ids)


def compute_resource_means(output_root: Path, path_id: int, periods):
    weather_dir = output_root / "01_weather_paths" / f"path_{path_id:03d}"
    wind_means = []
    solar_means = []
    for period in periods:
        with np.load(weather_dir / f"period_{period}.npz") as data:
            wind_means.append(float(np.mean(data["wind_cf"])))
            solar_means.append(float(np.mean(data["solar_cf"])))
    return float(np.mean(wind_means)), float(np.mean(solar_means))


def assign_stratum(wind_mean, solar_mean, wind_median, solar_median):
    wind_label = "low_wind" if wind_mean <= wind_median else "high_wind"
    solar_label = "low_solar" if solar_mean <= solar_median else "high_solar"
    return f"{wind_label}__{solar_label}"


def safe_scale(value):
    if pd.isna(value) or math.isclose(float(value), 0.0, abs_tol=1e-12):
        return 1.0
    return float(value)


def select_paths(config):
    ensure_experiment_dirs(config)
    preflight_runtime_check(config, require_solver=False)
    rows = []
    summary_rows = []
    validation_lines = [f"select-paths executed on {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    for group_id in config["run_order"]:
        output_root = config["baseline_runs"][group_id]
        path_ids = candidate_path_ids(output_root, config["periods"])
        if not path_ids:
            raise RuntimeError(f"{group_id}: no successful paths with complete weather files found.")

        resource_rows = []
        for path_id in path_ids:
            wind_mean, solar_mean = compute_resource_means(output_root, path_id, config["periods"])
            resource_rows.append(
                {
                    "group_id": group_id,
                    "path_id": path_id,
                    "wind_resource_mean": wind_mean,
                    "solar_resource_mean": solar_mean,
                }
            )
        group_df = pd.DataFrame(resource_rows)
        wind_median = float(group_df["wind_resource_mean"].median())
        solar_median = float(group_df["solar_resource_mean"].median())
        group_df["resource_stratum"] = group_df.apply(
            lambda row: assign_stratum(
                row["wind_resource_mean"],
                row["solar_resource_mean"],
                wind_median,
                solar_median,
            ),
            axis=1,
        )

        selected_frames = []
        for stratum, stratum_df in group_df.groupby("resource_stratum", sort=True):
            if len(stratum_df) < config["paths_per_stratum"]:
                raise RuntimeError(
                    f"{group_id}: stratum {stratum} only has {len(stratum_df)} paths, "
                    f"needs {config['paths_per_stratum']}."
                )
            mean_w = float(stratum_df["wind_resource_mean"].mean())
            mean_s = float(stratum_df["solar_resource_mean"].mean())
            std_w = safe_scale(stratum_df["wind_resource_mean"].std(ddof=0))
            std_s = safe_scale(stratum_df["solar_resource_mean"].std(ddof=0))
            scored = stratum_df.copy()
            scored["selection_distance"] = np.sqrt(
                ((scored["wind_resource_mean"] - mean_w) / std_w) ** 2
                + ((scored["solar_resource_mean"] - mean_s) / std_s) ** 2
            )
            scored = scored.sort_values(["selection_distance", "path_id"]).reset_index(drop=True)
            scored["selection_rank_within_stratum"] = np.arange(1, len(scored) + 1)
            selected_frames.append(scored.head(config["paths_per_stratum"]).copy())
            summary_rows.append(
                {
                    "group_id": group_id,
                    "resource_stratum": stratum,
                    "n_candidates": len(stratum_df),
                    "selected_count": config["paths_per_stratum"],
                    "wind_mean_center": mean_w,
                    "solar_mean_center": mean_s,
                    "wind_median_cut": wind_median,
                    "solar_median_cut": solar_median,
                }
            )

        selected = pd.concat(selected_frames, ignore_index=True)
        selected = selected.sort_values(
            ["resource_stratum", "selection_rank_within_stratum", "path_id"]
        ).reset_index(drop=True)
        if len(selected) != config["paths_per_group"]:
            raise RuntimeError(
                f"{group_id}: selected {len(selected)} paths, expected {config['paths_per_group']}."
            )
        rows.append(selected)
        validation_lines.append(f"{group_id}: selected {len(selected)} paths from {len(group_df)} valid paths.")

    selected_df = pd.concat(rows, ignore_index=True)
    selected_df.to_csv(config["selected_paths_file"], index=False)
    pd.DataFrame(summary_rows).sort_values(["group_id", "resource_stratum"]).to_csv(
        config["summary_dir"] / "path_selection_summary.csv",
        index=False,
    )
    (config["experiment_root"] / "baseline" / "results" / "source_runs.json").write_text(
        json.dumps(
            {
                "run_order": config["run_order"],
                "baseline_runs": {k: str(v) for k, v in config["baseline_runs"].items()},
                "selected_paths_file": str(config["selected_paths_file"]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_validation_report(config, validation_lines, mode="w")


def load_selected_paths(config):
    path = config["selected_paths_file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing selected paths file: {path}. Please run select-paths first.")
    df = pd.read_csv(path)
    df["group_id"] = df["group_id"].astype(str)
    df["path_id"] = pd.to_numeric(df["path_id"], errors="raise").astype(int)
    return df


def homogenize_cf_array(array_8760):
    hourly_mean = np.mean(array_8760, axis=1, keepdims=True)
    return np.repeat(hourly_mean, array_8760.shape[1], axis=1)


def build_counterfactual_weather(config, variant):
    if variant not in VARIANT_TO_TECH:
        raise ValueError(f"Unsupported variant: {variant}")
    ensure_experiment_dirs(config)
    preflight_runtime_check(config, require_solver=False)
    selected_paths = load_selected_paths(config)
    target_tech = VARIANT_TO_TECH[variant]
    variant_root = config["experiment_root"] / variant / "01_weather_paths"

    for group_id in config["run_order"]:
        output_root = config["baseline_runs"][group_id]
        src_weather_root = output_root / "01_weather_paths"
        dst_weather_root = variant_root / group_id
        dst_weather_root.mkdir(parents=True, exist_ok=True)
        group_df = selected_paths[selected_paths["group_id"] == group_id]
        for row in group_df.itertuples(index=False):
            path_id = int(row.path_id)
            src_path_root = src_weather_root / f"path_{path_id:03d}"
            dst_path_root = dst_weather_root / f"path_{path_id:03d}"
            dst_path_root.mkdir(parents=True, exist_ok=True)
            for period in config["periods"]:
                with np.load(src_path_root / f"period_{period}.npz") as data:
                    payload = {key: data[key] for key in data.files}
                if target_tech == "wind":
                    payload["wind_cf"] = homogenize_cf_array(payload["wind_cf"])
                else:
                    payload["solar_cf"] = homogenize_cf_array(payload["solar_cf"])
                np.savez_compressed(dst_path_root / f"period_{period}.npz", **payload)

    write_validation_report(config, [f"{variant}: homogenized weather written under {variant_root}"])


def _candidate_solver_parents():
    candidates = []
    cfg_dir = getattr(project_cfg, "BENDERS_V6_DIR", None)
    if cfg_dir is not None:
        cfg_dir = Path(cfg_dir)
        candidates.append(cfg_dir.parent)
    env_dir = None
    if "BENDERS_V6_DIR" in os.environ:
        env_dir = Path(os.environ["BENDERS_V6_DIR"])
        candidates.append(env_dir.parent)
    unique = []
    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def ensure_solver_import_ready():
    if importlib.util.find_spec("benders_fixed_v6") is not None:
        return {"ok": True, "message": "benders_fixed_v6 importable from current environment"}

    attempted = []
    for parent in _candidate_solver_parents():
        attempted.append(str(parent))
        if parent.exists() and str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        if importlib.util.find_spec("benders_fixed_v6") is not None:
            return {
                "ok": True,
                "message": f"benders_fixed_v6 importable after adding {parent}",
            }

    message = (
        "Current Python environment cannot import 'benders_fixed_v6'. "
        f"Tried direct import and configured parent paths: {attempted if attempted else '[]'}. "
        "Please check sceq_lib/config.py::BENDERS_V6_DIR or set environment variable BENDERS_V6_DIR."
    )
    return {"ok": False, "message": message}


def load_solver_backend():
    solver_check = ensure_solver_import_ready()
    if not solver_check["ok"]:
        raise RuntimeError(solver_check["message"])
    data_loader = importlib.import_module("benders_fixed_v6.data_loader")
    solver_mod = importlib.import_module("benders_fixed_v6.benders_persistent_v7")
    return SolverBackend(
        load_all=data_loader.load_all,
        run_benders=solver_mod.run_benders,
        label="benders_fixed_v6.benders_persistent_v7",
    )


def load_path_weather(weather_root: Path, path_id: int, periods):
    payload = {}
    path_root = weather_root / f"path_{path_id:03d}"
    for period in periods:
        with np.load(path_root / f"period_{period}.npz") as data:
            payload[period] = {key: data[key] for key in data.files}
    return payload


def try_write_parquet(df: pd.DataFrame, path: Path):
    try:
        df.to_parquet(path, index=False)
    except Exception:
        return False
    return True


class CounterfactualStepAdapter:
    def __init__(self, baseline_output_root: Path, result_root: Path, periods, ce_baseline_1152, solver_backend, sceq_cfg):
        self.baseline_output_root = baseline_output_root
        self.result_root = result_root
        self.periods = list(periods)
        self.ce_baseline_1152 = ce_baseline_1152
        self.solver_backend = solver_backend
        self.sceq_cfg = sceq_cfg
        self.work_root = result_root / "05_work_dirs"

        base_step_dir = self._find_first_step_dir(2025)
        self.base_predetermined = pd.read_csv(base_step_dir / "gen_build_predetermined.csv")
        gen_info = pd.read_csv(base_step_dir / "gen_info.csv")
        self.wind_projects = [g for g in gen_info["GENERATION_PROJECT"].astype(str).tolist() if g.endswith("_w")]
        self.solar_projects = [g for g in gen_info["GENERATION_PROJECT"].astype(str).tolist() if g.endswith("_s")]
        self.storage_projects = [
            g for g in gen_info["GENERATION_PROJECT"].astype(str).tolist() if "storage" in g.lower()
        ]

    def _find_first_step_dir(self, period: int):
        candidates = sorted((self.baseline_output_root / "05_work_dirs").glob(f"path_*\\step_{period}"))
        if not candidates:
            raise FileNotFoundError(
                f"No baseline template step_{period} found under {self.baseline_output_root / '05_work_dirs'}"
            )
        return candidates[0]

    def run_one_step(
        self,
        current_period,
        remaining_periods,
        path_weather_8760,
        previous_decisions,
        path_id,
        warm_start_sol=None,
    ):
        step_work_dir = self.work_root / f"path_{path_id:03d}" / f"step_{current_period}"
        if step_work_dir.exists():
            shutil.rmtree(step_work_dir)
        step_work_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        self._write_static_inputs(path_id, current_period, remaining_periods, step_work_dir)
        self._write_predetermined(step_work_dir, previous_decisions)
        self._write_variable_capacity_factors(step_work_dir, current_period, remaining_periods, path_weather_8760)
        self._write_loads(step_work_dir, current_period, remaining_periods, path_weather_8760)
        prepare_t = time.time() - t0

        data = self.solver_backend.load_all(str(step_work_dir))
        result = self.solver_backend.run_benders(
            data,
            max_iter=self.sceq_cfg["benders_max_iter"],
            gap_tol=self.sceq_cfg["benders_gap_tol"],
            output_dir=str(step_work_dir),
            warm_start_sol=warm_start_sol,
        )
        solve_t = time.time() - t0 - prepare_t

        decisions = self._extract_build_decisions(step_work_dir, current_period)
        master_sol = None
        if result.get("best_sol"):
            master_sol = result["best_sol"].get("master")
        ub = result.get("UB", float("nan"))
        lb = result.get("LB", float("nan"))
        gap = (ub - lb) / (abs(ub) + 1e-10) if np.isfinite(ub) and np.isfinite(lb) else 1.0
        solve_info = {
            "current_period": int(current_period),
            "prepare_s": round(prepare_t, 1),
            "solve_s": round(solve_t, 1),
            "total_s": round(time.time() - t0, 1),
            "ub": ub,
            "lb": lb,
            "iterations": int(result.get("iterations", 0)),
            "total_cuts": int(result.get("total_cuts", 0)),
            "gap": gap,
            "solver_backend": self.solver_backend.label,
        }
        return decisions, ub, solve_info, master_sol

    def _write_static_inputs(self, path_id, current_period, remaining_periods, step_work_dir: Path):
        src_step_dir = self.baseline_output_root / "05_work_dirs" / f"path_{path_id:03d}" / f"step_{current_period}"
        if not src_step_dir.exists():
            raise FileNotFoundError(f"Missing baseline step template: {src_step_dir}")
        for file in src_step_dir.iterdir():
            if file.name in STEP_OUTPUT_FILES or file.name in REWRITTEN_DYNAMIC_FILES:
                continue
            if file.suffix.lower() != ".csv":
                continue
            shutil.copy2(file, step_work_dir / file.name)
        build_costs = pd.read_csv(src_step_dir / "gen_build_costs.csv")
        build_costs = build_costs[build_costs["build_year"].isin(remaining_periods)].copy()
        build_costs.to_csv(step_work_dir / "gen_build_costs.csv", index=False)

    def _write_predetermined(self, step_work_dir: Path, previous_decisions):
        predetermined = self.base_predetermined.copy()
        extra_rows = []
        for period, decisions in previous_decisions.items():
            for generation_project, mw in decisions.items():
                if float(mw) > 1e-6:
                    extra_rows.append(
                        {
                            "GENERATION_PROJECT": generation_project,
                            "build_year": int(period),
                            "build_gen_predetermined": float(mw),
                        }
                    )
        if extra_rows:
            predetermined = pd.concat([predetermined, pd.DataFrame(extra_rows)], ignore_index=True)
        predetermined.to_csv(step_work_dir / "gen_build_predetermined.csv", index=False)

    def _select_288_for_period(self, current_period, remaining_periods, path_weather_8760, period, key):
        offset = self.periods.index(period) * 288
        if period == current_period or period not in remaining_periods:
            return aggregate_8760_to_288(path_weather_8760[period][key])
        return self.ce_baseline_1152[key][offset:offset + 288]

    def _write_variable_capacity_factors(self, step_work_dir, current_period, remaining_periods, path_weather_8760):
        rows = []
        for period in self.periods:
            offset = self.periods.index(period) * 288
            wind_288 = self._select_288_for_period(
                current_period, remaining_periods, path_weather_8760, period, "wind_cf"
            )
            solar_288 = self._select_288_for_period(
                current_period, remaining_periods, path_weather_8760, period, "solar_cf"
            )
            for tp_local in range(288):
                timepoint = offset + tp_local + 1
                for grid_idx, proj_name in enumerate(self.wind_projects):
                    rows.append(
                        {
                            "GENERATION_PROJECT": proj_name,
                            "timepoint": timepoint,
                            "gen_max_capacity_factor": round(float(wind_288[tp_local, grid_idx]), 6),
                        }
                    )
                for grid_idx, proj_name in enumerate(self.solar_projects):
                    rows.append(
                        {
                            "GENERATION_PROJECT": proj_name,
                            "timepoint": timepoint,
                            "gen_max_capacity_factor": round(float(solar_288[tp_local, grid_idx]), 6),
                        }
                    )
        cf_df = pd.DataFrame(rows)
        cf_df.to_csv(step_work_dir / "variable_capacity_factors.csv", index=False)
        try_write_parquet(cf_df, step_work_dir / "variable_capacity_factors.parquet")

    def _write_loads(self, step_work_dir, current_period, remaining_periods, path_weather_8760):
        rows = []
        for period in self.periods:
            offset = self.periods.index(period) * 288
            load_288 = self._select_288_for_period(
                current_period, remaining_periods, path_weather_8760, period, "load"
            )
            for tp_local in range(288):
                rows.append(
                    {
                        "LOAD_ZONE": "inside_lvliang",
                        "TIMEPOINT": offset + tp_local + 1,
                        "zone_demand_mw": round(float(load_288[tp_local]), 4),
                    }
                )
        load_df = pd.DataFrame(rows)
        load_df.to_csv(step_work_dir / "loads.csv", index=False)
        try_write_parquet(load_df, step_work_dir / "loads.parquet")

    def _extract_build_decisions(self, step_work_dir, current_period):
        build_file = step_work_dir / "gen_build.csv"
        if not build_file.exists():
            return {}
        df = pd.read_csv(build_file)
        decisions = {}
        for row in df.itertuples(index=False):
            try:
                row_period = int(getattr(row, "PERIOD"))
                build_gen = getattr(row, "BuildGen")
            except Exception:
                continue
            if row_period != int(current_period):
                continue
            if build_gen == "." or pd.isna(build_gen):
                continue
            build_val = float(build_gen)
            if build_val > 1e-6:
                decisions[getattr(row, "GENERATION_PROJECT")] = build_val
        return decisions


class CounterfactualGroupRunner:
    def __init__(self, config, group_id, variant, solver_backend):
        self.config = config
        self.group_id = group_id
        self.variant = variant
        self.solver_backend = solver_backend
        self.baseline_output_root = config["baseline_runs"][group_id]
        self.weather_root = config["experiment_root"] / variant / "01_weather_paths" / group_id
        self.result_root = config["experiment_root"] / variant / "results" / group_id
        self.result_root.mkdir(parents=True, exist_ok=True)
        for subdir in ["02_ce_baseline", "03_path_results", "04_summary", "05_work_dirs"]:
            (self.result_root / subdir).mkdir(parents=True, exist_ok=True)

        self.ce_baseline_1152 = {
            "wind_cf": np.load(self.baseline_output_root / "02_ce_baseline" / "ce_wind_cf_1152.npy"),
            "solar_cf": np.load(self.baseline_output_root / "02_ce_baseline" / "ce_solar_cf_1152.npy"),
            "load": np.load(self.baseline_output_root / "02_ce_baseline" / "ce_load_1152.npy"),
        }
        shutil.copy2(
            self.baseline_output_root / "04_summary" / "02_grid_robustness_data.csv",
            self.result_root / "04_summary" / "02_grid_robustness_data.csv",
        )
        np.save(self.result_root / "02_ce_baseline" / "ce_wind_cf_1152.npy", self.ce_baseline_1152["wind_cf"])
        np.save(self.result_root / "02_ce_baseline" / "ce_solar_cf_1152.npy", self.ce_baseline_1152["solar_cf"])
        np.save(self.result_root / "02_ce_baseline" / "ce_load_1152.npy", self.ce_baseline_1152["load"])
        self.adapter = CounterfactualStepAdapter(
            baseline_output_root=self.baseline_output_root,
            result_root=self.result_root,
            periods=config["periods"],
            ce_baseline_1152=self.ce_baseline_1152,
            solver_backend=solver_backend,
            sceq_cfg=config["sceq"],
        )

    def _load_checkpoint(self):
        path = self.result_root / "checkpoint.json"
        if not path.exists():
            return {"completed_paths": [], "all_results": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"completed_paths": [], "all_results": []}

    def _save_checkpoint(self, completed, all_results, t_start):
        payload = {
            "completed_paths": sorted(int(x) for x in completed),
            "all_results": all_results,
            "elapsed_s": time.time() - t_start,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "group_id": self.group_id,
            "variant": self.variant,
        }
        (self.result_root / "checkpoint.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def run_one_path(self, path_id):
        path_weather = load_path_weather(self.weather_root, path_id, self.config["periods"])
        previous_decisions = {}
        warm_start_sol = None
        steps = []
        cumul_wind = 0.0
        cumul_solar = 0.0
        cumul_storage = 0.0

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            for step_idx, period in enumerate(self.config["periods"]):
                remaining = self.config["periods"][step_idx:]
                decisions, total_cost, solve_info, master_sol = self.adapter.run_one_step(
                    current_period=period,
                    remaining_periods=remaining,
                    path_weather_8760=path_weather,
                    previous_decisions=previous_decisions,
                    path_id=path_id,
                    warm_start_sol=warm_start_sol,
                )
                warm_start_sol = master_sol
                previous_decisions[period] = decisions
                cumul_wind += sum(v for k, v in decisions.items() if k in self.adapter.wind_projects)
                cumul_solar += sum(v for k, v in decisions.items() if k in self.adapter.solar_projects)
                cumul_storage += sum(v for k, v in decisions.items() if k in self.adapter.storage_projects)
                steps.append(
                    {
                        "period": int(period),
                        "wind_build_mw": round(cumul_wind, 3),
                        "solar_build_mw": round(cumul_solar, 3),
                        "storage_build_mw": round(cumul_storage, 3),
                        "total_cost": float(total_cost) if np.isfinite(total_cost) else None,
                        "solve_info": solve_info,
                    }
                )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return {"path_id": int(path_id), "steps": steps}

    def run(self, selected_path_ids):
        checkpoint = self._load_checkpoint()
        completed = set(int(x) for x in checkpoint.get("completed_paths", []))
        existing_results = {
            int(row["path_id"]): row
            for row in checkpoint.get("all_results", [])
            if "path_id" in row
        }
        pending = [pid for pid in selected_path_ids if pid not in completed]
        if not pending:
            return [existing_results[pid] for pid in sorted(existing_results)]

        t_start = time.time()
        max_workers = min(max(1, self.config["sceq"]["n_workers"]), len(pending))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_one_path, pid): pid for pid in pending}
            for future in as_completed(futures):
                path_id = futures[future]
                try:
                    existing_results[path_id] = future.result()
                except Exception as exc:
                    existing_results[path_id] = {
                        "path_id": int(path_id),
                        "error": f"{type(exc).__name__}: {exc}",
                        "steps": [],
                    }
                    traceback.print_exc()
                completed.add(path_id)
                self._save_checkpoint(completed, [existing_results[k] for k in sorted(existing_results)], t_start)

        all_results = [existing_results[k] for k in sorted(existing_results)]
        self._save_checkpoint(completed, all_results, t_start)
        self.write_path_results(all_results)
        self.write_group_summary(all_results)
        return all_results

    def write_path_results(self, all_results):
        rows = []
        for result in all_results:
            if "error" in result or not result.get("steps"):
                rows.append({"path_id": result["path_id"], "group_id": self.group_id, "error": result.get("error", "")})
                continue
            final_step = sorted(result["steps"], key=lambda x: int(x["period"]))[-1]
            rows.append(
                {
                    "path_id": result["path_id"],
                    "group_id": self.group_id,
                    "wind_build_mw_2040": final_step.get("wind_build_mw", 0.0),
                    "solar_build_mw_2040": final_step.get("solar_build_mw", 0.0),
                    "storage_build_mw_2040": final_step.get("storage_build_mw", 0.0),
                    "total_cost_2040": final_step.get("total_cost"),
                    "final_gap": final_step.get("solve_info", {}).get("gap"),
                }
            )
        pd.DataFrame(rows).to_csv(self.result_root / "03_path_results" / "path_results_summary.csv", index=False)

    def write_group_summary(self, all_results):
        rows = []
        for period in self.config["periods"]:
            wind_vals = []
            solar_vals = []
            storage_vals = []
            gaps = []
            for result in all_results:
                if "error" in result or not result.get("steps"):
                    continue
                step = next((x for x in result["steps"] if int(x["period"]) == int(period)), None)
                if step is None:
                    continue
                wind_vals.append(float(step.get("wind_build_mw", 0.0) or 0.0))
                solar_vals.append(float(step.get("solar_build_mw", 0.0) or 0.0))
                storage_vals.append(float(step.get("storage_build_mw", 0.0) or 0.0))
                gaps.append(float(step.get("solve_info", {}).get("gap", np.nan)))
            rows.append(
                {
                    "group_id": self.group_id,
                    "period": int(period),
                    "n_success_paths": len(wind_vals),
                    "wind_mean_mw": float(np.mean(wind_vals)) if wind_vals else np.nan,
                    "solar_mean_mw": float(np.mean(solar_vals)) if solar_vals else np.nan,
                    "storage_mean_mw": float(np.mean(storage_vals)) if storage_vals else np.nan,
                    "mean_gap": float(np.nanmean(gaps)) if gaps else np.nan,
                }
            )
        pd.DataFrame(rows).to_csv(self.result_root / "04_summary" / "group_summary.csv", index=False)


def run_sceq_for_variant(config, variant, groups=None, dry_run=False):
    if variant not in VARIANT_TO_TECH:
        raise ValueError(f"Unsupported variant: {variant}")
    ensure_experiment_dirs(config)
    preflight_runtime_check(config, require_solver=not dry_run)
    selected_paths = load_selected_paths(config)
    target_groups = config["run_order"] if not groups else [g for g in config["run_order"] if g in groups]
    if not target_groups:
        raise RuntimeError("No matching groups to run.")

    if dry_run:
        lines = [f"{variant}: dry-run only, solver not launched."]
        for group_id in target_groups:
            weather_root = config["experiment_root"] / variant / "01_weather_paths" / group_id
            group_paths = selected_paths[selected_paths["group_id"] == group_id]["path_id"].astype(int).tolist()
            missing = []
            for path_id in group_paths:
                for period in config["periods"]:
                    file_path = weather_root / f"path_{path_id:03d}" / f"period_{period}.npz"
                    if not file_path.exists():
                        missing.append(str(file_path))
            if missing:
                raise FileNotFoundError(f"{group_id}: missing homogenized weather files.\n- " + "\n- ".join(missing[:10]))
            lines.append(f"{variant} {group_id}: weather ready for {len(group_paths)} paths.")
        write_validation_report(config, lines)
        return

    solver_backend = load_solver_backend()
    lines = [f"{variant}: using solver backend {solver_backend.label}"]
    for group_id in target_groups:
        group_paths = selected_paths[selected_paths["group_id"] == group_id]["path_id"].astype(int).tolist()
        runner = CounterfactualGroupRunner(config, group_id, variant, solver_backend)
        runner.run(group_paths)
        lines.append(f"{variant} {group_id}: finished {len(group_paths)} selected paths.")
    write_validation_report(config, lines)


def _find_col(df, candidates, required=True):
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def _safe_cv(std_val, mean_val):
    if pd.isna(mean_val) or math.isclose(float(mean_val), 0.0, abs_tol=1e-12):
        return np.nan
    return float(std_val) / float(mean_val)


def normalize_series(series):
    series = pd.Series(series, dtype=float)
    if series.empty:
        return series
    max_val = series.max()
    min_val = series.min()
    if pd.isna(max_val) or pd.isna(min_val) or math.isclose(max_val, min_val):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def _priority_class_from_metrics(top20_consensus, class_consistency, mean_frequency):
    if math.isclose(top20_consensus, 0.0, abs_tol=1e-12) and mean_frequency < 0.10:
        return "N"
    if top20_consensus >= 0.75 and class_consistency >= 0.75:
        return "R1"
    if top20_consensus >= 0.50 and class_consistency >= 0.50:
        return "R2"
    return "S"


def build_priority_summary_from_output_root(output_root: Path, run_name: str, periods, selected_path_ids=None):
    checkpoint_file = output_root / "checkpoint.json"
    work_dirs = output_root / "05_work_dirs"
    robustness_file = output_root / "04_summary" / "02_grid_robustness_data.csv"
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"{run_name}: missing checkpoint file: {checkpoint_file}")
    if not work_dirs.exists():
        raise FileNotFoundError(f"{run_name}: missing work dirs: {work_dirs}")
    if not robustness_file.exists():
        raise FileNotFoundError(f"{run_name}: missing robustness file: {robustness_file}")

    checkpoint = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    selected_set = None if selected_path_ids is None else {int(x) for x in selected_path_ids}
    success_paths = []
    for row in checkpoint.get("all_results", []):
        if "error" in row or not row.get("steps"):
            continue
        path_id = int(row["path_id"])
        if selected_set is not None and path_id not in selected_set:
            continue
        success_paths.append(row)
    if not success_paths:
        raise RuntimeError(f"{run_name}: no successful selected paths found in checkpoint")

    robustness = pd.read_csv(robustness_file)
    robustness.columns = [str(c).strip() for c in robustness.columns]
    robustness["grid_id"] = robustness["grid_id"].astype(str)
    if "lon" not in robustness.columns:
        robustness = robustness.rename(columns={_find_col(robustness, ["longitude"]): "lon"})
    if "lat" not in robustness.columns:
        robustness = robustness.rename(columns={_find_col(robustness, ["latitude"]): "lat"})
    grid_base = robustness[["grid_id", "lon", "lat"]].drop_duplicates("grid_id").copy()
    grid_ids = sorted(grid_base["grid_id"].unique())
    path_ids = sorted(int(row["path_id"]) for row in success_paths)

    new_build = {
        tech: {period: pd.DataFrame(0.0, index=grid_ids, columns=path_ids) for period in periods}
        for tech in TECHS
    }
    final_build = {
        tech: pd.DataFrame(0.0, index=grid_ids, columns=path_ids)
        for tech in TECHS
    }

    for result in success_paths:
        path_id = int(result["path_id"])
        cumul = {tech: {} for tech in TECHS}
        for step in sorted(result["steps"], key=lambda x: int(x["period"])):
            period = int(step["period"])
            build_file = work_dirs / f"path_{path_id:03d}" / f"step_{period}" / "gen_build.csv"
            if not build_file.exists():
                continue
            build_df = pd.read_csv(build_file)
            build_df = build_df[build_df["GENERATION_PROJECT"].astype(str).str.match(r"LL_\d+_[ws]$")].copy()
            build_df["PERIOD"] = pd.to_numeric(build_df["PERIOD"], errors="coerce")
            build_df["BuildGen"] = pd.to_numeric(build_df["BuildGen"], errors="coerce").fillna(0.0)
            build_df = build_df[build_df["PERIOD"] == period]
            build_df = build_df[build_df["BuildGen"] > 1e-6]
            if build_df.empty:
                continue
            build_df["grid_id"] = build_df["GENERATION_PROJECT"].str.extract(r"(LL_\d+)")
            build_df["tech"] = np.where(build_df["GENERATION_PROJECT"].str.endswith("_w"), "wind", "solar")
            grouped = build_df.groupby(["grid_id", "tech"], as_index=False)["BuildGen"].sum()
            for row in grouped.itertuples(index=False):
                new_build[row.tech][period].loc[row.grid_id, path_id] = float(row.BuildGen)
                cumul[row.tech][row.grid_id] = cumul[row.tech].get(row.grid_id, 0.0) + float(row.BuildGen)
                final_build[row.tech].loc[row.grid_id, path_id] = cumul[row.tech][row.grid_id]
        for tech in TECHS:
            for grid_id, cap in cumul[tech].items():
                final_build[tech].loc[grid_id, path_id] = cap

    template = next(work_dirs.glob("path_*\\step_2040\\gen_info.csv"), None)
    if template is None:
        raise FileNotFoundError(f"{run_name}: no step_2040/gen_info.csv found under {work_dirs}")
    gen_info = pd.read_csv(template)
    gen_info = gen_info[gen_info["GENERATION_PROJECT"].astype(str).str.match(r"LL_\d+_[ws]$")].copy()
    gen_info["gen_capacity_limit_mw"] = pd.to_numeric(gen_info["gen_capacity_limit_mw"], errors="coerce").fillna(0.0)
    gen_info["grid_id"] = gen_info["GENERATION_PROJECT"].str.extract(r"(LL_\d+)")
    gen_info["tech"] = np.where(gen_info["GENERATION_PROJECT"].str.endswith("_w"), "wind", "solar")
    limits = (
        gen_info.groupby(["grid_id", "tech"], as_index=False)["gen_capacity_limit_mw"]
        .sum()
        .rename(columns={"gen_capacity_limit_mw": "capacity_limit_mw"})
    )

    records = []
    for grid_id in grid_ids:
        base_row = grid_base[grid_base["grid_id"] == grid_id].iloc[0]
        row = {
            "grid_id": grid_id,
            "longitude": float(base_row["lon"]),
            "latitude": float(base_row["lat"]),
        }
        for tech in TECHS:
            final_arr = final_build[tech].loc[grid_id].to_numpy(dtype=float)
            positive_final = final_arr[final_arr > 1e-6]
            row[f"{tech}_freq_final"] = float(np.mean(final_arr > 1e-6))
            row[f"{tech}_recommended_mw"] = float(np.median(positive_final)) if len(positive_final) else 0.0
            row[f"{tech}_p25_mw"] = float(np.percentile(positive_final, 25)) if len(positive_final) else 0.0
            row[f"{tech}_p75_mw"] = float(np.percentile(positive_final, 75)) if len(positive_final) else 0.0
            row[f"{tech}_all_path_mean_mw"] = float(np.mean(final_arr))
            row[f"{tech}_all_path_std_mw"] = float(np.std(final_arr))

            first_period_counts = {}
            for period in periods:
                arr = new_build[tech][period].loc[grid_id].to_numpy(dtype=float)
                first_period_counts[period] = float(np.mean(arr > 1e-6))
                row[f"{tech}_new_freq_{period}"] = first_period_counts[period]
                row[f"{tech}_new_mean_{period}"] = float(np.mean(arr))

            first_build_periods = []
            for path_id in path_ids:
                first_period = None
                for period in periods:
                    if new_build[tech][period].loc[grid_id, path_id] > 1e-6:
                        first_period = period
                        break
                first_build_periods.append(first_period)
            developed_firsts = [x for x in first_build_periods if x is not None]
            row[f"{tech}_first_build_period_mode"] = (
                int(pd.Series(developed_firsts).mode().iloc[0]) if developed_firsts else np.nan
            )

            weights = {2025: 1.0, 2030: 0.75, 2035: 0.5, 2040: 0.25}
            earliness_score = 0.0
            for period, weight in weights.items():
                earliness_score += first_period_counts.get(period, 0.0) * weight
            row[f"{tech}_earliness_score"] = earliness_score
        records.append(row)

    stats = pd.DataFrame(records)
    limits_pivot = limits.pivot(index="grid_id", columns="tech", values="capacity_limit_mw").reset_index()
    stats = stats.merge(limits_pivot, on="grid_id", how="left")
    stats = stats.rename(columns={"wind": "wind_capacity_limit_mw", "solar": "solar_capacity_limit_mw"})
    for tech in TECHS:
        limit_col = f"{tech}_capacity_limit_mw"
        limit_series = pd.to_numeric(stats.get(limit_col, 0.0), errors="coerce").fillna(0.0)
        mean_cap = pd.to_numeric(stats[f"{tech}_all_path_mean_mw"], errors="coerce").fillna(0.0)
        freq = pd.to_numeric(stats[f"{tech}_freq_final"], errors="coerce").fillna(0.0)
        earliness = pd.to_numeric(stats[f"{tech}_earliness_score"], errors="coerce").fillna(0.0)
        intensity = mean_cap / limit_series.replace(0.0, np.nan)
        intensity = intensity.fillna(0.0).clip(0.0, 1.0)
        priority_score = 0.45 * normalize_series(freq) + 0.40 * normalize_series(intensity) + 0.15 * normalize_series(earliness)
        stats[f"{tech}_priority_score"] = priority_score
        stats[f"{tech}_priority_class"] = np.select(
            [
                (freq >= 0.75) & (intensity >= 0.40),
                (freq >= 0.50),
                (freq >= 0.20),
                (freq > 0.0),
            ],
            ["A", "B", "C", "D"],
            default="None",
        )
    return stats


def load_run_data(run_name, path, periods, selected_path_ids=None):
    df = build_priority_summary_from_output_root(path, run_name, periods, selected_path_ids=selected_path_ids)
    df.columns = [str(c).strip() for c in df.columns]
    df["grid_id"] = df["grid_id"].astype(str)
    df = df.drop_duplicates("grid_id").reset_index(drop=True)
    return RunData(name=run_name, path=path, df=df)


def validate_runs(run_data_list):
    issues = []
    if not run_data_list:
        return ["No runs loaded."]
    base_ids = set(run_data_list[0].df["grid_id"])
    for run in run_data_list[1:]:
        if set(run.df["grid_id"]) != base_ids:
            issues.append(f"{run.name}: grid set differs from {run_data_list[0].name}")
    return issues


def build_run_lookup(run_data_list, tech):
    lookup = {}
    for run in run_data_list:
        cols = [
            "grid_id",
            "longitude",
            "latitude",
            f"{tech}_priority_score",
            f"{tech}_priority_class",
            f"{tech}_freq_final",
            f"{tech}_recommended_mw",
            f"{tech}_first_build_period_mode",
            f"{tech}_earliness_score",
            f"{tech}_p25_mw",
            f"{tech}_p75_mw",
        ]
        frame = run.df[cols].copy().rename(
            columns={
                f"{tech}_priority_score": "priority_score",
                f"{tech}_priority_class": "priority_class",
                f"{tech}_freq_final": "freq_final",
                f"{tech}_recommended_mw": "recommended_mw",
                f"{tech}_first_build_period_mode": "first_build_period_mode",
                f"{tech}_earliness_score": "earliness_score",
                f"{tech}_p25_mw": "p25_mw",
                f"{tech}_p75_mw": "p75_mw",
            }
        )
        lookup[run.name] = frame.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return lookup


def compute_topk_overlap(run_lookup, tech, run_order):
    rows = []
    summary_rows = []
    for k in TOPK_LIST:
        overlaps = []
        for left, right in combinations(run_order, 2):
            left_ids = set(run_lookup[left].head(k)["grid_id"].tolist())
            right_ids = set(run_lookup[right].head(k)["grid_id"].tolist())
            overlap = len(left_ids & right_ids) / float(k)
            overlaps.append(overlap)
            rows.append(
                {
                    "technology": tech,
                    "k": k,
                    "run_left": left,
                    "run_right": right,
                    "pairwise_overlap": overlap,
                }
            )
        summary_rows.append(
            {
                "technology": tech,
                "k": k,
                "mean_pairwise_overlap": float(np.mean(overlaps)) if overlaps else np.nan,
                "min_pairwise_overlap": float(np.min(overlaps)) if overlaps else np.nan,
                "max_pairwise_overlap": float(np.max(overlaps)) if overlaps else np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def compute_rank_correlation(run_lookup, tech, run_order):
    rows = []
    vals = []
    merged = None
    for run_name in run_order:
        frame = run_lookup[run_name][["grid_id", "priority_score"]].rename(columns={"priority_score": run_name})
        merged = frame if merged is None else merged.merge(frame, on="grid_id", how="inner")
    for left, right in combinations(run_order, 2):
        corr, _ = spearmanr(merged[left], merged[right])
        corr = float(corr) if pd.notna(corr) else np.nan
        vals.append(corr)
        rows.append({"technology": tech, "run_left": left, "run_right": right, "spearman": corr})
    summary = pd.DataFrame(
        [
            {
                "technology": tech,
                "mean_spearman": float(np.nanmean(vals)) if vals else np.nan,
                "min_spearman": float(np.nanmin(vals)) if vals else np.nan,
                "max_spearman": float(np.nanmax(vals)) if vals else np.nan,
            }
        ]
    )
    return pd.DataFrame(rows), summary


def build_merged_grid_frame(run_lookup, run_order):
    merged = None
    for run_name in run_order:
        frame = run_lookup[run_name].rename(
            columns={
                "priority_score": f"{run_name}__priority_score",
                "priority_class": f"{run_name}__priority_class",
                "freq_final": f"{run_name}__freq_final",
                "recommended_mw": f"{run_name}__recommended_mw",
                "first_build_period_mode": f"{run_name}__first_build_period_mode",
                "earliness_score": f"{run_name}__earliness_score",
                "p25_mw": f"{run_name}__p25_mw",
                "p75_mw": f"{run_name}__p75_mw",
            }
        )
        merged = frame if merged is None else merged.merge(frame, on=["grid_id", "longitude", "latitude"], how="inner")
    return merged


def compute_grid_stability_tables(merged, tech, run_order):
    class_rows = []
    freq_rows = []
    cap_rows = []
    score_rows = []
    consensus_rows = []
    robust_rows = []
    top20_sets = {
        run_name: set(merged.sort_values(f"{run_name}__priority_score", ascending=False).head(20)["grid_id"].tolist())
        for run_name in run_order
    }

    for row in merged.itertuples(index=False):
        grid_id = row.grid_id
        longitude = float(row.longitude)
        latitude = float(row.latitude)
        classes = [str(getattr(row, f"{run_name}__priority_class")) for run_name in run_order]
        class_counts = pd.Series(classes).value_counts()
        dominant_class = str(class_counts.index[0]) if not class_counts.empty else "None"
        class_consistency = float(class_counts.iloc[0] / len(run_order)) if not class_counts.empty else 0.0
        freqs = [float(getattr(row, f"{run_name}__freq_final")) for run_name in run_order]
        caps = [float(getattr(row, f"{run_name}__recommended_mw")) for run_name in run_order]
        scores = [float(getattr(row, f"{run_name}__priority_score")) for run_name in run_order]
        top20_hits = sum(1 for run_name in run_order if grid_id in top20_sets[run_name])
        top20_consensus = top20_hits / float(len(run_order))

        class_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": longitude,
                "latitude": latitude,
                "dominant_class": dominant_class,
                "class_consistency": class_consistency,
            }
        )
        freq_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": longitude,
                "latitude": latitude,
                "mean_frequency": float(np.mean(freqs)),
                "std_frequency": float(np.std(freqs)),
            }
        )
        cap_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": longitude,
                "latitude": latitude,
                "mean_capacity": float(np.mean(caps)),
                "std_capacity": float(np.std(caps)),
                "capacity_cv": _safe_cv(np.std(caps), np.mean(caps)),
            }
        )
        score_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": longitude,
                "latitude": latitude,
                "mean_priority_score": float(np.mean(scores)),
                "std_priority_score": float(np.std(scores)),
            }
        )
        consensus_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": longitude,
                "latitude": latitude,
                "top20_occurrence_count": int(top20_hits),
                "top20_consensus": float(top20_consensus),
            }
        )
        robust_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": longitude,
                "latitude": latitude,
                "mean_priority_score": float(np.mean(scores)),
                "std_priority_score": float(np.std(scores)),
                "mean_frequency": float(np.mean(freqs)),
                "std_frequency": float(np.std(freqs)),
                "mean_capacity": float(np.mean(caps)),
                "std_capacity": float(np.std(caps)),
                "top20_occurrence_count": int(top20_hits),
                "top20_consensus": float(top20_consensus),
                "class_consistency": float(class_consistency),
                "dominant_class": dominant_class,
                "robust_priority_class": _priority_class_from_metrics(
                    top20_consensus, class_consistency, float(np.mean(freqs))
                ),
            }
        )

    return (
        pd.DataFrame(class_rows),
        pd.DataFrame(freq_rows),
        pd.DataFrame(cap_rows),
        pd.DataFrame(score_rows),
        pd.DataFrame(consensus_rows),
        pd.DataFrame(robust_rows),
    )


def build_priority_lists(robust_df, cap_df, freq_df, score_df):
    merged = (
        robust_df.merge(
            cap_df,
            on=["technology", "grid_id", "longitude", "latitude", "mean_capacity", "std_capacity"],
        ).merge(
            freq_df,
            on=["technology", "grid_id", "longitude", "latitude", "mean_frequency", "std_frequency"],
        ).merge(
            score_df,
            on=["technology", "grid_id", "longitude", "latitude", "mean_priority_score", "std_priority_score"],
        )
    )
    robust = merged[merged["robust_priority_class"].isin(["R1", "R2"])].copy()
    robust = robust.sort_values(
        ["top20_consensus", "mean_priority_score", "mean_frequency"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    robust["rank"] = np.arange(1, len(robust) + 1)
    sensitive = merged[merged["robust_priority_class"] == "S"].copy()
    sensitive = sensitive.sort_values("mean_priority_score", ascending=False).reset_index(drop=True)
    return robust.head(20).copy(), sensitive


def make_robustness_summary(overlap_summary, corr_summary, class_df, freq_df, cap_df, robust_df):
    rows = []
    for tech in TECHS:
        tech_overlap = overlap_summary[overlap_summary["technology"] == tech]
        tech_corr = corr_summary[corr_summary["technology"] == tech]
        tech_class = class_df[class_df["technology"] == tech]
        tech_freq = freq_df[freq_df["technology"] == tech]
        tech_cap = cap_df[cap_df["technology"] == tech]
        tech_robust = robust_df[robust_df["technology"] == tech]

        def overlap_mean(k):
            sub = tech_overlap[tech_overlap["k"] == k]
            return float(sub["mean_pairwise_overlap"].iloc[0]) if not sub.empty else np.nan

        rows.append(
            {
                "technology": tech,
                "n_total_grids": int(tech_class["grid_id"].nunique()),
                "n_R1": int((tech_robust["robust_priority_class"] == "R1").sum()),
                "n_R2": int((tech_robust["robust_priority_class"] == "R2").sum()),
                "n_S": int((tech_robust["robust_priority_class"] == "S").sum()),
                "n_N": int((tech_robust["robust_priority_class"] == "N").sum()),
                "top10_mean_overlap": overlap_mean(10),
                "top20_mean_overlap": overlap_mean(20),
                "top50_mean_overlap": overlap_mean(50),
                "mean_spearman": float(tech_corr["mean_spearman"].iloc[0]),
                "min_spearman": float(tech_corr["min_spearman"].iloc[0]),
                "mean_class_consistency": float(tech_class["class_consistency"].mean()),
                "mean_frequency_std": float(tech_freq["std_frequency"].mean()),
                "mean_capacity_cv": float(tech_cap["capacity_cv"].dropna().mean()),
            }
        )
    return pd.DataFrame(rows)


def compute_total_capacity_stats(run_sources, tech, selected_paths):
    values = []
    key = "wind_build_mw" if tech == "wind" else "solar_build_mw"
    for run_name, output_root in run_sources.items():
        selected_set = set(selected_paths[selected_paths["group_id"] == run_name]["path_id"].astype(int).tolist())
        payload = json.loads((output_root / "checkpoint.json").read_text(encoding="utf-8"))
        for row in payload.get("all_results", []):
            if "error" in row or not row.get("steps"):
                continue
            if int(row["path_id"]) not in selected_set:
                continue
            final_step = sorted(row["steps"], key=lambda item: int(item["period"]))[-1]
            values.append(float(final_step.get(key, 0.0) or 0.0))
    return float(np.mean(values)) if values else np.nan, float(np.std(values)) if values else np.nan


def analyze_experiment(config, experiment):
    ensure_experiment_dirs(config)
    preflight_runtime_check(config, require_solver=False)
    selected_paths = load_selected_paths(config)
    if experiment == "baseline":
        run_sources = config["baseline_runs"]
    else:
        run_sources = {
            run_name: config["experiment_root"] / experiment / "results" / run_name
            for run_name in config["run_order"]
        }

    run_data_list = []
    for run_name in config["run_order"]:
        selected_ids = selected_paths[selected_paths["group_id"] == run_name]["path_id"].astype(int).tolist()
        run_data_list.append(
            load_run_data(run_name, run_sources[run_name], config["periods"], selected_path_ids=selected_ids)
        )

    issues = validate_runs(run_data_list)
    csv_dir = config["experiment_root"] / experiment / "robustness" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    overlap_frames = []
    overlap_summary_frames = []
    corr_frames = []
    corr_summary_frames = []
    class_frames = []
    freq_frames = []
    cap_frames = []
    score_frames = []
    consensus_frames = []
    robust_frames = []
    robust_tops = {}
    sensitive_tops = {}

    for tech in TECHS:
        run_lookup = build_run_lookup(run_data_list, tech)
        topk_df, topk_summary_df = compute_topk_overlap(run_lookup, tech, config["run_order"])
        corr_df, corr_summary_df = compute_rank_correlation(run_lookup, tech, config["run_order"])
        merged = build_merged_grid_frame(run_lookup, config["run_order"])
        class_df, freq_df, cap_df, score_df, consensus_df, robust_df = compute_grid_stability_tables(
            merged,
            tech,
            config["run_order"],
        )
        robust_top20, sensitive_df = build_priority_lists(robust_df, cap_df, freq_df, score_df)
        overlap_frames.append(topk_df)
        overlap_summary_frames.append(topk_summary_df)
        corr_frames.append(corr_df)
        corr_summary_frames.append(corr_summary_df)
        class_frames.append(class_df)
        freq_frames.append(freq_df)
        cap_frames.append(cap_df)
        score_frames.append(score_df)
        consensus_frames.append(consensus_df)
        robust_frames.append(robust_df)
        robust_tops[tech] = robust_top20
        sensitive_tops[tech] = sensitive_df

    overlap_df = pd.concat(overlap_frames, ignore_index=True)
    overlap_summary_df = pd.concat(overlap_summary_frames, ignore_index=True)
    corr_df = pd.concat(corr_frames, ignore_index=True)
    corr_summary_df = pd.concat(corr_summary_frames, ignore_index=True)
    class_df = pd.concat(class_frames, ignore_index=True)
    freq_df = pd.concat(freq_frames, ignore_index=True)
    cap_df = pd.concat(cap_frames, ignore_index=True)
    score_df = pd.concat(score_frames, ignore_index=True)
    consensus_df = pd.concat(consensus_frames, ignore_index=True)
    robust_df = pd.concat(robust_frames, ignore_index=True)
    summary_df = make_robustness_summary(overlap_summary_df, corr_summary_df, class_df, freq_df, cap_df, robust_df)

    overlap_df.to_csv(csv_dir / "robustness_topk_overlap.csv", index=False)
    overlap_summary_df.to_csv(csv_dir / "robustness_topk_overlap_summary.csv", index=False)
    corr_df.to_csv(csv_dir / "robustness_rank_correlation.csv", index=False)
    class_df.to_csv(csv_dir / "priority_class_stability.csv", index=False)
    freq_df.to_csv(csv_dir / "frequency_stability.csv", index=False)
    cap_df.to_csv(csv_dir / "capacity_stability.csv", index=False)
    score_df.to_csv(csv_dir / "priority_score_stability.csv", index=False)
    consensus_df.to_csv(csv_dir / "grid_priority_consensus.csv", index=False)
    robust_df.to_csv(csv_dir / "robust_priority_classification.csv", index=False)
    summary_df.to_csv(csv_dir / "robustness_summary.csv", index=False)
    robust_tops["wind"].to_csv(csv_dir / "robust_wind_priority_top20.csv", index=False)
    robust_tops["solar"].to_csv(csv_dir / "robust_solar_priority_top20.csv", index=False)
    sensitive_tops["wind"].to_csv(csv_dir / "scenario_sensitive_wind_grids.csv", index=False)
    sensitive_tops["solar"].to_csv(csv_dir / "scenario_sensitive_solar_grids.csv", index=False)

    line = f"{experiment}: data issues -> none" if not issues else f"{experiment}: data issues -> {'; '.join(issues)}"
    write_validation_report(config, [line])
    return summary_df, issues


def summarize_mechanism_experiment(config):
    ensure_experiment_dirs(config)
    selected_paths = load_selected_paths(config)
    rows = []
    text_lines = [
        "Spatial Resource Homogenization Counterfactual Experiment",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for experiment in ["baseline", "wind_homogenized", "solar_homogenized"]:
        summary_file = config["experiment_root"] / experiment / "robustness" / "csv" / "robustness_summary.csv"
        if not summary_file.exists():
            continue
        summary_df = pd.read_csv(summary_file)
        if experiment == "baseline":
            run_sources = config["baseline_runs"]
        else:
            run_sources = {
                run_name: config["experiment_root"] / experiment / "results" / run_name
                for run_name in config["run_order"]
            }
        for tech in TECHS:
            tech_row = summary_df[summary_df["technology"] == tech].iloc[0]
            total_mean, total_std = compute_total_capacity_stats(run_sources, tech, selected_paths)
            rows.append(
                {
                    "experiment": experiment,
                    "technology": tech,
                    "mean_top20_overlap": tech_row["top20_mean_overlap"],
                    "mean_spearman": tech_row["mean_spearman"],
                    "mean_capacity_cv": tech_row["mean_capacity_cv"],
                    "sensitive_grid_count": tech_row["n_S"],
                    "total_capacity_mean": total_mean,
                    "total_capacity_std": total_std,
                }
            )
            text_lines.append(
                f"{experiment} | {tech}: "
                f"top20={tech_row['top20_mean_overlap']:.3f}, "
                f"spearman={tech_row['mean_spearman']:.3f}, "
                f"capacity_cv={tech_row['mean_capacity_cv']:.3f}, "
                f"sensitive={int(tech_row['n_S'])}, "
                f"total_mean={total_mean:.1f}, total_std={total_std:.1f}"
            )
    pd.DataFrame(rows).to_csv(config["summary_dir"] / "mechanism_experiment_summary.csv", index=False)
    (config["summary_dir"] / "mechanism_experiment_summary.txt").write_text("\n".join(text_lines), encoding="utf-8")


def write_validation_report(config, lines, mode="a"):
    report_file = config["summary_dir"] / "validation_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    if mode == "w":
        report_file.write_text("Validation Report\n\n", encoding="utf-8")
    with report_file.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Spatial resource homogenization counterfactual experiment.")
    parser.add_argument(
        "--config",
        default=str(EXPERIMENT_ROOT_DEFAULT / "config.template.json"),
        help="Path to the experiment config JSON.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("select-paths", help="Select representative weather paths.")
    build_parser = subparsers.add_parser("build-weather", help="Build homogenized weather files.")
    build_parser.add_argument("--variant", choices=sorted(VARIANT_TO_TECH), required=True)
    run_parser = subparsers.add_parser("run-sceq", help="Run the full four-stage counterfactual SCEQ.")
    run_parser.add_argument("--variant", choices=sorted(VARIANT_TO_TECH), required=True)
    run_parser.add_argument("--groups", nargs="*", help="Optional subset of groups, for example V1 V2.")
    run_parser.add_argument("--dry-run", action="store_true", help="Check files only, do not launch the solver.")
    analyze_parser = subparsers.add_parser("analyze", help="Run robustness analysis.")
    analyze_parser.add_argument("--experiment", choices=["baseline", *sorted(VARIANT_TO_TECH)], required=True)
    subparsers.add_parser("summarize", help="Build the final summary files.")
    subparsers.add_parser("clean", help="Delete generated experiment outputs inside mechanism_experiment.")
    run_all_parser = subparsers.add_parser("run-all", help="Run the whole experiment pipeline.")
    run_all_parser.add_argument("--skip-sceq", action="store_true", help="Skip the counterfactual reruns.")
    run_all_parser.add_argument("--dry-run", action="store_true", help="Check SCEQ inputs without solving.")
    run_all_parser.add_argument("--clean", action="store_true", help="Clean generated experiment outputs first.")
    return parser.parse_args()


def run_all(config, skip_sceq=False, dry_run=False, do_clean=False):
    if do_clean:
        clean_generated_outputs(config)
    select_paths(config)
    build_counterfactual_weather(config, "wind_homogenized")
    build_counterfactual_weather(config, "solar_homogenized")
    analyze_experiment(config, "baseline")
    if not skip_sceq:
        run_sceq_for_variant(config, "wind_homogenized", dry_run=dry_run)
        run_sceq_for_variant(config, "solar_homogenized", dry_run=dry_run)
        if not dry_run:
            analyze_experiment(config, "wind_homogenized")
            analyze_experiment(config, "solar_homogenized")
    summarize_mechanism_experiment(config)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.command == "select-paths":
        select_paths(config)
    elif args.command == "build-weather":
        build_counterfactual_weather(config, args.variant)
    elif args.command == "run-sceq":
        run_sceq_for_variant(config, args.variant, groups=args.groups, dry_run=args.dry_run)
    elif args.command == "analyze":
        analyze_experiment(config, args.experiment)
    elif args.command == "summarize":
        summarize_mechanism_experiment(config)
    elif args.command == "clean":
        clean_generated_outputs(config)
    elif args.command == "run-all":
        run_all(config, skip_sceq=args.skip_sceq, dry_run=args.dry_run, do_clean=args.clean)


if __name__ == "__main__":
    main()



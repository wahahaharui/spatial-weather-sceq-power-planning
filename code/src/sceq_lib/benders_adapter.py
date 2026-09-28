"""
Benders V7 Adapter - original project adapter.
"""
import os
import sys
import time
import gc
import numpy as np
import pandas as pd

from .config import cfg_v2 as cfg
from .time_aggregator import aggregate_8760_to_288

_benders_dir = str(cfg.BENDERS_V6_DIR.parent)
if _benders_dir not in sys.path:
    sys.path.insert(0, _benders_dir)


class BendersAdapter:
    def __init__(self):
        benders_dir = str(cfg.BENDERS_V6_DIR.parent)
        if benders_dir not in sys.path:
            sys.path.insert(0, benders_dir)

        self._template_csvs = {}
        for fname in os.listdir(cfg.INPUTS_TEMPLATE):
            if fname.endswith(".csv") and fname not in [
                "variable_capacity_factors.csv",
                "loads.csv",
                "gen_build_predetermined.csv",
            ]:
                self._template_csvs[fname] = pd.read_csv(cfg.INPUTS_TEMPLATE / fname)

        self._base_gen_build_costs = pd.read_csv(cfg.INPUTS_TEMPLATE / "gen_build_costs.csv")
        self._base_predetermined = pd.read_csv(cfg.INPUTS_TEMPLATE / "gen_build_predetermined.csv")

        gen_info = pd.read_csv(cfg.INPUTS_TEMPLATE / "gen_info.csv")
        self.wind_projects = [
            g for g in gen_info["GENERATION_PROJECT"].tolist()
            if g.lower().endswith("_w") or "wind" in g.lower()
        ]
        self.solar_projects = [
            g for g in gen_info["GENERATION_PROJECT"].tolist()
            if g.lower().endswith("_s") or "solar" in g.lower() or "pv" in g.lower()
        ]
        self.storage_projects = [
            g for g in gen_info["GENERATION_PROJECT"].tolist()
            if "storage" in g.lower() or "battery" in g.lower() or "stor" in g.lower()
        ]
        self.gen_info = gen_info

    def run_one_step(self, current_period, remaining_periods,
                     path_weather_8760, previous_decisions,
                     ce_baseline_1152, path_id,
                     warm_start_sol=None):
        step_work_dir = cfg.WORK_DIRS / f"path_{path_id:03d}" / f"step_{current_period}"
        step_work_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()

        self._write_period_csvs(step_work_dir, remaining_periods)
        self._write_predetermined(step_work_dir, previous_decisions)
        self._write_cf_parquet(
            step_work_dir, current_period, remaining_periods,
            path_weather_8760, ce_baseline_1152
        )
        self._write_load_parquet(
            step_work_dir, current_period, remaining_periods,
            path_weather_8760, ce_baseline_1152
        )

        prepare_t = time.time() - t0

        from benders_fixed_v6.data_loader import load_all
        from benders_fixed_v6.benders_persistent_v7 import run_benders

        data = load_all(str(step_work_dir))
        result = run_benders(
            data,
            max_iter=cfg.BENDERS_MAX_ITER,
            gap_tol=cfg.BENDERS_GAP_TOL,
            output_dir=str(step_work_dir),
            warm_start_sol=warm_start_sol,
        )
        solve_t = time.time() - t0 - prepare_t

        decisions = self._extract_build_decisions(step_work_dir, current_period)

        master_sol = None
        if result.get("best_sol"):
            master_sol = result["best_sol"]["master"]

        del data
        gc.collect()

        total_t = time.time() - t0
        solve_info = {
            "current_period": current_period,
            "prepare_s": round(prepare_t, 1),
            "solve_s": round(solve_t, 1),
            "total_s": round(total_t, 1),
            "ub": result.get("UB", float("nan")),
            "lb": result.get("LB", float("nan")),
            "iterations": result.get("iterations", 0),
            "total_cuts": result.get("total_cuts", 0),
            "gap": (result["UB"] - result["LB"]) / (abs(result["UB"]) + 1e-10)
                   if result.get("UB") and result.get("LB") else 1.0,
        }

        return decisions, result.get("UB", float("nan")), solve_info, master_sol

    def _write_period_csvs(self, work_dir, remaining_periods):
        for fname, df in self._template_csvs.items():
            df.to_csv(work_dir / fname, index=False)
        gbc = self._base_gen_build_costs[
            self._base_gen_build_costs["build_year"].isin(remaining_periods)
        ]
        gbc.to_csv(work_dir / "gen_build_costs.csv", index=False)

    def _write_predetermined(self, work_dir, previous_decisions):
        pdf = self._base_predetermined.copy()
        extra_rows = []
        for period, decisions in previous_decisions.items():
            for gen_name, mw in decisions.items():
                if mw > 1e-6:
                    extra_rows.append(
                        {
                            "GENERATION_PROJECT": gen_name,
                            "build_year": period,
                            "build_gen_predetermined": float(mw),
                        }
                    )
        if extra_rows:
            pdf = pd.concat([pdf, pd.DataFrame(extra_rows)], ignore_index=True)
        pdf.to_csv(work_dir / "gen_build_predetermined.csv", index=False)

    def _write_cf_parquet(self, work_dir, current_period, remaining_periods,
                          path_weather_8760, ce_baseline_1152):
        all_periods = cfg.PERIODS
        period_offset = {p: i * 288 for i, p in enumerate(all_periods)}
        rows = []

        for period in all_periods:
            offset = period_offset[period]
            if period == current_period:
                pd_data = path_weather_8760[period]
                wind_288 = aggregate_8760_to_288(pd_data["wind_cf"])
                solar_288 = aggregate_8760_to_288(pd_data["solar_cf"])
            elif period in remaining_periods and period != current_period:
                wind_288 = ce_baseline_1152["wind_cf"][offset:offset + 288]
                solar_288 = ce_baseline_1152["solar_cf"][offset:offset + 288]
            else:
                pd_data = path_weather_8760[period]
                wind_288 = aggregate_8760_to_288(pd_data["wind_cf"])
                solar_288 = aggregate_8760_to_288(pd_data["solar_cf"])

            for tp_local in range(288):
                tp_global = offset + tp_local + 1
                for grid_idx, proj_name in enumerate(self.wind_projects):
                    rows.append(
                        {
                            "GENERATION_PROJECT": proj_name,
                            "timepoint": tp_global,
                            "gen_max_capacity_factor": round(float(wind_288[tp_local, grid_idx]), 6),
                        }
                    )
                for grid_idx, proj_name in enumerate(self.solar_projects):
                    rows.append(
                        {
                            "GENERATION_PROJECT": proj_name,
                            "timepoint": tp_global,
                            "gen_max_capacity_factor": round(float(solar_288[tp_local, grid_idx]), 6),
                        }
                    )

        cf_df = pd.DataFrame(rows)
        cf_df.to_parquet(work_dir / "variable_capacity_factors.parquet", index=False)
        cf_df.to_csv(work_dir / "variable_capacity_factors.csv", index=False)

    def _write_load_parquet(self, work_dir, current_period, remaining_periods,
                            path_weather_8760, ce_baseline_1152):
        all_periods = cfg.PERIODS
        period_offset = {p: i * 288 for i, p in enumerate(all_periods)}
        rows = []

        for period in all_periods:
            offset = period_offset[period]
            if period == current_period:
                pd_data = path_weather_8760[period]
                load_288 = aggregate_8760_to_288(pd_data["load"])
            elif period in remaining_periods and period != current_period:
                load_288 = ce_baseline_1152["load"][offset:offset + 288]
            else:
                pd_data = path_weather_8760[period]
                load_288 = aggregate_8760_to_288(pd_data["load"])

            for tp_local in range(288):
                tp_global = offset + tp_local + 1
                rows.append(
                    {
                        "LOAD_ZONE": "inside_lvliang",
                        "TIMEPOINT": tp_global,
                        "zone_demand_mw": round(float(load_288[tp_local]), 4),
                    }
                )

        load_df = pd.DataFrame(rows)
        load_df.to_parquet(work_dir / "loads.parquet", index=False)
        load_df.to_csv(work_dir / "loads.csv", index=False)

    def _extract_build_decisions(self, work_dir, current_period):
        build_csv = work_dir / "gen_build.csv"
        if not build_csv.exists():
            return {}
        df = pd.read_csv(build_csv)
        decisions = {}
        for _, row in df.iterrows():
            if int(row["PERIOD"]) == current_period:
                val = row.get("BuildGen", ".")
                if val != "." and float(val) > 1e-6:
                    decisions[row["GENERATION_PROJECT"]] = float(val)
        return decisions

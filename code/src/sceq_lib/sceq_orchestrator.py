"""
SCEQ Orchestrator V2 - original project orchestrator.
"""
import json
import time
import gc
import traceback
import threading
import io
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import cfg_v2 as cfg


def _worker_run_path_v2(path_id, ce_baseline_1152):
    from .scenario_generator import ScenarioGenerator
    from .benders_adapter import BendersAdapter

    path_weather = ScenarioGenerator.load_path(path_id)
    adapter = BendersAdapter()
    previous_decisions = {}
    warm_start_sol = None
    steps = []
    cumul_wind = cumul_solar = cumul_storage = 0.0

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    for step_idx, period in enumerate(cfg.PERIODS):
        remaining = cfg.PERIODS[step_idx:]

        try:
            decisions, cost, info, master_sol = adapter.run_one_step(
                current_period=period,
                remaining_periods=remaining,
                path_weather_8760=path_weather,
                previous_decisions=previous_decisions,
                ce_baseline_1152=ce_baseline_1152,
                path_id=path_id,
                warm_start_sol=warm_start_sol,
            )
        except Exception as exc:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return {"path_id": path_id, "error": f"Step {period}: {exc}", "steps": steps}

        warm_start_sol = master_sol
        previous_decisions[period] = decisions

        cumul_wind += sum(v for k, v in decisions.items() if k in adapter.wind_projects)
        cumul_solar += sum(v for k, v in decisions.items() if k in adapter.solar_projects)
        cumul_storage += sum(v for k, v in decisions.items() if k in adapter.storage_projects)

        steps.append(
            {
                "period": period,
                "wind_build_mw": round(cumul_wind, 1),
                "solar_build_mw": round(cumul_solar, 1),
                "storage_build_mw": round(cumul_storage, 1),
                "total_cost": cost if np.isfinite(cost) else None,
                "solve_info": info,
                "_n_decisions": len(decisions),
            }
        )

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        ws_tag = "[WS]" if warm_start_sol is not None else "[cold]"
        print(
            f"  [Path {path_id:03d}] {ws_tag} Step {period}: "
            f"Wind+{cumul_wind:.0f}MW Solar+{cumul_solar:.0f}MW "
            f"Stor+{cumul_storage:.0f}MW | {info['solve_s']:.0f}s "
            f"| iter={info['iterations']} gap={info.get('gap',1):.4f}"
        )
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    del path_weather, adapter
    gc.collect()
    return {"path_id": path_id, "steps": steps}


class SCEQOrchestrator:
    def __init__(self):
        self.checkpoint_file = cfg.CHECKPOINT_FILE
        self.t_start = None
        self._lock = threading.Lock()
        self._n_completed = 0

    def run(self, ce_baseline_1152):
        self.t_start = time.time()

        checkpoint = self._load_checkpoint()
        completed = set(checkpoint.get("completed_paths", []))
        all_results = checkpoint.get("all_results", [])
        self._n_completed = len(completed)

        if completed:
            print(f"[SCEQ] Resuming: {len(completed)} done, {cfg.N_PATHS - len(completed)} remaining")

        pending = [pid for pid in range(cfg.N_PATHS) if pid not in completed]
        if not pending:
            print("[SCEQ] All paths complete!")
            return all_results

        n_workers = min(cfg.N_WORKERS, len(pending))
        print(f"[SCEQ] {n_workers} workers, {len(pending)} paths pending")

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_worker_run_path_v2, pid, ce_baseline_1152): pid for pid in pending}
            for future in as_completed(futures):
                path_id = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                except Exception as exc:
                    print(f"  ERROR Path {path_id:03d}: {exc}")
                    traceback.print_exc()
                    all_results.append({"path_id": path_id, "error": str(exc), "steps": []})

                completed.add(path_id)
                self._n_completed = len(completed)

                with self._lock:
                    if self._n_completed % cfg.CHECKPOINT_INTERVAL == 0:
                        self._save_checkpoint(list(completed), all_results)
                    if self._n_completed % 5 == 0:
                        self._print_progress(all_results, completed)

        self._save_checkpoint(list(completed), all_results)
        total_t = time.time() - self.t_start
        n_success = sum(1 for r in all_results if "error" not in r)
        n_fail = sum(1 for r in all_results if "error" in r)
        print(f"\n{'='*70}")
        print(f"[SCEQ] Done! Wall time: {total_t/3600:.1f}h")
        print(f"  Success: {n_success} | Failed: {n_fail}")
        print(f"{'='*70}")
        return all_results

    def _print_progress(self, all_results, completed):
        n = len(completed)
        if n < 2:
            return
        elapsed = time.time() - self.t_start
        avg_t = elapsed / n
        eta_s = avg_t * (cfg.N_PATHS - n)

        success = [r for r in all_results if "error" not in r and r.get("steps")]
        if not success:
            return

        costs = []
        iters_list = []
        for result in success:
            for step in result["steps"]:
                if step.get("total_cost") and np.isfinite(step["total_cost"]):
                    costs.append(step["total_cost"])
                iters_list.append(step.get("solve_info", {}).get("iterations", 0))

        print(f"\n  {'-'*55}")
        print(f"  SCEQ Progress: {n}/{cfg.N_PATHS} paths ({n*100//cfg.N_PATHS}%)")
        print(f"  Elapsed: {elapsed/60:.0f}min | ETA: {eta_s/60:.0f}min")
        if costs:
            print(f"  Avg cost: ${np.mean(costs):,.0f} | Avg iter: {np.mean(iters_list):.0f}")
        print(f"  {'-'*55}")

    def _load_checkpoint(self):
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"completed_paths": [], "all_results": []}

    def _save_checkpoint(self, completed, all_results):
        serializable = []
        for result in all_results:
            row = {"path_id": result["path_id"]}
            if "error" in result:
                row["error"] = result["error"]
            if "steps" in result:
                row["steps"] = []
                for step in result["steps"]:
                    row["steps"].append(
                        {
                            "period": step["period"],
                            "wind_build_mw": step.get("wind_build_mw", 0),
                            "solar_build_mw": step.get("solar_build_mw", 0),
                            "storage_build_mw": step.get("storage_build_mw", 0),
                            "total_cost": step.get("total_cost"),
                            "solve_info": step.get("solve_info", {}),
                        }
                    )
            serializable.append(row)
        with open(self.checkpoint_file, "w") as f:
            json.dump(
                {
                    "completed_paths": sorted(completed),
                    "all_results": serializable,
                    "elapsed_s": time.time() - self.t_start,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
                default=str,
            )

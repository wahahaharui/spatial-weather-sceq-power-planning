import json
import math
import os
import sys
import textwrap
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT))

# ============================================================
# User configuration
# ============================================================
MULTI_RUN_ROOT = Path(os.environ.get("SCEQ_MULTI_RUN_ROOT", CODE_ROOT / "output_runs")).resolve()
RUN_FILES = {
    "V1": Path(os.environ.get("SCEQ_RUN_V1", MULTI_RUN_ROOT / "output")).resolve(),
    "V2": Path(os.environ.get("SCEQ_RUN_V2", MULTI_RUN_ROOT / "output_v2")).resolve(),
    "V3": Path(os.environ.get("SCEQ_RUN_V3", MULTI_RUN_ROOT / "output_v3")).resolve(),
    "V4": Path(os.environ.get("SCEQ_RUN_V4", MULTI_RUN_ROOT / "output_v4")).resolve(),
}

OUTPUT_ROOT = Path(os.environ.get("SCEQ_ROBUSTNESS_OUTPUT", CODE_ROOT / "robustness_analysis")).resolve()

RUN_ORDER = ["V1", "V2", "V3", "V4"]
TECHS = ["wind", "solar"]
TOPK_LIST = [5, 10, 20, 50]
PERIODS = [2025, 2030, 2035, 2040]
VALID_CLASSES = {"A", "B", "C", "D", "None"}


@dataclass
class RunData:
    name: str
    path: Path
    df: pd.DataFrame


def ensure_dirs():
    dirs = {
        "root": OUTPUT_ROOT,
        "csv": OUTPUT_ROOT / "csv",
        "figures": OUTPUT_ROOT / "figures",
        "latex": OUTPUT_ROOT / "latex",
        "report": OUTPUT_ROOT / "report",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _find_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None


def _safe_cv(std_val, mean_val):
    if pd.isna(mean_val) or math.isclose(float(mean_val), 0.0, abs_tol=1e-12):
        return np.nan
    return float(std_val) / float(mean_val)


def safe_div(numerator, denominator):
    if denominator in (0, 0.0) or pd.isna(denominator):
        return 0.0
    return float(numerator) / float(denominator)


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


def _latex_escape(x):
    text = str(x)
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ]:
        text = text.replace(a, b)
    return text


def load_run_csv(run_name, path):
    if not path.exists():
        raise FileNotFoundError(f"{run_name} source not found: {path}")

    if path.is_dir():
        df = build_priority_summary_from_output_root(path, run_name)
    else:
        df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    grid_col = _find_col(df, ["grid_id"])
    lon_col = _find_col(df, ["longitude", "lon"])
    lat_col = _find_col(df, ["latitude", "lat"])

    rename_map = {
        grid_col: "grid_id",
        lon_col: "longitude",
        lat_col: "latitude",
    }

    for tech in TECHS:
        rename_map[_find_col(df, [f"{tech}_freq_final"])] = f"{tech}_freq_final"
        rename_map[_find_col(df, [f"{tech}_recommended_mw"])] = f"{tech}_recommended_mw"
        rename_map[_find_col(df, [f"{tech}_priority_score"])] = f"{tech}_priority_score"
        rename_map[_find_col(df, [f"{tech}_priority_class"])] = f"{tech}_priority_class"
        rename_map[_find_col(df, [f"{tech}_first_build_period_mode"])] = f"{tech}_first_build_period_mode"
        rename_map[_find_col(df, [f"{tech}_earliness_score"])] = f"{tech}_earliness_score"
        rename_map[_find_col(df, [f"{tech}_p25_mw"])] = f"{tech}_p25_mw"
        rename_map[_find_col(df, [f"{tech}_p75_mw"])] = f"{tech}_p75_mw"

    df = df.rename(columns=rename_map).copy()
    df["grid_id"] = df["grid_id"].astype(str)
    df = df.drop_duplicates("grid_id").reset_index(drop=True)

    numeric_cols = [
        "longitude",
        "latitude",
        *[f"{t}_freq_final" for t in TECHS],
        *[f"{t}_recommended_mw" for t in TECHS],
        *[f"{t}_priority_score" for t in TECHS],
        *[f"{t}_first_build_period_mode" for t in TECHS],
        *[f"{t}_earliness_score" for t in TECHS],
        *[f"{t}_p25_mw" for t in TECHS],
        *[f"{t}_p75_mw" for t in TECHS],
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for tech in TECHS:
        df[f"{tech}_priority_class"] = df[f"{tech}_priority_class"].astype(str).fillna("None")

    return RunData(run_name, path, df)


def build_priority_summary_from_output_root(output_root, run_name):
    checkpoint_file = output_root / "checkpoint.json"
    robustness_file = output_root / "04_summary" / "02_grid_robustness_data.csv"
    work_dirs = output_root / "05_work_dirs"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"{run_name}: missing checkpoint file: {checkpoint_file}")
    if not robustness_file.exists():
        raise FileNotFoundError(f"{run_name}: missing robustness file: {robustness_file}")
    if not work_dirs.exists():
        raise FileNotFoundError(f"{run_name}: missing work dirs: {work_dirs}")

    with checkpoint_file.open("r", encoding="utf-8") as f:
        checkpoint = json.load(f)
    success_paths = [r for r in checkpoint.get("all_results", []) if "error" not in r and r.get("steps")]
    if not success_paths:
        raise RuntimeError(f"{run_name}: no successful paths found in checkpoint")

    robustness = pd.read_csv(robustness_file)
    robustness.columns = [str(c).strip() for c in robustness.columns]
    robustness["grid_id"] = robustness["grid_id"].astype(str)

    coord_candidates = [c for c in ["lon", "longitude"] if c in robustness.columns]
    lat_candidates = [c for c in ["lat", "latitude"] if c in robustness.columns]
    if not coord_candidates or not lat_candidates:
        raise KeyError(f"{run_name}: robustness file must contain lon/lat or longitude/latitude")
    if "lon" not in robustness.columns:
        robustness = robustness.rename(columns={coord_candidates[0]: "lon"})
    if "lat" not in robustness.columns:
        robustness = robustness.rename(columns={lat_candidates[0]: "lat"})

    grid_base = robustness[["grid_id", "lon", "lat"]].drop_duplicates("grid_id").copy()
    grid_ids = sorted(grid_base["grid_id"].unique())
    path_ids = sorted(r["path_id"] for r in success_paths)

    new_build = {
        tech: {
            period: pd.DataFrame(0.0, index=grid_ids, columns=path_ids)
            for period in PERIODS
        }
        for tech in TECHS
    }
    final_build = {
        tech: pd.DataFrame(0.0, index=grid_ids, columns=path_ids)
        for tech in TECHS
    }

    for result in success_paths:
        path_id = result["path_id"]
        cumul = {tech: {} for tech in TECHS}
        for step in sorted(result["steps"], key=lambda x: x["period"]):
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
                prev_val = cumul[row.tech].get(row.grid_id, 0.0)
                cumul[row.tech][row.grid_id] = prev_val + float(row.BuildGen)
                final_build[row.tech].loc[row.grid_id, path_id] = cumul[row.tech][row.grid_id]

        for tech in TECHS:
            for grid_id, cap in cumul[tech].items():
                final_build[tech].loc[grid_id, path_id] = cap

    template = work_dirs / "path_000" / "step_2040" / "gen_info.csv"
    if not template.exists():
        candidates = sorted(work_dirs.glob("path_*\\step_2040\\gen_info.csv"))
        if not candidates:
            raise FileNotFoundError(f"{run_name}: no step_2040/gen_info.csv found under {work_dirs}")
        template = candidates[0]
    gen_info = pd.read_csv(template)
    mask = gen_info["GENERATION_PROJECT"].astype(str).str.match(r"LL_\d+_[ws]$")
    gen_info = gen_info.loc[mask, ["GENERATION_PROJECT", "gen_capacity_limit_mw"]].copy()
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
            "lon": float(base_row["lon"]),
            "lat": float(base_row["lat"]),
        }
        for tech in TECHS:
            final_arr = final_build[tech].loc[grid_id].to_numpy(dtype=float)
            positive_final = final_arr[final_arr > 1e-6]
            row[f"{tech}_developed_paths"] = int(np.sum(final_arr > 1e-6))
            row[f"{tech}_freq_final"] = float(np.mean(final_arr > 1e-6))
            row[f"{tech}_recommended_mw"] = float(np.median(positive_final)) if len(positive_final) else 0.0
            row[f"{tech}_conditional_mean_mw"] = float(np.mean(positive_final)) if len(positive_final) else 0.0
            row[f"{tech}_p25_mw"] = float(np.percentile(positive_final, 25)) if len(positive_final) else 0.0
            row[f"{tech}_p75_mw"] = float(np.percentile(positive_final, 75)) if len(positive_final) else 0.0
            row[f"{tech}_all_path_mean_mw"] = float(np.mean(final_arr))
            row[f"{tech}_all_path_std_mw"] = float(np.std(final_arr))
            if len(positive_final) > 1 and np.mean(positive_final) > 0:
                cv = float(np.std(positive_final) / np.mean(positive_final))
            else:
                cv = 0.0 if len(positive_final) == 1 else 1.0
            row[f"{tech}_stability_raw"] = max(0.0, 1.0 - min(cv, 1.0))

            first_period_counts = {}
            for period in PERIODS:
                arr = new_build[tech][period].loc[grid_id].to_numpy(dtype=float)
                first_period_counts[period] = float(np.mean(arr > 1e-6))
                row[f"{tech}_new_freq_{period}"] = first_period_counts[period]
                row[f"{tech}_new_mean_{period}"] = float(np.mean(arr))

            first_build_periods = []
            for path_id in path_ids:
                first_period = None
                for period in PERIODS:
                    if new_build[tech][period].loc[grid_id, path_id] > 1e-6:
                        first_period = period
                        break
                first_build_periods.append(first_period)
            developed_firsts = [p for p in first_build_periods if p is not None]
            row[f"{tech}_first_build_period_mode"] = (
                int(pd.Series(developed_firsts).mode().iloc[0]) if developed_firsts else np.nan
            )

            weights = {2025: 1.0, 2030: 0.75, 2035: 0.5, 2040: 0.25}
            earliness_score = 0.0
            for period, weight in weights.items():
                earliness_score += first_period_counts[period] * weight
            row[f"{tech}_earliness_score"] = earliness_score

        records.append(row)

    stats = pd.DataFrame(records)
    limits_pivot = limits.pivot(index="grid_id", columns="tech", values="capacity_limit_mw").reset_index()
    stats = stats.merge(limits_pivot, on="grid_id", how="left")
    stats = stats.rename(columns={"wind": "wind_capacity_limit_mw", "solar": "solar_capacity_limit_mw"})
    stats["wind_capacity_limit_mw"] = stats["wind_capacity_limit_mw"].fillna(0.0)
    stats["solar_capacity_limit_mw"] = stats["solar_capacity_limit_mw"].fillna(0.0)

    for tech in TECHS:
        stats[f"{tech}_development_intensity"] = stats.apply(
            lambda r: safe_div(r[f"{tech}_recommended_mw"], r[f"{tech}_capacity_limit_mw"]),
            axis=1,
        )
        stats[f"{tech}_expected_mw"] = stats[f"{tech}_all_path_mean_mw"]

    stats = score_and_classify_priority(stats)
    ordered_cols = [
        "grid_id",
        "lon",
        "lat",
        "combined_priority_class",
        "combined_priority_score",
        "combined_recommended_mw",
        "wind_priority_class",
        "wind_priority_score",
        "wind_freq_final",
        "wind_first_build_period_mode",
        "wind_earliness_score",
        "wind_recommended_mw",
        "wind_p25_mw",
        "wind_p75_mw",
        "wind_capacity_limit_mw",
        "wind_development_intensity",
        "solar_priority_class",
        "solar_priority_score",
        "solar_freq_final",
        "solar_first_build_period_mode",
        "solar_earliness_score",
        "solar_recommended_mw",
        "solar_p25_mw",
        "solar_p75_mw",
        "solar_capacity_limit_mw",
        "solar_development_intensity",
    ]
    return stats[ordered_cols].copy()


def score_and_classify_priority(stats):
    stats = stats.copy()
    for tech in TECHS:
        stats[f"{tech}_capacity_norm"] = normalize_series(stats[f"{tech}_recommended_mw"])
        stats[f"{tech}_freq_norm"] = normalize_series(stats[f"{tech}_freq_final"])
        stats[f"{tech}_earliness_norm"] = normalize_series(stats[f"{tech}_earliness_score"])
        stats[f"{tech}_stability_norm"] = normalize_series(stats[f"{tech}_stability_raw"])
        stats[f"{tech}_intensity_norm"] = normalize_series(stats[f"{tech}_development_intensity"])

        stats[f"{tech}_priority_score"] = (
            0.35 * stats[f"{tech}_freq_norm"]
            + 0.25 * stats[f"{tech}_earliness_norm"]
            + 0.20 * stats[f"{tech}_capacity_norm"]
            + 0.10 * stats[f"{tech}_stability_norm"]
            + 0.10 * stats[f"{tech}_intensity_norm"]
        )

        stats[f"{tech}_priority_class"] = stats.apply(lambda r: priority_class(r, tech), axis=1)

    stats["combined_priority_score"] = 0.5 * stats["wind_priority_score"] + 0.5 * stats["solar_priority_score"]
    stats["combined_recommended_mw"] = stats["wind_recommended_mw"] + stats["solar_recommended_mw"]
    stats["combined_priority_class"] = stats.apply(combined_priority_class, axis=1)
    return stats.sort_values("combined_priority_score", ascending=False).reset_index(drop=True)


def priority_class(row, tech):
    freq = row[f"{tech}_freq_final"]
    score = row[f"{tech}_priority_score"]
    rec = row[f"{tech}_recommended_mw"]
    if rec <= 1e-6 and freq <= 1e-6:
        return "None"
    if freq >= 0.70 and score >= 0.65:
        return "A"
    if freq >= 0.35 and score >= 0.42:
        return "B"
    if freq >= 0.10 and rec > 0:
        return "C"
    return "D"


def combined_priority_class(row):
    wind_score = row["wind_priority_score"]
    solar_score = row["solar_priority_score"]
    if max(wind_score, solar_score) < 0.05:
        return "None"
    score = row["combined_priority_score"]
    if score >= 0.60:
        return "A"
    if score >= 0.38:
        return "B"
    if score >= 0.12:
        return "C"
    return "D"


def validate_runs(run_data_list):
    issues = []
    base_ids = set(run_data_list[0].df["grid_id"])
    base_coords = run_data_list[0].df.set_index("grid_id")[["longitude", "latitude"]]

    for run_data in run_data_list:
        df = run_data.df
        if df["grid_id"].duplicated().any():
            issues.append(f"{run_data.name}: duplicated grid_id found")
        if df["grid_id"].isna().any():
            issues.append(f"{run_data.name}: missing grid_id found")

        for tech in TECHS:
            score_col = f"{tech}_priority_score"
            freq_col = f"{tech}_freq_final"
            cap_col = f"{tech}_recommended_mw"
            cls_col = f"{tech}_priority_class"

            if df[score_col].isna().any():
                issues.append(f"{run_data.name}: NaN in {score_col}")
            bad_freq = df[(df[freq_col] < -1e-9) | (df[freq_col] > 1 + 1e-9)]
            if not bad_freq.empty:
                issues.append(f"{run_data.name}: {freq_col} outside [0,1]")
            bad_cap = df[df[cap_col] < -1e-9]
            if not bad_cap.empty:
                issues.append(f"{run_data.name}: {cap_col} contains negative values")
            bad_cls = sorted(set(df[cls_col]) - VALID_CLASSES)
            if bad_cls:
                issues.append(f"{run_data.name}: unexpected classes in {cls_col}: {bad_cls}")

        ids = set(df["grid_id"])
        missing_vs_base = sorted(base_ids - ids)
        extra_vs_base = sorted(ids - base_ids)
        if missing_vs_base or extra_vs_base:
            issues.append(
                f"{run_data.name}: grid_id set differs from {run_data_list[0].name} "
                f"(missing={len(missing_vs_base)}, extra={len(extra_vs_base)})"
            )

        merged = df.set_index("grid_id")[["longitude", "latitude"]].join(
            base_coords, how="inner", lsuffix="_this", rsuffix="_base"
        )
        if not merged.empty:
            mismatch = merged[
                (~np.isclose(merged["longitude_this"], merged["longitude_base"], equal_nan=True))
                | (~np.isclose(merged["latitude_this"], merged["latitude_base"], equal_nan=True))
            ]
            if not mismatch.empty:
                issues.append(
                    f"{run_data.name}: coordinate mismatch for {len(mismatch)} grids relative to {run_data_list[0].name}"
                )

    return issues


def save_data_quality_report(run_data_list, issues, csv_dir, report_dir):
    rows = []
    for run_data in run_data_list:
        rows.append(
            {
                "run": run_data.name,
                "source_file": str(run_data.path),
                "n_rows": len(run_data.df),
                "n_unique_grids": run_data.df["grid_id"].nunique(),
            }
        )
    pd.DataFrame(rows).to_csv(csv_dir / "data_quality_overview.csv", index=False)

    common = sorted(set.intersection(*(set(r.df["grid_id"]) for r in run_data_list)))
    union = sorted(set.union(*(set(r.df["grid_id"]) for r in run_data_list)))
    pd.DataFrame({"grid_id": common}).to_csv(csv_dir / "common_grid_ids.csv", index=False)
    pd.DataFrame({"grid_id": union}).to_csv(csv_dir / "union_grid_ids.csv", index=False)

    lines = ["# Data Quality Report", ""]
    lines.append("## Input Files")
    for run_data in run_data_list:
        lines.append(f"- {run_data.name}: `{run_data.path}`")
    lines.append("")
    if issues:
        lines.append("## Warnings")
        for issue in issues:
            lines.append(f"- {issue}")
    else:
        lines.append("## Warnings")
        lines.append("- No data quality warnings detected.")
    (report_dir / "data_quality_report.md").write_text("\n".join(lines), encoding="utf-8")


def build_run_lookup(run_data_list, tech):
    lookup = {}
    for run_data in run_data_list:
        sub = run_data.df[
            [
                "grid_id",
                "longitude",
                "latitude",
                f"{tech}_freq_final",
                f"{tech}_recommended_mw",
                f"{tech}_priority_score",
                f"{tech}_priority_class",
                f"{tech}_first_build_period_mode",
                f"{tech}_earliness_score",
                f"{tech}_p25_mw",
                f"{tech}_p75_mw",
            ]
        ].copy()
        sub = sub.rename(
            columns={
                f"{tech}_freq_final": "frequency",
                f"{tech}_recommended_mw": "capacity",
                f"{tech}_priority_score": "priority_score",
                f"{tech}_priority_class": "priority_class",
                f"{tech}_first_build_period_mode": "first_build_period_mode",
                f"{tech}_earliness_score": "earliness_score",
                f"{tech}_p25_mw": "p25_mw",
                f"{tech}_p75_mw": "p75_mw",
            }
        )
        lookup[run_data.name] = sub
    return lookup


def compute_topk_overlap(run_lookup, tech):
    pair_rows = []
    summary_rows = []
    rank_map = {}
    for run_name, df in run_lookup.items():
        ranked = df.sort_values(["priority_score", "grid_id"], ascending=[False, True]).reset_index(drop=True)
        rank_map[run_name] = ranked

    for k in TOPK_LIST:
        ratios = []
        for run_i, run_j in combinations(RUN_ORDER, 2):
            top_i = set(rank_map[run_i].head(k)["grid_id"])
            top_j = set(rank_map[run_j].head(k)["grid_id"])
            overlap_count = len(top_i & top_j)
            overlap_ratio = overlap_count / k
            ratios.append(overlap_ratio)
            pair_rows.append(
                {
                    "technology": tech,
                    "k": k,
                    "run_i": run_i,
                    "run_j": run_j,
                    "overlap_count": overlap_count,
                    "overlap_ratio": overlap_ratio,
                }
            )
        summary_rows.append(
            {
                "technology": tech,
                "k": k,
                "mean_pairwise_overlap": float(np.mean(ratios)),
                "std_pairwise_overlap": float(np.std(ratios)),
                "min_pairwise_overlap": float(np.min(ratios)),
                "max_pairwise_overlap": float(np.max(ratios)),
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def compute_rank_correlation(run_lookup, tech):
    rows = []
    summary_vals = []
    heatmap = pd.DataFrame(np.eye(len(RUN_ORDER)), index=RUN_ORDER, columns=RUN_ORDER)

    for run_i, run_j in combinations(RUN_ORDER, 2):
        merged = run_lookup[run_i][["grid_id", "priority_score"]].merge(
            run_lookup[run_j][["grid_id", "priority_score"]],
            on="grid_id",
            how="inner",
            suffixes=(f"_{run_i}", f"_{run_j}"),
        )
        rho, pvalue = spearmanr(
            merged[f"priority_score_{run_i}"],
            merged[f"priority_score_{run_j}"],
            nan_policy="omit",
        )
        tau, _ = kendalltau(
            merged[f"priority_score_{run_i}"],
            merged[f"priority_score_{run_j}"],
            nan_policy="omit",
        )
        rows.append(
            {
                "technology": tech,
                "run_i": run_i,
                "run_j": run_j,
                "n_grids": len(merged),
                "spearman_rho": float(rho),
                "spearman_pvalue": float(pvalue),
                "kendall_tau": float(tau) if tau is not None else np.nan,
            }
        )
        summary_vals.append(float(rho))
        heatmap.loc[run_i, run_j] = float(rho)
        heatmap.loc[run_j, run_i] = float(rho)

    summary = pd.DataFrame(
        [
            {
                "technology": tech,
                "mean_spearman": float(np.mean(summary_vals)),
                "std_spearman": float(np.std(summary_vals)),
                "min_spearman": float(np.min(summary_vals)),
                "max_spearman": float(np.max(summary_vals)),
            }
        ]
    )
    return pd.DataFrame(rows), summary, heatmap


def build_merged_grid_frame(run_lookup):
    merged = None
    for run_name in RUN_ORDER:
        df = run_lookup[run_name].copy()
        rename = {
            "longitude": f"{run_name.lower()}_longitude",
            "latitude": f"{run_name.lower()}_latitude",
            "frequency": f"{run_name.lower()}_frequency",
            "capacity": f"{run_name.lower()}_capacity",
            "priority_score": f"{run_name.lower()}_priority_score",
            "priority_class": f"{run_name.lower()}_class",
            "first_build_period_mode": f"{run_name.lower()}_first_build_period_mode",
            "earliness_score": f"{run_name.lower()}_earliness_score",
            "p25_mw": f"{run_name.lower()}_p25_mw",
            "p75_mw": f"{run_name.lower()}_p75_mw",
        }
        df = df.rename(columns=rename)
        merged = df if merged is None else merged.merge(df, on="grid_id", how="outer")

    lon_cols = [f"{run.lower()}_longitude" for run in RUN_ORDER]
    lat_cols = [f"{run.lower()}_latitude" for run in RUN_ORDER]
    merged["longitude"] = merged[lon_cols].bfill(axis=1).iloc[:, 0]
    merged["latitude"] = merged[lat_cols].bfill(axis=1).iloc[:, 0]
    return merged


def compute_grid_stability_tables(merged, tech):
    run_cols = {
        "frequency": [f"{run.lower()}_frequency" for run in RUN_ORDER],
        "capacity": [f"{run.lower()}_capacity" for run in RUN_ORDER],
        "priority_score": [f"{run.lower()}_priority_score" for run in RUN_ORDER],
        "priority_class": [f"{run.lower()}_class" for run in RUN_ORDER],
    }

    class_rows = []
    freq_rows = []
    cap_rows = []
    score_rows = []
    consensus_rows = []
    robust_rows = []

    score_rankings = {}
    for run in RUN_ORDER:
        score_rankings[run] = (
            merged[["grid_id", f"{run.lower()}_priority_score"]]
            .dropna()
            .sort_values([f"{run.lower()}_priority_score", "grid_id"], ascending=[False, True])
            .reset_index(drop=True)
        )

    topk_sets = {
        run: {
            k: set(score_rankings[run].head(k)["grid_id"])
            for k in [10, 20, 50]
        }
        for run in RUN_ORDER
    }

    for _, row in merged.iterrows():
        grid_id = row["grid_id"]
        lon = row["longitude"]
        lat = row["latitude"]

        classes = [row.get(f"{run.lower()}_class", np.nan) for run in RUN_ORDER]
        classes_clean = [c for c in classes if pd.notna(c)]
        counts = pd.Series(classes_clean).value_counts() if classes_clean else pd.Series(dtype=int)
        dominant_class = counts.index[0] if not counts.empty else np.nan
        dominant_class_frequency = counts.iloc[0] / len(RUN_ORDER) if not counts.empty else np.nan
        class_consistency = dominant_class_frequency if not counts.empty else np.nan

        class_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": lon,
                "latitude": lat,
                "v1_class": row.get("v1_class"),
                "v2_class": row.get("v2_class"),
                "v3_class": row.get("v3_class"),
                "v4_class": row.get("v4_class"),
                "dominant_class": dominant_class,
                "dominant_class_frequency": dominant_class_frequency,
                "class_consistency": class_consistency,
            }
        )

        freq_vals = pd.to_numeric(row[run_cols["frequency"]], errors="coerce").to_numpy(dtype=float)
        freq_mean = np.nanmean(freq_vals)
        freq_std = np.nanstd(freq_vals)
        freq_min = np.nanmin(freq_vals)
        freq_max = np.nanmax(freq_vals)
        freq_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": lon,
                "latitude": lat,
                "v1_frequency": row.get("v1_frequency"),
                "v2_frequency": row.get("v2_frequency"),
                "v3_frequency": row.get("v3_frequency"),
                "v4_frequency": row.get("v4_frequency"),
                "mean_frequency": freq_mean,
                "std_frequency": freq_std,
                "min_frequency": freq_min,
                "max_frequency": freq_max,
                "frequency_range": freq_max - freq_min,
                "frequency_mean_absolute_deviation": float(np.nanmean(np.abs(freq_vals - freq_mean))),
                "cv_frequency": _safe_cv(freq_std, freq_mean),
            }
        )

        cap_vals = pd.to_numeric(row[run_cols["capacity"]], errors="coerce").to_numpy(dtype=float)
        cap_mean = np.nanmean(cap_vals)
        cap_std = np.nanstd(cap_vals)
        cap_min = np.nanmin(cap_vals)
        cap_max = np.nanmax(cap_vals)
        cap_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": lon,
                "latitude": lat,
                "v1_capacity": row.get("v1_capacity"),
                "v2_capacity": row.get("v2_capacity"),
                "v3_capacity": row.get("v3_capacity"),
                "v4_capacity": row.get("v4_capacity"),
                "mean_capacity": cap_mean,
                "std_capacity": cap_std,
                "min_capacity": cap_min,
                "max_capacity": cap_max,
                "capacity_range": cap_max - cap_min,
                "capacity_cv": _safe_cv(cap_std, cap_mean),
                "relative_capacity_range": np.nan if math.isclose(cap_mean, 0.0, abs_tol=1e-12) else (cap_max - cap_min) / cap_mean,
            }
        )

        score_vals = pd.to_numeric(row[run_cols["priority_score"]], errors="coerce").to_numpy(dtype=float)
        score_mean = np.nanmean(score_vals)
        score_std = np.nanstd(score_vals)
        score_min = np.nanmin(score_vals)
        score_max = np.nanmax(score_vals)
        score_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": lon,
                "latitude": lat,
                "v1_priority_score": row.get("v1_priority_score"),
                "v2_priority_score": row.get("v2_priority_score"),
                "v3_priority_score": row.get("v3_priority_score"),
                "v4_priority_score": row.get("v4_priority_score"),
                "mean_priority_score": score_mean,
                "std_priority_score": score_std,
                "min_priority_score": score_min,
                "max_priority_score": score_max,
                "priority_score_range": score_max - score_min,
                "priority_score_cv": _safe_cv(score_std, score_mean),
            }
        )

        top10_count = sum(grid_id in topk_sets[run][10] for run in RUN_ORDER)
        top20_count = sum(grid_id in topk_sets[run][20] for run in RUN_ORDER)
        top50_count = sum(grid_id in topk_sets[run][50] for run in RUN_ORDER)
        consensus_rows.append(
            {
                "technology": tech,
                "grid_id": grid_id,
                "longitude": lon,
                "latitude": lat,
                "top10_occurrence_count": top10_count,
                "top10_consensus": top10_count / 4.0,
                "top20_occurrence_count": top20_count,
                "top20_consensus": top20_count / 4.0,
                "top50_occurrence_count": top50_count,
                "top50_consensus": top50_count / 4.0,
                "mean_priority_score": score_mean,
            }
        )

    class_df = pd.DataFrame(class_rows)
    freq_df = pd.DataFrame(freq_rows)
    cap_df = pd.DataFrame(cap_rows)
    score_df = pd.DataFrame(score_rows)
    consensus_df = pd.DataFrame(consensus_rows).sort_values(
        ["top20_consensus", "mean_priority_score"], ascending=[False, False]
    )

    merged_all = (
        class_df.merge(freq_df, on=["technology", "grid_id", "longitude", "latitude"])
        .merge(cap_df, on=["technology", "grid_id", "longitude", "latitude"])
        .merge(score_df, on=["technology", "grid_id", "longitude", "latitude"])
        .merge(consensus_df.drop(columns=["mean_priority_score"]), on=["technology", "grid_id", "longitude", "latitude"])
    )

    for _, row in merged_all.iterrows():
        robust_rows.append(
            {
                "technology": tech,
                "grid_id": row["grid_id"],
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "mean_priority_score": row["mean_priority_score"],
                "std_priority_score": row["std_priority_score"],
                "mean_frequency": row["mean_frequency"],
                "std_frequency": row["std_frequency"],
                "mean_capacity": row["mean_capacity"],
                "std_capacity": row["std_capacity"],
                "top20_occurrence_count": row["top20_occurrence_count"],
                "top20_consensus": row["top20_consensus"],
                "class_consistency": row["class_consistency"],
                "dominant_class": row["dominant_class"],
                "robust_priority_class": _priority_class_from_metrics(
                    row["top20_consensus"], row["class_consistency"], row["mean_frequency"]
                ),
            }
        )
    robust_df = pd.DataFrame(robust_rows)
    return class_df, freq_df, cap_df, score_df, consensus_df, robust_df


def build_priority_lists(robust_df, cap_df, freq_df, score_df, class_df, tech):
    merged = (
        robust_df.merge(cap_df, on=["technology", "grid_id", "longitude", "latitude", "mean_capacity", "std_capacity"])
        .merge(freq_df, on=["technology", "grid_id", "longitude", "latitude", "mean_frequency", "std_frequency"])
        .merge(score_df, on=["technology", "grid_id", "longitude", "latitude", "mean_priority_score", "std_priority_score"])
    )

    robust = merged[merged["robust_priority_class"].isin(["R1", "R2"])].copy()
    robust = robust.sort_values(
        ["top20_consensus", "mean_priority_score", "mean_frequency"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    robust["rank"] = np.arange(1, len(robust) + 1)
    robust_top20 = robust.head(20).copy()

    sensitive = merged[merged["robust_priority_class"] == "S"].copy()
    sensitive = sensitive.sort_values("mean_priority_score", ascending=False).reset_index(drop=True)

    return robust_top20, sensitive


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
                "n_total_grids": tech_class["grid_id"].nunique(),
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


def save_latex_table(df, path, columns=None, float_fmt="{:.3f}"):
    use_df = df.copy()
    if columns is not None:
        use_df = use_df[columns].copy()

    lines = [r"\begin{tabular}{" + "l" * len(use_df.columns) + "}", r"\toprule"]
    lines.append(" & ".join(_latex_escape(c) for c in use_df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in use_df.iterrows():
        vals = []
        for val in row:
            if pd.isna(val):
                vals.append("")
            elif isinstance(val, (float, np.floating)):
                vals.append(float_fmt.format(float(val)))
            else:
                vals.append(_latex_escape(val))
        lines.append(" & ".join(vals) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_topk_overlap(overlap_summary, fig_dir):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for tech, color in [("wind", "#1f5a91"), ("solar", "#c67c00")]:
        sub = overlap_summary[overlap_summary["technology"] == tech].sort_values("k")
        ax.plot(sub["k"], sub["mean_pairwise_overlap"], marker="o", linewidth=2.2, label=tech.title(), color=color)
    ax.set_xticks(TOPK_LIST)
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Mean Pairwise Overlap")
    ax.set_ylim(0, 1.05)
    ax.set_title("Top-K Overlap Comparison")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.savefig(fig_dir / "fig_topk_overlap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rank_heatmap(heatmap, tech, fig_dir):
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    arr = heatmap.loc[RUN_ORDER, RUN_ORDER].to_numpy(dtype=float)
    im = ax.imshow(arr, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(RUN_ORDER)))
    ax.set_yticks(np.arange(len(RUN_ORDER)))
    ax.set_xticklabels(RUN_ORDER)
    ax.set_yticklabels(RUN_ORDER)
    ax.set_title(f"{tech.title()} Spearman Rank Correlation")
    for i in range(len(RUN_ORDER)):
        for j in range(len(RUN_ORDER)):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman rho")
    fig.savefig(fig_dir / f"fig_{tech}_rank_correlation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_consensus_map(df, tech, fig_dir):
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["top20_occurrence_count"],
        cmap="YlOrRd",
        s=40 + 90 * df["top20_consensus"],
        vmin=0,
        vmax=4,
        edgecolors="black",
        linewidths=0.25,
        alpha=0.9,
    )
    ax.set_title(f"{tech.title()} Top-20 Consensus Map")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Top-20 occurrence count")
    fig.savefig(fig_dir / f"fig_{tech}_top20_consensus_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_stability(robust_top, cap_df, tech, fig_dir):
    use_ids = robust_top["grid_id"].tolist()
    sub = cap_df[(cap_df["technology"] == tech) & (cap_df["grid_id"].isin(use_ids))].copy()
    sub = sub.set_index("grid_id").loc[use_ids].reset_index()

    fig, ax = plt.subplots(figsize=(max(10, len(use_ids) * 0.5), 5.8))
    x = np.arange(len(use_ids))
    offsets = [-0.24, -0.08, 0.08, 0.24]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for offset, run, color in zip(offsets, RUN_ORDER, colors):
        ax.scatter(
            x + offset,
            sub[f"{run.lower()}_capacity"],
            color=color,
            label=run,
            s=45,
            alpha=0.9,
        )
    ax.plot(x, sub["mean_capacity"], color="black", linewidth=2.0, marker="D", markersize=4, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(use_ids, rotation=75, ha="right")
    ax.set_ylabel("Recommended Capacity (MW)")
    ax.set_title(f"{tech.title()} Capacity Stability of Robust Priority Grids")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncol=5, fontsize=9)
    fig.savefig(fig_dir / f"fig_{tech}_capacity_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_stability(robust_top, freq_df, tech, fig_dir):
    use_ids = robust_top["grid_id"].tolist()
    sub = freq_df[(freq_df["technology"] == tech) & (freq_df["grid_id"].isin(use_ids))].copy()
    sub = sub.set_index("grid_id").loc[use_ids].reset_index()

    fig, ax = plt.subplots(figsize=(max(10, len(use_ids) * 0.5), 5.8))
    x = np.arange(len(use_ids))
    offsets = [-0.24, -0.08, 0.08, 0.24]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for offset, run, color in zip(offsets, RUN_ORDER, colors):
        ax.scatter(
            x + offset,
            sub[f"{run.lower()}_frequency"],
            color=color,
            label=run,
            s=45,
            alpha=0.9,
        )
    ax.plot(x, sub["mean_frequency"], color="black", linewidth=2.0, marker="D", markersize=4, label="Mean")
    ax.set_xticks(x)
    ax.set_xticklabels(use_ids, rotation=75, ha="right")
    ax.set_ylabel("Development Frequency")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{tech.title()} Frequency Stability of Robust Priority Grids")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncol=5, fontsize=9)
    fig.savefig(fig_dir / f"fig_{tech}_frequency_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(
    summary_df,
    overlap_summary,
    corr_summary,
    robust_tops,
    sensitive_tops,
    report_dir,
    issues,
):
    lines = ["# Sampling Stability Analysis Report", ""]
    if issues:
        lines.append("## Data Warnings")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    for tech in TECHS:
        tech_summary = summary_df[summary_df["technology"] == tech].iloc[0]
        tech_overlap = overlap_summary[overlap_summary["technology"] == tech].set_index("k")
        tech_corr = corr_summary[corr_summary["technology"] == tech].iloc[0]
        robust_top = robust_tops[tech]
        sensitive_top = sensitive_tops[tech]

        lines.append(f"## {tech.title()}")
        lines.append(
            f"- Mean Top-10 overlap: {tech_overlap.loc[10, 'mean_pairwise_overlap']:.3f}"
        )
        lines.append(
            f"- Mean Top-20 overlap: {tech_overlap.loc[20, 'mean_pairwise_overlap']:.3f}"
        )
        lines.append(
            f"- Mean Top-50 overlap: {tech_overlap.loc[50, 'mean_pairwise_overlap']:.3f}"
        )
        lines.append(f"- Mean Spearman rho: {tech_corr['mean_spearman']:.3f}")
        lines.append(f"- Minimum Spearman rho: {tech_corr['min_spearman']:.3f}")
        lines.append(f"- Number of R1 grids: {int(tech_summary['n_R1'])}")
        lines.append(f"- Number of R2 grids: {int(tech_summary['n_R2'])}")
        lines.append(f"- Number of scenario-sensitive grids: {int(tech_summary['n_S'])}")

        top_ids = ", ".join(robust_top["grid_id"].head(10).tolist())
        sens_ids = ", ".join(sensitive_top["grid_id"].head(10).tolist())
        lines.append(f"- Most stable top grids: {top_ids if top_ids else 'None'}")
        lines.append(f"- Most scenario-sensitive grids: {sens_ids if sens_ids else 'None'}")
        lines.append("")

        descriptor = "strong" if tech_corr["mean_spearman"] >= 0.8 else "moderate" if tech_corr["mean_spearman"] >= 0.6 else "limited"
        lines.append(
            textwrap.fill(
                f"For {tech}, the four independent 1000-path samples indicate {descriptor} sampling stability in grid-level priority rankings. "
                f"The mean pairwise Top-20 overlap is {tech_overlap.loc[20, 'mean_pairwise_overlap']:.3f}, while the mean Spearman rank correlation is "
                f"{tech_corr['mean_spearman']:.3f}. These statistics should be interpreted as evidence of robustness to stochastic scenario sampling, "
                f"rather than a proof of worst-case robustness.",
                width=100,
            )
        )
        lines.append("")

    (report_dir / "robustness_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")

    draft_lines = [
        r"\section{Sampling Stability Across Independent Scenario Samples}",
        "",
    ]
    for tech in TECHS:
        tech_summary = summary_df[summary_df["technology"] == tech].iloc[0]
        draft_lines.append(
            textwrap.fill(
                f"For {tech} development, the identified spatial priority pattern remained stable across four independent 1000-path samples. "
                f"The mean Top-20 overlap was {tech_summary['top20_mean_overlap']:.3f}, and the mean Spearman rank correlation was "
                f"{tech_summary['mean_spearman']:.3f}. A total of {int(tech_summary['n_R1'])} grids were classified as highly stable priority sites "
                f"(R1), whereas {int(tech_summary['n_S'])} grids were classified as scenario-sensitive. These results indicate sampling stability "
                f"of the priority pattern without implying guaranteed worst-case robustness.",
                width=100,
            )
        )
        draft_lines.append("")
    (report_dir / "results_sampling_stability_draft.tex").write_text("\n".join(draft_lines), encoding="utf-8")


def print_executive_summary(summary_df, robust_tops, sensitive_tops):
    print("=" * 47)
    print("SCEQ SAMPLING STABILITY ANALYSIS")
    print("=" * 47)
    for tech in TECHS:
        row = summary_df[summary_df["technology"] == tech].iloc[0]
        print(tech.upper())
        print(f"Top-10 mean overlap: {row['top10_mean_overlap']:.3f}")
        print(f"Top-20 mean overlap: {row['top20_mean_overlap']:.3f}")
        print(f"Top-50 mean overlap: {row['top50_mean_overlap']:.3f}")
        print(f"Mean Spearman rho: {row['mean_spearman']:.3f}")
        print(f"Min Spearman rho: {row['min_spearman']:.3f}")
        print(f"Number of R1 grids: {int(row['n_R1'])}")
        print(f"Number of R2 grids: {int(row['n_R2'])}")
        print(f"Number of scenario-sensitive grids: {int(row['n_S'])}")
        print(
            "Most stable grids: "
            + ", ".join(robust_tops[tech]["grid_id"].head(5).tolist())
        )
        print(
            "Most scenario-sensitive grids: "
            + ", ".join(sensitive_tops[tech]["grid_id"].head(5).tolist())
        )
        print("")
    print("=" * 47)


def main():
    warnings.filterwarnings("ignore")
    dirs = ensure_dirs()

    run_data_list = [load_run_csv(run, RUN_FILES[run]) for run in RUN_ORDER]
    issues = validate_runs(run_data_list)
    save_data_quality_report(run_data_list, issues, dirs["csv"], dirs["report"])

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
        topk_df, topk_summary_df = compute_topk_overlap(run_lookup, tech)
        corr_df, corr_summary_df, heatmap = compute_rank_correlation(run_lookup, tech)
        merged = build_merged_grid_frame(run_lookup)
        class_df, freq_df, cap_df, score_df, consensus_df, robust_df = compute_grid_stability_tables(merged, tech)
        robust_top20, sensitive_df = build_priority_lists(robust_df, cap_df, freq_df, score_df, class_df, tech)

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

        plot_rank_heatmap(heatmap, tech, dirs["figures"])
        plot_consensus_map(consensus_df, tech, dirs["figures"])
        plot_capacity_stability(robust_top20, cap_df, tech, dirs["figures"])
        plot_frequency_stability(robust_top20, freq_df, tech, dirs["figures"])

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

    overlap_df.to_csv(dirs["csv"] / "robustness_topk_overlap.csv", index=False)
    overlap_summary_df.to_csv(dirs["csv"] / "robustness_topk_overlap_summary.csv", index=False)
    corr_df.to_csv(dirs["csv"] / "robustness_rank_correlation.csv", index=False)
    class_df.to_csv(dirs["csv"] / "priority_class_stability.csv", index=False)
    freq_df.to_csv(dirs["csv"] / "frequency_stability.csv", index=False)
    cap_df.to_csv(dirs["csv"] / "capacity_stability.csv", index=False)
    score_df.to_csv(dirs["csv"] / "priority_score_stability.csv", index=False)
    consensus_df.to_csv(dirs["csv"] / "grid_priority_consensus.csv", index=False)
    robust_df.to_csv(dirs["csv"] / "robust_priority_classification.csv", index=False)
    summary_df.to_csv(dirs["csv"] / "robustness_summary.csv", index=False)

    robust_tops["wind"].to_csv(dirs["csv"] / "robust_wind_priority_top20.csv", index=False)
    robust_tops["solar"].to_csv(dirs["csv"] / "robust_solar_priority_top20.csv", index=False)
    sensitive_tops["wind"].to_csv(dirs["csv"] / "scenario_sensitive_wind_grids.csv", index=False)
    sensitive_tops["solar"].to_csv(dirs["csv"] / "scenario_sensitive_solar_grids.csv", index=False)

    plot_topk_overlap(overlap_summary_df, dirs["figures"])

    save_latex_table(
        summary_df,
        dirs["latex"] / "table_robustness_summary.tex",
        columns=[
            "technology",
            "n_R1",
            "n_R2",
            "n_S",
            "top20_mean_overlap",
            "mean_spearman",
            "mean_class_consistency",
        ],
    )
    save_latex_table(
        robust_tops["wind"],
        dirs["latex"] / "table_wind_robust_priority.tex",
        columns=[
            "rank",
            "grid_id",
            "top20_consensus",
            "mean_priority_score",
            "mean_frequency",
            "mean_capacity",
            "dominant_class",
            "class_consistency",
            "robust_priority_class",
        ],
    )
    save_latex_table(
        robust_tops["solar"],
        dirs["latex"] / "table_solar_robust_priority.tex",
        columns=[
            "rank",
            "grid_id",
            "top20_consensus",
            "mean_priority_score",
            "mean_frequency",
            "mean_capacity",
            "dominant_class",
            "class_consistency",
            "robust_priority_class",
        ],
    )
    save_latex_table(
        sensitive_tops["wind"].head(20),
        dirs["latex"] / "table_scenario_sensitive_wind.tex",
        columns=[
            "grid_id",
            "mean_priority_score",
            "std_priority_score",
            "mean_frequency",
            "std_frequency",
            "mean_capacity",
            "std_capacity",
            "top20_consensus",
            "class_consistency",
        ],
    )
    save_latex_table(
        sensitive_tops["solar"].head(20),
        dirs["latex"] / "table_scenario_sensitive_solar.tex",
        columns=[
            "grid_id",
            "mean_priority_score",
            "std_priority_score",
            "mean_frequency",
            "std_frequency",
            "mean_capacity",
            "std_capacity",
            "top20_consensus",
            "class_consistency",
        ],
    )

    write_markdown_report(
        summary_df,
        overlap_summary_df,
        corr_summary_df,
        robust_tops,
        sensitive_tops,
        dirs["report"],
        issues,
    )

    print_executive_summary(summary_df, robust_tops, sensitive_tops)
    print("Input files used:")
    for run, path in RUN_FILES.items():
        print(f"- {run}: {path}")
    print(f"Output root: {OUTPUT_ROOT}")
    if issues:
        print("Data warnings detected:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("No data quality warnings detected.")


if __name__ == "__main__":
    main()


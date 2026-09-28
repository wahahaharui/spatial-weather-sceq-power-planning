import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import linregress, t


# ============================================================
# Standalone configuration
# This script is self-contained and only relies on relative paths.
# ============================================================
FINAL_PERIOD = 2040
SCENARIO_PREFIX = "path_"

OUTPUT_DIRNAME = "outputs"
OUTPUT_FIGURE = "technology_interaction_scatter_triptych_standalone.png"
OUTPUT_DETAIL_CSV = "scenario_capacity_detail_standalone.csv"

FONT_SIZE = 11
FIGURE_DPI = 300

TECH_COLORS = {
    "Wind": "#1F77B4",
    "Solar": "#F2A104",
    "Storage": "#2CA58D",
}


def module_root() -> Path:
    return Path(__file__).resolve().parent


def sceq_root() -> Path:
    return module_root().parent


def project_output_root() -> Path:
    return sceq_root() / "output"


def local_output_dir() -> Path:
    out = module_root() / OUTPUT_DIRNAME
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.dpi": FIGURE_DPI,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#404040",
            "axes.linewidth": 0.8,
            "grid.color": "#D7D7D7",
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
        }
    )


def fmt_mw(value, _pos=None):
    return f"{value:,.0f}"


def fmt_p_value(value):
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def load_checkpoint_paths():
    checkpoint_path = project_output_root() / "checkpoint.json"
    with checkpoint_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("all_results", [])


def extract_capacity_row(path_id: int):
    scenario_name = f"{SCENARIO_PREFIX}{path_id:03d}"
    gen_cap_path = (
        project_output_root()
        / "05_work_dirs"
        / scenario_name
        / f"step_{FINAL_PERIOD}"
        / "gen_cap.csv"
    )
    if not gen_cap_path.exists():
        raise FileNotFoundError(f"Missing capacity file: {gen_cap_path}")

    df = pd.read_csv(gen_cap_path)
    grouped = df.groupby("gen_energy_source", dropna=False)["GenCapacity"].sum().to_dict()
    return {
        "scenario_id": path_id,
        "pathway_name": scenario_name,
        "wind_capacity": float(grouped.get("wind", 0.0)),
        "solar_capacity": float(grouped.get("solar", 0.0)),
        "storage_capacity": float(grouped.get("storage", 0.0)),
    }


def build_capacity_table():
    results = load_checkpoint_paths()
    rows = []
    for result in results:
        path_id = int(result["path_id"])
        rows.append(extract_capacity_row(path_id))
    return pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)


def regression_with_ci(x: np.ndarray, y: np.ndarray, confidence: float = 0.95):
    result = linregress(x, y)
    n = len(x)
    x_mean = np.mean(x)
    ssx = np.sum((x - x_mean) ** 2)

    x_line = np.linspace(np.min(x), np.max(x), 200)
    y_line = result.intercept + result.slope * x_line

    residuals = y - (result.intercept + result.slope * x)
    s_err = np.sqrt(np.sum(residuals ** 2) / (n - 2))
    t_crit = t.ppf((1 + confidence) / 2.0, df=n - 2)
    se_mean = s_err * np.sqrt((1 / n) + ((x_line - x_mean) ** 2) / ssx)

    lower = y_line - t_crit * se_mean
    upper = y_line + t_crit * se_mean
    return result, x_line, y_line, lower, upper


def add_panel(ax, df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, color: str):
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    result, x_line, y_line, lower, upper = regression_with_ci(x, y)

    ax.scatter(
        x,
        y,
        s=42,
        color=color,
        alpha=0.72,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )
    ax.fill_between(
        x_line,
        lower,
        upper,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.plot(x_line, y_line, color="#303030", linewidth=1.8, zorder=4)

    ax.set_title(f"{x_label} vs {y_label}")
    ax.set_xlabel(f"{x_label} Capacity (MW)")
    ax.set_ylabel(f"{y_label} Capacity (MW)")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_mw))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_mw))
    ax.text(
        0.03,
        0.97,
        f"r = {result.rvalue:.2f}\np = {fmt_p_value(result.pvalue)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZE,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "#CFCFCF",
            "alpha": 0.95,
        },
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
    ax.set_axisbelow(True)


def plot_triptych(detail_df: pd.DataFrame, output_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9), gridspec_kw={"wspace": 0.26})

    add_panel(
        axes[0],
        detail_df,
        "wind_capacity",
        "solar_capacity",
        "Wind",
        "Solar",
        TECH_COLORS["Wind"],
    )
    add_panel(
        axes[1],
        detail_df,
        "solar_capacity",
        "storage_capacity",
        "Solar",
        "Storage",
        TECH_COLORS["Solar"],
    )
    add_panel(
        axes[2],
        detail_df,
        "wind_capacity",
        "storage_capacity",
        "Wind",
        "Storage",
        TECH_COLORS["Storage"],
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.15)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    setup_style()
    out_dir = local_output_dir()

    detail_df = build_capacity_table()
    detail_df.to_csv(out_dir / OUTPUT_DETAIL_CSV, index=False, encoding="utf-8-sig")

    plot_triptych(detail_df, out_dir / OUTPUT_FIGURE)

    print(f"Saved: {out_dir / OUTPUT_DETAIL_CSV}")
    print(f"Saved: {out_dir / OUTPUT_FIGURE}")


if __name__ == "__main__":
    main()

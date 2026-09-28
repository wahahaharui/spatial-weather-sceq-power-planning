from pathlib import Path


FINAL_PERIOD = 2040
SCENARIO_PREFIX = "path_"

OUTPUT_DIRNAME = "outputs"
LOG_DIRNAME = "logs"

DETAIL_CSV = "scenario_capacity_detail.csv"
CORRELATION_CSV = "technology_correlation_matrix.csv"
REGRESSION_CSV = "regression_summary.csv"
FIGURE_FILE = "technology_interaction_analysis.png"
FIGURE_TRIPTYCH = "technology_interaction_scatter_triptych.png"
FIGURE_HEATMAP_SCATTER = "technology_interaction_heatmap_plus_solar_storage.png"
FIGURE_COMBINED = "technology_interaction_combined_scatter.png"
RUN_LOG = "run_log.txt"

TECHNOLOGIES = ["Wind", "Solar", "Storage"]
TECH_COLUMN_MAP = {
    "Wind": "wind_capacity",
    "Solar": "solar_capacity",
    "Storage": "storage_capacity",
}

TECH_COLORS = {
    "Wind": "#1F77B4",
    "Solar": "#F2A104",
    "Storage": "#2CA58D",
}

BASE_FONT_SIZE = 11
TITLE_FONT_SIZE = 11
LABEL_FONT_SIZE = 11
TICK_FONT_SIZE = 11
LEGEND_FONT_SIZE = 11
ANNOTATION_FONT_SIZE = 11
FIGURE_DPI = 300


def module_root() -> Path:
    return Path(__file__).resolve().parent


def sceq_root() -> Path:
    return module_root().parent


def output_root() -> Path:
    return sceq_root() / "output"

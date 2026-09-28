"""Lightweight validation for the packaged SCEQ repository.

This script intentionally avoids weather generation and optimization. It checks
that the code package is self-contained at the source-code level and that known
large external data are only referenced through the central configuration.
"""
from __future__ import annotations

import ast
import compileall
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CODE_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(CODE_ROOT))

FORBIDDEN = [
    "D:\\switch_lvliang_project",
    "D:\\wjr",
    "D:\\claude_switch_lvliang",
    "D:\\claude\\_switch",
]
REQUIRED_FILES = [
    "scripts/02_run_sceq.py",
    "scripts/generate_scenarios.py",
    "scripts/04_v2_plots.py",
    "scripts/plot_0714.py",
    "scripts/06_plot.py",
    "scripts/robustness_sampling_analysis.py",
    "mechanism_experiment/run_spatial_homogenization_experiment.py",
    "mechanism_experiment/config.template.json",
    "algorithm_comparison/generate_comparison.py",
    "algorithm_comparison/generate_charts.py",
    "09_technology_interaction_analysis/run_scatter_triptych_standalone.py",
    "src/sceq_lib/config.py",
    "src/sceq_lib/scenario_generator.py",
    "src/sceq_lib/benders_adapter.py",
    "benders_fixed_v6/data_loader.py",
    "benders_fixed_v6/benders_persistent_v7.py",
    "data/inputs_data_add_storage/gen_info.csv",
    "data/inputs_data_add_storage/gen_build_costs.csv",
    "data/static/Lvliang_10km_Features_with_Shear.csv",
]


def scan_forbidden_paths() -> list[str]:
    hits = []
    for path in list((CODE_ROOT / "scripts").glob("*.py")) + list((CODE_ROOT / "src").rglob("*.py")) + list((CODE_ROOT / "mechanism_experiment").glob("*.py")) + list((CODE_ROOT / "09_technology_interaction_analysis").glob("*.py")):
        text = path.read_text(encoding="utf-8-sig", errors="ignore")
        for token in FORBIDDEN:
            if token in text:
                hits.append(f"{path.relative_to(CODE_ROOT)}: {token}")
    return hits


def scan_local_imports() -> list[str]:
    missing = []
    local_roots = {"sceq_lib", "benders_fixed_v6"}
    for path in CODE_ROOT.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig", errors="ignore"))
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module.split(".")[0]
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in local_roots:
                        mod = root
            if mod == "sceq_lib" and not (SRC_ROOT / "sceq_lib").exists():
                missing.append(f"{path.relative_to(CODE_ROOT)} imports sceq_lib")
            if mod == "benders_fixed_v6" and not (CODE_ROOT / "benders_fixed_v6").exists():
                missing.append(f"{path.relative_to(CODE_ROOT)} imports benders_fixed_v6")
    return missing


def main() -> int:
    print(f"CODE_ROOT={CODE_ROOT}")
    missing_files = [p for p in REQUIRED_FILES if not (CODE_ROOT / p).exists()]
    if missing_files:
        print("Missing required package files:")
        for item in missing_files:
            print(f"  - {item}")
        return 1

    if not compileall.compile_dir(str(CODE_ROOT), quiet=1):
        print("Python syntax compilation failed.")
        return 1

    forbidden_hits = scan_forbidden_paths()
    if forbidden_hits:
        print("Forbidden absolute paths remain in Python files:")
        for hit in forbidden_hits:
            print(f"  - {hit}")
        return 1

    import_errors = scan_local_imports()
    if import_errors:
        print("Missing local import roots:")
        for item in import_errors:
            print(f"  - {item}")
        return 1

    from sceq_lib.config import cfg_v2 as cfg

    print("Central configuration loaded.")
    print(f"  DATA_ROOT={cfg.DATA_ROOT}")
    print(f"  OUTPUT_DIR={cfg.OUTPUT_DIR}")
    print(f"  INPUTS_TEMPLATE exists={cfg.INPUTS_TEMPLATE.exists()}")
    print(f"  STATIC_CSV exists={cfg.STATIC_CSV.exists()}")
    print(f"  BENDERS_V6_DIR exists={cfg.BENDERS_V6_DIR.exists()}")
    print("Smoke check passed. External models/data may still need to be placed before full runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


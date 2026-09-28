# CODE_CHANGES_FOR_RELEASE

This file records the intentional changes made when preparing the public GitHub reproduction package from the original research workspace. The goal of these changes is portability and publication hygiene. They do not change the SCEQ model formulation, stochastic scenario logic, Benders decomposition algorithm, solver tolerances, planning periods, resource conversion equations, or paper-analysis calculations.

## Scope of release changes

1. Centralized local paths

- Original machine-specific workspace paths were removed from Python runtime code and replaced with repository-relative paths plus documented environment-variable overrides.
- Public code now resolves paths relative to the repository root, with optional environment-variable overrides in `src/sceq_lib/config.py`.
- Original numerical parameters were retained, including `N_PATHS=200`, `PERIODS=[2025, 2030, 2035, 2040]`, `P_EXTREME=0.10`, `BENDERS_MAX_ITER=200`, and `BENDERS_GAP_TOL=0.005`.

2. Packaged import layout

- Scripts now prepend `code/src` and `code/` to `sys.path` so they can import `sceq_lib` and `benders_fixed_v6` after the folder is moved to another computer.
- This changes import resolution only; model logic and calculation code are unchanged.

3. SCEQ runner portability

- The original `02_run_sceq.py` copied weather paths from an old local `sceq_planning/output/01_weather_paths` directory when paths were missing.
- The public `scripts/02_run_sceq.py` instead generates missing weather paths with `ScenarioGenerator`, using the same packaged configuration and model parameters.
- This removes dependency on a historical local result directory and supports true from-scratch reproduction.

4. Plotting paths

- `plot_0714.py` and `06_plot.py` now use `cfg.BOUNDARY_SHP_PATH` and `cfg.CLCD_TIF_PATH` instead of hard-coded local paths.
- The plotting algorithms and plotted quantities were not changed.

5. Robustness and mechanism paths

- `robustness_sampling_analysis.py` now reads multi-run outputs from `output_runs/` or environment variables such as `SCEQ_MULTI_RUN_ROOT`.
- `mechanism_experiment/run_spatial_homogenization_experiment.py` now imports from packaged `src/sceq_lib`.
- The analysis formulas, selected metrics, and mechanism-experiment calculations were not changed.

6. `plot_0714.py` f-string repair

- One target-package line contained a syntax-damaged f-string fragment around the P5-P95 label: the expression for `p95_vals[-1]` was missing the opening `{` in the copied text.
- It was repaired to the intended formatting expression `{p95_vals[-1]:,.0f}`.
- This is a syntax/text repair for a legend label only. It does not affect numerical data, plotted series, model logic, or algorithm results.

7. Release metadata

- Added `README.md`, `requirements.txt`, `environment.yml`, `.gitignore`, `data_required_manifest.csv`, and `scripts/smoke_check.py`.
- These files document installation, external data placement, excluded large files, and lightweight validation.

## Unchanged algorithmic components

- `sceq_lib/scenario_generator.py`: scenario generation logic, seeds, model usage, CF calculation calls, and load-prediction calls are preserved.
- `sceq_lib/cf_calculator.py`: wind/solar CF formulas are preserved.
- `sceq_lib/load_predictor.py`: TensorFlow demand-model inference and temperature correction logic are preserved.
- `sceq_lib/ce_baseline.py`: CE baseline aggregation logic is preserved.
- `sceq_lib/benders_adapter.py`: data preparation for Benders and rolling-step interface are preserved except for packaged path resolution through config.
- `benders_fixed_v6/data_loader.py` and `benders_fixed_v6/benders_persistent_v7.py`: copied from the actual runtime chain used by the original SCEQ adapter.

## Validation performed

- `python scripts/smoke_check.py` passed.
- `python -m compileall -q .` passed.
- Target Python files were scanned for old local absolute paths.
- README command targets were checked for existence.
- Low-cost mechanism-experiment `--help` check passed.


# Technology Interaction Analysis

This module analyzes coordination and substitution relationships among wind,
solar, and storage capacities across 200 SCEQ planning pathways.

## Inputs

- `../output/checkpoint.json`
- `../output/05_work_dirs/path_xxx/step_2040/gen_cap.csv`

## Outputs

- `outputs/scenario_capacity_detail.csv`
- `outputs/technology_correlation_matrix.csv`
- `outputs/regression_summary.csv`
- `outputs/technology_interaction_analysis.png`

## Run

```bash
python 09_technology_interaction_analysis/run_technology_interaction_analysis.py
```

## Notes

- This module does not modify historical code or raw outputs.
- All new artifacts are written under `09_technology_interaction_analysis`.

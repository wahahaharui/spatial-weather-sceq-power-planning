# Spatial Resource Homogenization Counterfactual Experiment

这个目录现在是一个完全自包含的实验入口，只改写 `mechanism_experiment` 自己，不再复制或覆盖 `sceq_planning_v2\sceq_lib` 里的任何文件。

## 核心原则

- 路径选择直接读取各组基线结果的 `checkpoint.json` 和 `01_weather_paths`
- 反事实重跑直接以各组 `output*\\05_work_dirs\\path_xxx\\step_yyyy` 作为逐期模板
- 不再使用 `inputs_data_add_storage`
- 不再检查天气生成模型、静态特征 CSV、外部数据集目录
- 不主动去别的文件夹找源码或求解器路径
- 真正求解时只尝试从当前 Python 环境 `import benders_fixed_v6`

## 需要的基线结果结构

每个 `baseline_runs` 对应目录都需要至少包含：

- `checkpoint.json`
- `01_weather_paths/path_xxx/period_YYYY.npz`
- `02_ce_baseline/*.npy`
- `04_summary/02_grid_robustness_data.csv`
- `05_work_dirs/path_xxx/step_YYYY/*.csv`

## 一行全流程命令

在 `sceq_planning_v2` 目录下运行：

```powershell
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json run-all --clean
```

如果只想先检查而不启动求解器：

```powershell
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json run-all --clean --dry-run
```

## 分步命令

```powershell
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json select-paths
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json build-weather --variant wind_homogenized
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json build-weather --variant solar_homogenized
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json run-sceq --variant wind_homogenized
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json run-sceq --variant solar_homogenized
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json analyze --experiment baseline
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json analyze --experiment wind_homogenized
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json analyze --experiment solar_homogenized
python mechanism_experiment\run_spatial_homogenization_experiment.py --config mechanism_experiment\config.template.json summarize
```

## 本机与另一台设备

- 本机先用 `config.local_smoke.json` 做小规模检查
- 另一台设备用 `config.template.json`
- 如果另一台设备的 Python 环境本来就能跑 SCEQ，这个脚本会直接复用那个环境里的 `benders_fixed_v6`

## 输出结构

```text
mechanism_experiment/
  selected_paths.csv
  baseline/
    results/
      source_runs.json
    robustness/
      csv/
  wind_homogenized/
    01_weather_paths/
    results/
      V1/
      V2/
      V3/
      V4/
    robustness/
      csv/
  solar_homogenized/
    01_weather_paths/
    results/
      V1/
      V2/
      V3/
      V4/
    robustness/
      csv/
  summary/
    path_selection_summary.csv
    mechanism_experiment_summary.csv
    mechanism_experiment_summary.txt
    validation_report.txt
```

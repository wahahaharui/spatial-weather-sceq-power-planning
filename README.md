# 城市级 SCEQ 规划代码包

本目录是论文复现实验的独立代码包。原始工程中的本机绝对路径已经集中改为相对路径和环境变量。

## 1. 项目结构

```text
code/
├─ README.md
├─ requirements.txt
├─ environment.yml
├─ data_required_manifest.csv
├─ configs/path_config.py
├─ scripts/
│  ├─ generate_scenarios.py
│  ├─ 02_run_sceq.py
│  ├─ 04_v2_plots.py
│  ├─ plot_0714.py
│  ├─ 06_plot.py
│  ├─ robustness_sampling_analysis.py
│  └─ smoke_check.py
├─ src/sceq_lib/
├─ benders_fixed_v6/
├─ mechanism_experiment/
├─ algorithm_comparison/
├─ 09_technology_interaction_analysis/
└─ data/
   ├─ inputs_data_add_storage/
   └─ static/
```

默认输出写入 `code/output/`。多组稳健性分析默认读取 `code/output_runs/output`、`output_v2`、`output_v3`、`output_v4`。

## 2. Python 和求解器要求

建议使用 Python 3.10。当前源环境中核对到的主要包版本见 `requirements.txt` 和 `environment.yml`。

Gurobi 是必需的：`benders_fixed_v6` 使用 `gurobipy` 求解 LP。请先安装 Gurobi 11.0 或兼容版本，并确认 license 可用：

```bash
gurobi_cl --version
python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

天气情景生成使用 PyTorch、torch-geometric 和 TensorFlow。若从头生成情景，建议使用 CUDA GPU；仅运行后处理绘图时不需要 GPU。

## 3. 安装环境

Conda 方式：

```bash
conda env create -f environment.yml
conda activate sceq-lvliang-repro
```

pip 方式：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# 然后按本机 CUDA/CPU 环境单独安装 torch 和 torch-geometric
```

注意：源环境使用 `torch==2.8.0+cu126` 和 `torch-geometric==2.6.1`。这两个包强依赖 CUDA/CPU 平台，未直接写入普通 pip 安装列表；请按 PyTorch 和 PyG 官方说明为本机选择匹配 wheel。`rasterio/geopandas` 在 Windows 上建议优先使用 conda-forge。

## 4. 外部数据和模型放置

小型静态输入已放入 `data/inputs_data_add_storage/` 和 `data/static/`。大模型和大数据由于数据量较大未放入github仓库

按默认配置，请将外部文件放成下面结构：

```text
code/data/
├─ models/
│  ├─ GNN_cGAN_Normal_Outputs_0502/checkpoints/gnn_cgan_normal_epoch_1300.pth
│  ├─ GNN_cGAN_Extreme_Outputs_0502/checkpoints/gnn_cgan_extreme_latest.pth
│  └─ demand_model/
│     ├─ MSE_weather_extreme_load_predictor_model.h5
│     ├─ MSE_weather_extreme_preprocessor.joblib
│     └─ MSE_weather_extreme_scaler_y.joblib
├─ Datasets_calibration_0428/Normal/*.npy
└─ external/
   ├─ CLCD_v01_2022_albert.tif
   └─ township_boundary/...
```

完整清单见 `data_required_manifest.csv`。如果外部数据放在其他位置，只需要设置环境变量，例如：

```bash
set SCEQ_DATA_ROOT=E:\sceq_data
set SCEQ_OUTPUT_DIR=E:\sceq_outputs\main_run
set SCEQ_MODEL_ROOT=E:\sceq_data\models
```

也可以单独覆盖某个文件：`SCEQ_NORMAL_MODEL_PATH`、`SCEQ_EXTREME_MODEL_PATH`、`SCEQ_DEMAND_MODEL_DIR`、`SCEQ_CLCD_TIF_PATH` 等。集中配置入口在 `src/sceq_lib/config.py`。

## 5. 轻量检查

复制代码包到本地后，先运行：

```bash
python scripts/smoke_check.py
```

该命令只检查代码结构、Python 语法、相对路径配置和关键小型输入文件，不会生成情景，也不会调用 Gurobi 做完整求解。

## 6. 情景生成

从 GNN 模型和负荷模型生成 `N_PATHS` 条天气路径：

```bash
python scripts/generate_scenarios.py
```

默认 `N_PATHS=200`。测试时可临时减少路径数：

```bash
set SCEQ_N_PATHS=2
python scripts/generate_scenarios.py
```

输出位置：`output/01_weather_paths/path_XXX/period_YYYY.npz`。

## 7. SCEQ 主求解

完整主流程：检查/生成天气路径、构建 certainty-equivalent baseline、运行滚动 SCEQ+Benders：

```bash
python scripts/02_run_sceq.py
```

主要输出：

```text
output/checkpoint.json
output/00_run_log.txt
output/02_ce_baseline/*.npy
output/04_summary/run_summary.json
output/05_work_dirs/path_XXX/step_YYYY/
```

完整运行计算量很大，取决于路径数、线程数、Gurobi license 和硬件。

## 8. 结果汇总和绘图

先生成基础汇总和图：

```bash
python scripts/04_v2_plots.py
```

再生成论文风格地图/路径图：

```bash
python scripts/plot_0714.py
python scripts/06_plot.py
```

`plot_0714.py` 和 `06_plot.py` 可使用 CLCD 土地利用底图；若 `SCEQ_CLCD_TIF_PATH` 指向的 tif 不存在，脚本会跳过该底图部分或不显示 CLCD 图层。

## 9. 风光储投资相关性分析

该脚本读取 `output/checkpoint.json` 和 `output/05_work_dirs/*/step_2040/gen_cap.csv`：

```bash
python 09_technology_interaction_analysis/run_scatter_triptych_standalone.py
```

输出位置：`09_technology_interaction_analysis/outputs/`。

## 10. 多求解方法比较

多方法比较脚本使用论文中已整理的 Switch、Single-Cut、Multi-Cut 指标，重新生成表格和图：

```bash
python algorithm_comparison/generate_comparison.py
python algorithm_comparison/generate_charts.py
```

输出位置：`algorithm_comparison/`。这些输出已在 `.gitignore` 中排除，可由脚本重建。

## 11. Robustness sampling analysis

该分析比较多组 1000 条结果。默认读取：

```text
output_runs/output/
output_runs/output_v2/
output_runs/output_v3/
output_runs/output_v4/
```

每个目录至少需要：

```text
checkpoint.json
04_summary/02_grid_robustness_data.csv
05_work_dirs/
```

运行：

```bash
python scripts/robustness_sampling_analysis.py
```

如果多组结果在其他位置，可设置：

```bash
set SCEQ_MULTI_RUN_ROOT=E:\sceq_multi_runs
```

或分别设置 `SCEQ_RUN_V1`、`SCEQ_RUN_V2`、`SCEQ_RUN_V3`、`SCEQ_RUN_V4`。输出位置默认是 `robustness_analysis/`。

## 12. Spatial homogenization mechanism experiment

机制实验默认读取同样的 `output/output_v2/output_v3/output_v4` 逻辑。当前配置文件在：

```text
mechanism_experiment/config.template.json
```

完整运行：

```bash
python mechanism_experiment/run_spatial_homogenization_experiment.py --config mechanism_experiment/config.template.json run-all --clean
```

分步运行也可用：

```bash
python mechanism_experiment/run_spatial_homogenization_experiment.py --config mechanism_experiment/config.template.json select-paths
python mechanism_experiment/run_spatial_homogenization_experiment.py --config mechanism_experiment/config.template.json build-weather
python mechanism_experiment/run_spatial_homogenization_experiment.py --config mechanism_experiment/config.template.json run-sceq
python mechanism_experiment/run_spatial_homogenization_experiment.py --config mechanism_experiment/config.template.json analyze
python mechanism_experiment/run_spatial_homogenization_experiment.py --config mechanism_experiment/config.template.json summarize
```

输出位置：`mechanism_experiment/` 下的 `baseline/`、`wind_homogenized/`、`solar_homogenized/`、`summary/`。

## 13. 推荐完整运行顺序

1. `python scripts/smoke_check.py`
2. 放置外部模型和数据，确认 `data_required_manifest.csv`
3. `python scripts/generate_scenarios.py`
4. `python scripts/02_run_sceq.py`
5. `python scripts/04_v2_plots.py`
6. `python scripts/plot_0714.py`
7. `python scripts/06_plot.py`
8. `python 09_technology_interaction_analysis/run_scatter_triptych_standalone.py`
9. `python algorithm_comparison/generate_comparison.py`
10. `python algorithm_comparison/generate_charts.py`
11. 准备多组 `output_runs/output*` 后运行 `python scripts/robustness_sampling_analysis.py`
12. 准备多组输出后运行机制实验 `run-all --clean`

## 14. 计算量提示

情景生成需要 GNN 权重、TensorFlow 负荷模型和 Normal 校准 `.npy`，计算量较大。SCEQ 主求解会对每条路径、每个规划期调用 Benders/Gurobi，完整论文规模不是 smoke test。robustness 和机制实验依赖多组完整 SCEQ 输出，数据体量也较大。

## 15. 常见问题

- `ModuleNotFoundError: sceq_lib`：请从 `code/` 根目录运行命令，或确认脚本没有被移动。
- `gurobipy.GurobiError`：检查 Gurobi license 和版本。
- `torch_geometric` 安装失败：按本机 CUDA 和 PyTorch 版本安装匹配 wheel。
- 找不到 `.pth/.h5/.joblib/.npy`：按 `data_required_manifest.csv` 放置外部模型和数据，或设置环境变量覆盖路径。
- 地图缺少 CLCD 底图：设置 `SCEQ_CLCD_TIF_PATH` 指向 `CLCD_v01_2022_albert.tif`。
- robustness 或 mechanism 报缺少 `output_v2`：这些脚本不是从单次 `output/` 自动生成多组结果，需要先准备四组完整输出或用环境变量指定。

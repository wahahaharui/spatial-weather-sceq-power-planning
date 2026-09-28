"""SCEQ V2 configuration for the reproducible code package.

This file keeps the original model parameters and solver settings, but replaces
machine-specific absolute paths with paths relative to the packaged repository
or environment variables. See README.md for the expected data layout.
"""
from __future__ import annotations

import os
from pathlib import Path


class SCEQConfigV2:
    CODE_ROOT = Path(__file__).resolve().parents[2]
    PROJECT_ROOT = CODE_ROOT
    SCEQ_ROOT = CODE_ROOT

    DATA_ROOT = Path(os.environ.get("SCEQ_DATA_ROOT", CODE_ROOT / "data")).resolve()
    MODEL_ROOT = Path(os.environ.get("SCEQ_MODEL_ROOT", DATA_ROOT / "models")).resolve()
    OUTPUT_DIR = Path(os.environ.get("SCEQ_OUTPUT_DIR", CODE_ROOT / "output")).resolve()

    NORMAL_MODEL_PATH = Path(os.environ.get(
        "SCEQ_NORMAL_MODEL_PATH",
        MODEL_ROOT / "GNN_cGAN_Normal_Outputs_0502" / "checkpoints" / "gnn_cgan_normal_epoch_1300.pth",
    )).resolve()
    EXTREME_MODEL_PATH = Path(os.environ.get(
        "SCEQ_EXTREME_MODEL_PATH",
        MODEL_ROOT / "GNN_cGAN_Extreme_Outputs_0502" / "checkpoints" / "gnn_cgan_extreme_latest.pth",
    )).resolve()
    STATIC_CSV = Path(os.environ.get(
        "SCEQ_STATIC_CSV",
        DATA_ROOT / "static" / "Lvliang_10km_Features_with_Shear.csv",
    )).resolve()
    NORMAL_DATA_DIR = Path(os.environ.get(
        "SCEQ_NORMAL_DATA_DIR",
        DATA_ROOT / "Datasets_calibration_0428" / "Normal",
    )).resolve()

    DEMAND_MODEL_DIR = Path(os.environ.get("SCEQ_DEMAND_MODEL_DIR", MODEL_ROOT / "demand_model")).resolve()
    DEMAND_MODEL_H5 = Path(os.environ.get(
        "SCEQ_DEMAND_MODEL_H5",
        DEMAND_MODEL_DIR / "MSE_weather_extreme_load_predictor_model.h5",
    )).resolve()
    PREPROCESSOR_JOBLIB = Path(os.environ.get(
        "SCEQ_PREPROCESSOR_JOBLIB",
        DEMAND_MODEL_DIR / "MSE_weather_extreme_preprocessor.joblib",
    )).resolve()
    SCALER_Y_JOBLIB = Path(os.environ.get(
        "SCEQ_SCALER_Y_JOBLIB",
        DEMAND_MODEL_DIR / "MSE_weather_extreme_scaler_y.joblib",
    )).resolve()

    INPUTS_TEMPLATE = Path(os.environ.get(
        "SCEQ_INPUTS_TEMPLATE",
        DATA_ROOT / "inputs_data_add_storage",
    )).resolve()
    BENDERS_V6_DIR = Path(os.environ.get("SCEQ_BENDERS_V6_DIR", CODE_ROOT / "benders_fixed_v6")).resolve()
    BOUNDARY_SHP_PATH = Path(os.environ.get(
        "SCEQ_BOUNDARY_SHP_PATH",
        DATA_ROOT / "static" / "Lvliang_Boundary_WGS84.shp",
    )).resolve()
    CLCD_TIF_PATH = Path(os.environ.get(
        "SCEQ_CLCD_TIF_PATH",
        DATA_ROOT / "external" / "CLCD_v01_2022_albert.tif",
    )).resolve()

    N_PATHS = int(os.environ.get("SCEQ_N_PATHS", "200"))
    PERIODS = [2025, 2030, 2035, 2040]
    P_EXTREME = 0.10

    EXTREME_CLASSES = {"Heatwave": 1, "Coldwave": 2, "Precipitation": 3}
    EXTREME_WEIGHTS = [0.3, 0.4, 0.3]
    EVENTS_PER_YEAR_MIN = 1
    EVENTS_PER_YEAR_MAX = 3
    EXTREME_SLICES_PER_TYPE = 50

    N_NODES = 222
    N_FEATURES = 4
    K_NEIGHBORS = 6
    SEQ_LEN = 24
    LATENT_DIM = 64
    EMBED_DIM = 32
    HIDDEN_DIM = 64
    CLASS_EMBED_DIM = 16
    COND_FEATURE_DIM = 8

    V_IN = 3.0
    V_OUT = 25.0
    V_RATED = 13.0
    W_ETA = 3.68e-3
    W_K = 0.908
    S_G_STC = 1000.0
    S_T_STC = 25.0
    S_ALPHA = -3.7e-3
    S_U = 0.0256
    W_EFF = 0.85
    HUB_HEIGHT = 80.0
    REF_HEIGHT = 10.0

    T_COOL_THRESH = 26.0
    T_HEAT_THRESH = 5.0
    K_COOL = 0.03
    K_HEAT = 0.01
    LVLIANG_LOAD_RATIO = 0.11087

    DEVICE = os.environ.get("SCEQ_DEVICE", "cuda")
    N_WORKERS = int(os.environ.get("SCEQ_N_WORKERS", "3"))
    CHECKPOINT_INTERVAL = 5

    BENDERS_MAX_ITER = int(os.environ.get("SCEQ_BENDERS_MAX_ITER", "200"))
    BENDERS_GAP_TOL = float(os.environ.get("SCEQ_BENDERS_GAP_TOL", "0.005"))

    def __init__(self):
        self.WEATHER_PATHS_DIR = self.OUTPUT_DIR / "01_weather_paths"
        self.CE_BASELINE_DIR = self.OUTPUT_DIR / "02_ce_baseline"
        self.PATH_RESULTS_DIR = self.OUTPUT_DIR / "03_path_results"
        self.SUMMARY_DIR = self.OUTPUT_DIR / "04_summary"
        self.WORK_DIRS = self.OUTPUT_DIR / "05_work_dirs"

        for d in [
            self.WEATHER_PATHS_DIR,
            self.CE_BASELINE_DIR,
            self.PATH_RESULTS_DIR,
            self.SUMMARY_DIR,
            self.WORK_DIRS,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        self.CHECKPOINT_FILE = self.OUTPUT_DIR / "checkpoint.json"
        self.LOG_FILE = self.OUTPUT_DIR / "00_run_log.txt"


cfg_v2 = SCEQConfigV2()
cfg = cfg_v2

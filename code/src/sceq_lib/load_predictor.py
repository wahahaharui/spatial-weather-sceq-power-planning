"""
负荷预测模块 — LSTM + 温度物理修正
基于 run_new_data_pipeline_260304.py 的 create_demand_features + predict_load
"""
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from joblib import load

from .config import cfg

# 模块级缓存 (避免重复加载)
_model_cache = None
_preprocessor_cache = None
_scaler_y_cache = None


def _get_demand_model():
    """延迟加载 LSTM 需求预测模型 (单例)"""
    global _model_cache, _preprocessor_cache, _scaler_y_cache
    if _model_cache is None:
        _model_cache = load_model(str(cfg.DEMAND_MODEL_H5))
        _preprocessor_cache = load(str(cfg.PREPROCESSOR_JOBLIB))
        _scaler_y_cache = load(str(cfg.SCALER_Y_JOBLIB))
    return _model_cache, _preprocessor_cache, _scaler_y_cache


def create_demand_features(df):
    """
    构建负荷预测模型所需的特征列
    完全复制 run_new_data_pipeline_260304.py 的逻辑
    """
    df = df.copy()
    df["time_dt"] = pd.to_datetime("2024-" + df["time"])
    df = df.set_index("time_dt")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wind_speed"] = df["v_80m"]
    df["delta_t"] = df["temp_c"].diff().fillna(0)
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_business_hour"] = (
        (df["hour"] >= 9) & (df["hour"] <= 17) & (df["is_weekend"] == 0)
    ).astype(int)

    def get_hour_type(hour):
        if 0 <= hour <= 5:
            return "overnight_valley"
        if 6 <= hour <= 8:
            return "morning_ramp"
        if 9 <= hour <= 11:
            return "morning_peak_work"
        if 12 <= hour <= 13:
            return "midday_lull"
        if 14 <= hour <= 17:
            return "afternoon_work"
        if 18 <= hour <= 22:
            return "evening_peak_residential"
        return "late_night"

    df["hour_type"] = df["hour"].apply(get_hour_type)
    df["t2m_rolling_mean_6"] = df["temp_c"].rolling(window=6).mean().ffill().bfill()
    df["t2m"] = df["temp_c"]
    df["tp"] = df["pr"].fillna(0)
    return df


def predict_load_8760(weather_8760, alpha):
    """
    从 8760h 气象数据预测吕梁负荷 (MW)

    Args:
        weather_8760: (8760, 222, 4) — [t2m, w10, GHI, precip]
        alpha: (222,) — 风切变指数

    Returns:
        load_mw: (8760,) — 吕梁负荷 MW
    """
    model, preprocessor, scaler_y = _get_demand_model()

    # 空间平均气象 → 单站点表示
    avg_temp = np.mean(weather_8760[:, :, 0], axis=1)   # (8760,)
    avg_w10 = np.mean(weather_8760[:, :, 1], axis=1)    # (8760,) 10m wind
    avg_pr = np.mean(weather_8760[:, :, 3], axis=1)     # (8760,) precip

    # 10m → 80m 风速转换 (用于负荷预测特征)
    alpha_mean = np.mean(alpha)
    avg_v80 = avg_w10 * ((cfg.HUB_HEIGHT / cfg.REF_HEIGHT) ** alpha_mean)

    # 构建时间序列 DataFrame
    hours_per_day = 24
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    time_strings = []
    for m, ndays in enumerate(days_in_month, 1):
        for d in range(1, ndays + 1):
            for h in range(24):
                time_strings.append(f"{m:02d}-{d:02d} {h:02d}:00")

    df = pd.DataFrame({
        "time": time_strings,
        "temp_c": avg_temp,
        "v_80m": avg_v80,
        "pr": avg_pr,
    })

    # LSTM 特征工程
    df_feat = create_demand_features(df)

    features_to_use = [
        "t2m", "wind_speed", "tp", "delta_t", "t2m_rolling_mean_6",
        "hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_business_hour", "hour_type",
    ]
    X_proc = preprocessor.transform(df_feat[features_to_use])

    # LSTM 序列预测 (time_steps=24)
    time_steps = 24
    padding = np.tile(X_proc[0], (time_steps - 1, 1))
    X_padded = np.vstack([padding, X_proc])
    X_seq = np.array([
        X_padded[i : i + time_steps] for i in range(len(X_proc))
    ])
    y_scaled = model.predict(X_seq, verbose=0, batch_size=512)
    y_base = scaler_y.inverse_transform(y_scaled).flatten()

    # ---- 温度物理修正 (与现有代码一致) ----
    corrected_load = y_base.copy()
    temp_array = avg_temp

    for i in range(len(temp_array)):
        T = temp_array[i]
        if T > cfg.T_COOL_THRESH:
            corrected_load[i] = y_base[i] * (
                1 + (T - cfg.T_COOL_THRESH) * cfg.K_COOL
            )
        elif T < cfg.T_HEAT_THRESH:
            corrected_load[i] = y_base[i] * (
                1 + (cfg.T_HEAT_THRESH - T) * cfg.K_HEAT
            )

    # 吕梁负荷折算
    load_mw = corrected_load * cfg.LVLIANG_LOAD_RATIO

    return load_mw.astype(np.float64)

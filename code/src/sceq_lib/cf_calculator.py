"""
容量因子 (CF) 计算模块 — 含 UTC→北京时间修复
"""
import numpy as np

from .config import cfg


def calculate_cf(weather_8760, alpha):
    """
    从气象数据计算风电/光伏 CF

    Args:
        weather_8760: (8760, 222, 4) — [t2m(°C), w10(m/s), GHI(W/m²), precip(mm/h)]
        alpha: (222,) — 各网格风切变指数

    Returns:
        wind_cf:  (8760, 222) clipped to [0, 1]
        solar_cf: (8760, 222) clipped to [0, 1]
    """
    T, N = weather_8760.shape[0], weather_8760.shape[1]

    # 确保非负
    weather_8760 = weather_8760.copy()
    weather_8760[:, :, 1:] = np.maximum(weather_8760[:, :, 1:], 0.0)

    temp = weather_8760[:, :, 0]   # °C
    v10 = weather_8760[:, :, 1]    # m/s at 10m
    ghi = weather_8760[:, :, 2]    # W/m²

    # ---- 风电 CF ----
    # 10m → 80m 风速外推
    v80 = v10 * ((cfg.HUB_HEIGHT / cfg.REF_HEIGHT) ** np.tile(alpha, (T, 1)))

    w_cf = np.zeros_like(v80)
    # 区间1: V_in ~ V_rated (线性爬坡)
    m1 = (v80 >= cfg.V_IN) & (v80 <= cfg.V_RATED)
    w_cf[m1] = (
        (cfg.W_ETA * temp[m1] + cfg.W_K)
        * (v80[m1] - cfg.V_IN)
        / (cfg.V_RATED - cfg.V_IN)
    )
    # 区间2: V_rated ~ V_out (额定满发)
    m2 = (v80 > cfg.V_RATED) & (v80 < cfg.V_OUT)
    w_cf[m2] = cfg.W_ETA * temp[m2] + cfg.W_K

    # 综合效率折损 (15%)
    w_cf = w_cf * cfg.W_EFF

    # ---- 光伏 CF ----
    s_cf = np.zeros_like(ghi)
    ms = ghi > 0
    t_mod = temp[ms] + (cfg.S_U * ghi[ms])
    s_cf[ms] = (ghi[ms] / cfg.S_G_STC) * (
        1 + cfg.S_ALPHA * (t_mod - cfg.S_T_STC)
    )

    return np.clip(w_cf, 0, 1), np.clip(s_cf, 0, 1)


def apply_utc_to_beijing(data_8760):
    """
    UTC → 北京时间 (UTC+8)
    使用 np.roll(shift=8, axis=0)

    Args:
        data_8760: (8760, ...) 任意维度数组

    Returns:
        shifted: (8760, ...) 北京时间
    """
    return np.roll(data_8760, shift=8, axis=0)

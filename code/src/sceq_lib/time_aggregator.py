"""
时间聚合: 8760h → 288h (12月 × 24h 月均小时剖面)
"""
import numpy as np

# 每月天数 (非闰年, 与现有数据一致)
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def aggregate_8760_to_288(data_8760):
    """
    将 (8760, ...) 聚合为 (288, ...)
    方法: 对每月每天同一小时取均值

    Args:
        data_8760: (8760, ...) — 支持 1D/2D/3D

    Returns:
        data_288: (288, ...)
    """
    shape_8760 = data_8760.shape
    if shape_8760[0] != 8760:
        raise ValueError(f"Expected 8760 time steps, got {shape_8760[0]}")

    result_shape = (288,) + shape_8760[1:]
    result = np.zeros(result_shape, dtype=data_8760.dtype)

    idx_8760 = 0
    for m, ndays in enumerate(DAYS_IN_MONTH):
        for h in range(24):
            # 取该月每天第 h 小时的值
            hour_indices = np.arange(idx_8760 + h, idx_8760 + ndays * 24, 24)
            result[m * 24 + h] = np.mean(data_8760[hour_indices], axis=0)
        idx_8760 += ndays * 24

    return result


def expand_288_to_1152(cf_288_list, load_288_list):
    """
    将4个时期的 288h 数据拼接为 1152h (Benders 模型格式)

    Args:
        cf_288_list:  list of 4 arrays, each (288, 222)
        load_288_list: list of 4 arrays, each (288,)

    Returns:
        cf_1152:  (1152, 222) — 每个时期 288h 按序拼接
        load_1152: (1152,) — 同上
    """
    cf_1152 = np.concatenate(cf_288_list, axis=0)
    load_1152 = np.concatenate(load_288_list, axis=0)
    return cf_1152, load_1152


def build_timepoint_mapping():
    """
    构建 timepoint → 8760h 位置 的映射 (用于调试和验证)

    Returns:
        tp_to_hour: {timepoint_id: (month, hour_of_day)}
    """
    tp_to_hour = {}
    tp_id = 1
    for period_idx in range(4):
        for m in range(12):
            for h in range(24):
                tp_to_hour[tp_id] = (period_idx, m + 1, h)
                tp_id += 1
    return tp_to_hour

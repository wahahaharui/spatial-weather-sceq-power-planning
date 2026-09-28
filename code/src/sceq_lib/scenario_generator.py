"""
SCEQ 气象路径生成器
每路径 = 4 期 × 8760h × 222 网格 × (气象 + CF + 负荷)
"""
import os
import glob
import json
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from .config import cfg
from .gnn_models import GeneratorNormal, GeneratorExtreme
from .cf_calculator import calculate_cf, apply_utc_to_beijing
from .load_predictor import predict_load_8760


class ScenarioGenerator:
    """生成 200 条气象路径"""

    def __init__(self):
        self.device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"[ScenarioGenerator] 设备: {self.device}")

        # 加载静态数据
        self.grid_df = pd.read_csv(cfg.STATIC_CSV)
        self.alpha = self.grid_df["shear_all_"].values.astype(np.float64)

        # 构建空间图
        self.edge_index = self._build_graph().to(self.device)

        # 准备 scaler (在 Normal 训练集上拟合)
        self.scaler = self._prepare_scaler()

        # 加载 GNN 生成器
        self.G_n = self._load_normal_generator()
        self.G_e = self._load_extreme_generator()

        # 预生成极端切片池
        self.extreme_bank = self._pre_generate_extreme_bank()

    # ================================================================
    # 内部: 图构建 & 模型加载
    # ================================================================

    def _build_graph(self):
        coords = self.grid_df[["lat", "lon"]].values
        nbrs = NearestNeighbors(n_neighbors=cfg.K_NEIGHBORS + 1, algorithm="ball_tree")
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)
        src, dst = [], []
        for i in range(cfg.N_NODES):
            for j in range(1, cfg.K_NEIGHBORS + 1):
                src.append(i)
                dst.append(indices[i, j])
        return torch.tensor([src, dst], dtype=torch.long)

    def _prepare_scaler(self):
        """在 Normal 训练集上拟合 MinMaxScaler (与 prepare_context 一致)"""
        normal_files = glob.glob(os.path.join(cfg.NORMAL_DATA_DIR, "*.npy"))
        X_list = [np.load(f) for f in normal_files]

        # 80% 训练集
        indices = np.arange(len(X_list))
        train_idx, _ = train_test_split(indices, test_size=0.2, random_state=42)
        X_fit = np.array(X_list)[train_idx].reshape(-1, 4)

        # log1p 降水
        X_fit[:, 3] = np.log1p(np.maximum(X_fit[:, 3], 0.0))

        scaler = MinMaxScaler((-1, 1))
        scaler.fit(X_fit)

        # 准备种子张量 (12月31日最后一小时)
        seed_files = [f for f in normal_files if "1231" in os.path.basename(f)]
        if not seed_files:
            seed_files = [normal_files[0]]
        seed_path = seed_files[0]
        seed_data = np.load(seed_path)[-1, :, :].reshape(-1, 4).copy()
        seed_data[:, 3] = np.log1p(np.maximum(seed_data[:, 3], 0.0))
        self.seed_tensor = torch.FloatTensor(
            scaler.transform(seed_data).reshape(1, cfg.N_NODES, cfg.N_FEATURES)
        ).to(self.device)

        print(f"[ScenarioGenerator] Scaler 拟合完成, 种子文件: {os.path.basename(seed_path)}")
        return scaler

    def _load_normal_generator(self):
        G = GeneratorNormal().to(self.device)
        ckpt = torch.load(str(cfg.NORMAL_MODEL_PATH), map_location=self.device)
        G.load_state_dict(ckpt["G_state_dict"])
        G.eval()
        print(f"[ScenarioGenerator] Normal 生成器加载完成")
        return G

    def _load_extreme_generator(self):
        G = GeneratorExtreme(num_classes=4).to(self.device)
        ckpt = torch.load(str(cfg.EXTREME_MODEL_PATH), map_location=self.device)
        G.load_state_dict(ckpt["G_state_dict"])
        G.eval()
        print(f"[ScenarioGenerator] Extreme 生成器加载完成")
        return G

    def _pre_generate_extreme_bank(self):
        """预生成极端切片池 (每种类型 50 个 24h 切片)"""
        bank = {}
        with torch.no_grad():
            for label_name, label in cfg.EXTREME_CLASSES.items():
                z = torch.randn(cfg.EXTREME_SLICES_PER_TYPE, cfg.LATENT_DIM).to(self.device)
                c = torch.full((cfg.EXTREME_SLICES_PER_TYPE,), label, dtype=torch.long).to(self.device)
                slices = self.G_e(z, self.edge_index, c).cpu().numpy()  # (50, 24, 222, 4)
                bank[label] = slices
        print(f"[ScenarioGenerator] 极端切片池预生成完成: {list(bank.keys())}")
        return bank

    # ================================================================
    # 主入口
    # ================================================================

    def generate_all_paths(self, resume_from=None):
        """
        生成全部 200 条气象路径, 逐条保存到磁盘

        Args:
            resume_from: set of already-completed path IDs (断点恢复)

        Returns:
            weather_paths_meta: list of path metadata dicts
        """
        completed = resume_from or set()
        meta_list = []

        t_start = time.time()
        for path_id in range(cfg.N_PATHS):
            if path_id in completed:
                print(f"  Path {path_id:03d}: 断点跳过 (已完成)")
                meta_list.append({"path_id": path_id, "skipped": True})
                continue

            t_path = time.time()
            path_dir = cfg.WEATHER_PATHS_DIR / f"path_{path_id:03d}"
            path_dir.mkdir(parents=True, exist_ok=True)

            try:
                weather_data = self._generate_one_path(path_id)
                self._save_path(path_id, weather_data)
                del weather_data
                gc.collect()

                elapsed = time.time() - t_path
                meta_list.append({
                    "path_id": path_id,
                    "elapsed_s": elapsed,
                    "skipped": False,
                })

                # 实时进度
                avg_t = (time.time() - t_start) / (len(meta_list) - len(completed) + 1)
                remaining = cfg.N_PATHS - path_id - 1
                eta_s = avg_t * remaining
                print(f"  Path {path_id:03d}/{cfg.N_PATHS-1} | "
                      f"耗时: {elapsed:.0f}s | ETA: {eta_s/60:.0f}min")

            except Exception as e:
                print(f"  ❌ Path {path_id:03d} 失败: {e}")
                meta_list.append({"path_id": path_id, "error": str(e)})

        total_t = time.time() - t_start
        print(f"\n[ScenarioGenerator] 全部完成! 总耗时: {total_t/60:.1f}min")

        # 保存元数据
        import json
        with open(cfg.WEATHER_PATHS_DIR / "generation_meta.json", "w") as f:
            json.dump({
                "n_paths": cfg.N_PATHS,
                "n_completed": sum(1 for m in meta_list if m.get("skipped") or "elapsed_s" in m),
                "total_elapsed_s": total_t,
                "paths": [
                    {k: v for k, v in m.items() if k != "error"}
                    for m in meta_list
                ],
            }, f, indent=2)

        return meta_list

    # ================================================================
    # 单条路径生成
    # ================================================================

    def _generate_one_path(self, path_id):
        """生成单条路径的 4 期气象"""
        base_seed = 10000 + path_id
        weather_data = {}

        for period_idx, period in enumerate(cfg.PERIODS):
            year_seed = base_seed * 100 + period_idx
            rng = np.random.default_rng(year_seed + 9999)

            # 判断该期是否为极端年
            is_extreme = rng.random() < cfg.P_EXTREME

            # 生成 8760h 气象
            weather_raw = self._generate_weather_year(year_seed, is_extreme, rng)

            # 计算 CF (在 UTC 修复前, CF 函数内部处理)
            wind_cf, solar_cf = calculate_cf(weather_raw, self.alpha)

            # UTC → 北京时间
            weather_bj = apply_utc_to_beijing(weather_raw)
            wind_cf_bj = apply_utc_to_beijing(wind_cf)
            solar_cf_bj = apply_utc_to_beijing(solar_cf)

            # 负荷预测 (基于北京时间的天气)
            load_mw = predict_load_8760(weather_bj, self.alpha)

            weather_data[period] = {
                "weather": weather_bj.astype(np.float32),
                "wind_cf": wind_cf_bj.astype(np.float32),
                "solar_cf": solar_cf_bj.astype(np.float32),
                "load": load_mw.astype(np.float32),
                "is_extreme": is_extreme,
            }

        return weather_data

    def _generate_weather_year(self, seed, is_extreme, rng):
        """
        生成一个完整气象年 (8760h, 222, 4)

        步骤:
        1. Normal GNN 自回归生成 365 天 (每天 24h)
        2. 如 is_extreme: 注入极端事件
        3. inverse_transform + expm1(precip) → 物理值
        """
        torch.manual_seed(seed)

        with torch.no_grad():
            xp = self.seed_tensor.clone()
            daily_outputs = []

            for d in range(365):
                # 月份条件 (近似: 每30天一个月份)
                m = (d // 30) % 12 + 1
                c = torch.tensor([[
                    np.sin(2 * np.pi * m / 12),
                    np.cos(2 * np.pi * m / 12),
                    0, 0, 0, 0, 0, 0,
                ]], dtype=torch.float).to(self.device)

                gd = self.G_n(
                    torch.randn(1, cfg.LATENT_DIM).to(self.device),
                    self.edge_index, c, xp,
                )
                daily_outputs.append(gd.squeeze(0).cpu().numpy())
                xp = gd[:, -1, :, :]

        # 拼接 365 天 → (8760, 222, 4)
        weather_scaled = np.concatenate(daily_outputs, axis=0)

        # inverse_transform → 物理值
        weather_phys = self.scaler.inverse_transform(
            weather_scaled.reshape(-1, 4)
        ).reshape(8760, cfg.N_NODES, 4)

        # 降水: expm1 还原
        weather_phys[:, :, 3] = np.expm1(weather_phys[:, :, 3])

        # 极端事件注入
        if is_extreme:
            weather_phys = self._inject_extreme_events(weather_phys, rng)

        # 确保非负 (风速、GHI、降水)
        weather_phys[:, :, 1:] = np.maximum(weather_phys[:, :, 1:], 0.0)

        return weather_phys

    # ================================================================
    # 极端事件注入
    # ================================================================

    def _inject_extreme_events(self, weather, rng):
        """在 8760h 基准年中注入极端 24h 切片"""
        n_events = rng.integers(cfg.EVENTS_PER_YEAR_MIN, cfg.EVENTS_PER_YEAR_MAX + 1)

        for _ in range(n_events):
            label = int(rng.choice([1, 2, 3], p=cfg.EXTREME_WEIGHTS))

            # 季节窗口 (24h 起始位置)
            windows = self._get_seasonal_windows(label)
            if not windows:
                continue

            start = int(rng.choice(windows))

            # 从极端池随机取一切片
            bank_idx = rng.integers(0, cfg.EXTREME_SLICES_PER_TYPE)
            e_slice = self.extreme_bank[label][bank_idx]  # (24, 222, 4)

            # inverse_transform 极端切片
            e_phys = self.scaler.inverse_transform(
                e_slice.reshape(-1, 4)
            ).reshape(24, cfg.N_NODES, 4)
            e_phys[:, :, 3] = np.expm1(e_phys[:, :, 3])
            e_phys[:, :, 1:] = np.maximum(e_phys[:, :, 1:], 0.0)

            # 鲁棒注入: 取 min(基准, 极端)
            end = min(start + 24, 8760)
            actual_len = end - start
            weather[start:end] = np.minimum(
                weather[start:end], e_phys[:actual_len]
            )

        return weather

    def _get_seasonal_windows(self, label):
        """返回该极端类型的季节窗口 (24h 起始索引列表)"""
        if label == 2:  # Coldwave: Dec-Feb
            return list(range(0, 1416, 24)) + list(range(8016, 8736, 24))
        elif label == 1:  # Heatwave: Jun-Aug
            return list(range(3624, 5832, 24))
        elif label == 3:  # Precipitation: Jul-Sep
            return list(range(4344, 6552, 24))
        else:
            return list(range(0, 8736, 24))

    # ================================================================
    # 保存 & 加载
    # ================================================================

    def _save_path(self, path_id, weather_data):
        """保存单条路径到磁盘"""
        path_dir = cfg.WEATHER_PATHS_DIR / f"path_{path_id:03d}"
        path_dir.mkdir(parents=True, exist_ok=True)

        for period, data in weather_data.items():
            fpath = path_dir / f"period_{period}.npz"
            np.savez_compressed(
                fpath,
                weather=data["weather"],
                wind_cf=data["wind_cf"],
                solar_cf=data["solar_cf"],
                load=data["load"],
                is_extreme=np.array([data["is_extreme"]]),
            )

    @staticmethod
    def load_path(path_id):
        """加载单条路径的所有期数据"""
        path_dir = cfg.WEATHER_PATHS_DIR / f"path_{path_id:03d}"
        weather_data = {}
        for period in cfg.PERIODS:
            fpath = path_dir / f"period_{period}.npz"
            if fpath.exists():
                loaded = np.load(fpath)
                weather_data[period] = {
                    "weather": loaded["weather"],
                    "wind_cf": loaded["wind_cf"],
                    "solar_cf": loaded["solar_cf"],
                    "load": loaded["load"],
                    "is_extreme": bool(loaded["is_extreme"].item()),
                }
        return weather_data

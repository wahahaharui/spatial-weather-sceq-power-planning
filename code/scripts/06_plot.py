"""
Publication-style plotting for SCEQ planning results.

Outputs:
1. Capacity Pathways
2. Wind Build Frequency (1x4 panel)
3. Solar Build Frequency (1x4 panel)
4. Wind Capacity (1x4 panel)
5. Solar Capacity (1x4 panel)
6. Wind Build Frequency (2040)
7. Solar Build Frequency (2040)
8. Wind Capacity (2040)
9. Solar Capacity (2040)
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.lines import Line2D
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT))
from sceq_lib.config import cfg_v2 as cfg

try:
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    from rasterio.windows import from_bounds

    GIS_AVAILABLE = True
except ImportError:
    GIS_AVAILABLE = False


PROJECT_DIR = cfg.PROJECT_ROOT
SCEQ_DIR = cfg.SCEQ_ROOT
OUTDIR = SCEQ_DIR / "06_plot"
OUTDIR.mkdir(parents=True, exist_ok=True)

BOUND_SHP_PATH = cfg.BOUNDARY_SHP_PATH
CLCD_TIF_PATH = cfg.CLCD_TIF_PATH
SUMMARY_CSV = cfg.SUMMARY_DIR / "02_grid_robustness_data.csv"
CHECKPOINT_FILE = cfg.CHECKPOINT_FILE
WEATHER_ROOT = cfg.WEATHER_PATHS_DIR
PERIODS = cfg.PERIODS

TECH_COLORS = {
    "wind": "#1f5a91",
    "solar": "#c67c00",
    "storage": "#2a9d8f",
}

CLCD_CLASSES = {
    1: ("Cropland", "#c4da83"),
    2: ("Forest", "#6faf7a"),
    3: ("Shrub", "#aea2df"),
    4: ("Grassland", "#9dd074"),
    5: ("Water", "#73bee8"),
    6: ("Snow/Ice", "#f3f5f7"),
    7: ("Barren", "#e8c24a"),
    8: ("Impervious", "#ca8e8a"),
    9: ("Wetland", "#7fced5"),
}
CLCD_CMAP = mcolors.ListedColormap([CLCD_CLASSES[k][1] for k in sorted(CLCD_CLASSES)])
CLCD_NORM = mcolors.BoundaryNorm(boundaries=list(range(1, 11)), ncolors=9)


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )


def load_results():
    with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    all_results = ckpt.get("all_results", [])
    success = [r for r in all_results if "error" not in r and r.get("steps")]
    return success


def build_capacity_dataframe(success):
    rows = []
    for r in success:
        for step in r["steps"]:
            rows.append(
                {
                    "path_id": r["path_id"],
                    "period": step["period"],
                    "wind_mw": float(step.get("wind_build_mw", 0) or 0),
                    "solar_mw": float(step.get("solar_build_mw", 0) or 0),
                    "storage_mw": float(step.get("storage_build_mw", 0) or 0),
                }
            )
    return pd.DataFrame(rows)


def build_capacity_array(df, tech):
    pivot = df.pivot(index="path_id", columns="period", values=f"{tech}_mw")
    pivot = pivot.reindex(columns=PERIODS).sort_index()
    return pivot.values


def plot_capacity_pathways(df):
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    x = np.arange(len(PERIODS))

    label_cfg = {
        "wind": {"dy": 55, "box_fc": "#eef5fb", "box_ec": "#b9d4ea", "range_y": 130},
        "solar": {"dy": 45, "box_fc": "#fff4e6", "box_ec": "#f2c98a", "range_y": 95},
        "storage": {"dy": 40, "box_fc": "#eaf7f4", "box_ec": "#a8ddd2", "range_y": 78},
    }

    for tech, label in [("wind", "Wind"), ("solar", "Solar"), ("storage", "Storage")]:
        arr = build_capacity_array(df, tech)
        mean_vals = np.mean(arr, axis=0)
        p5_vals = np.percentile(arr, 5, axis=0)
        p95_vals = np.percentile(arr, 95, axis=0)

        ax.fill_between(x, p5_vals, p95_vals, color=TECH_COLORS[tech], alpha=0.12, zorder=1)
        ax.plot(
            x,
            mean_vals,
            color=TECH_COLORS[tech],
            linewidth=3.0,
            marker="o",
            markersize=8.5,
            markerfacecolor="white",
            markeredgewidth=2.2,
            label=label,
            zorder=3,
        )

        for xi, mean_val in zip(x, mean_vals):
            dx = 0.02 if (tech == "wind" and xi == x[-1]) else 0.0
            ax.text(
                xi + dx,
                mean_val + label_cfg[tech]["dy"],
                f"{mean_val:,.0f}",
                ha="center",
                va="bottom",
                fontsize=10.0,
                color=TECH_COLORS[tech],
                bbox=dict(
                    boxstyle="round,pad=0.22",
                    facecolor=label_cfg[tech]["box_fc"],
                    edgecolor=label_cfg[tech]["box_ec"],
                    linewidth=0.8,
                    alpha=0.96,
                ),
                zorder=5,
            )

        ax.annotate(
            f"P5-P95: {p5_vals[-1]:,.0f}-{p95_vals[-1]:,.0f}",
            xy=(x[-1], mean_vals[-1]),
            xytext=(x[-1] - 0.86, mean_vals[-1] + label_cfg[tech]["range_y"]),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=10.0,
            color=TECH_COLORS[tech],
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.92),
            arrowprops=dict(
                arrowstyle="-",
                color=TECH_COLORS[tech],
                lw=1.0,
                alpha=0.75,
                shrinkA=0,
                shrinkB=5,
            ),
            zorder=5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in PERIODS])
    ax.set_xlim(-0.12, len(PERIODS) - 0.86)
    ax.set_ylabel("Cumulative Capacity (MW)")
    ax.set_xlabel("Period")
    ax.set_title("Capacity Pathways", fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#d0d0d0")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=0.35)
    fig.savefig(OUTDIR / "01_capacity_pathways.png")
    plt.close(fig)


def clip_clcd_ly():
    if not GIS_AVAILABLE:
        return None, None, None

    if not BOUND_SHP_PATH.exists():
        return None, None, None

    boundary_gdf = gpd.read_file(BOUND_SHP_PATH)
    if boundary_gdf.crs is None:
        boundary_gdf.set_crs("EPSG:4326", inplace=True)
    elif boundary_gdf.crs != "EPSG:4326":
        boundary_gdf = boundary_gdf.to_crs("EPSG:4326")

    raster_data = None
    raster_extent = None
    if CLCD_TIF_PATH.exists():
        with rasterio.open(CLCD_TIF_PATH) as src:
            boundary_proj = boundary_gdf.to_crs(src.crs)
            xmin_proj, ymin_proj, xmax_proj, ymax_proj = boundary_proj.total_bounds
            x_buf = (xmax_proj - xmin_proj) * 0.05
            y_buf = (ymax_proj - ymin_proj) * 0.05
            window = from_bounds(
                xmin_proj - x_buf,
                ymin_proj - y_buf,
                xmax_proj + x_buf,
                ymax_proj + y_buf,
                src.transform,
            )
            window = window.round_shape()
            window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))

            out_image = src.read(window=window)
            out_transform = rasterio.windows.transform(window, src.transform)

            transform, width, height = calculate_default_transform(
                src.crs,
                "EPSG:4326",
                out_image.shape[2],
                out_image.shape[1],
                *rasterio.transform.array_bounds(out_image.shape[1], out_image.shape[2], out_transform),
            )
            dst_image = np.zeros((out_image.shape[0], height, width), dtype=out_image.dtype)
            reproject(
                source=out_image,
                destination=dst_image,
                src_transform=out_transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
            )
            shapes = [(geom, 1) for geom in boundary_gdf.geometry]
            mask_arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8")
            raster_data = np.where(mask_arr == 1, dst_image[0], np.nan)
            raster_data = np.where(raster_data == 0, np.nan, raster_data)
            xmin, ymin, xmax, ymax = rasterio.transform.array_bounds(height, width, transform)
            raster_extent = [xmin, xmax, ymin, ymax]

    return boundary_gdf, raster_data, raster_extent


def clcd_legend_handles():
    return [
        mpatches.Patch(facecolor=CLCD_CLASSES[k][1], edgecolor="none", alpha=0.75, label=CLCD_CLASSES[k][0])
        for k in sorted(CLCD_CLASSES.keys())
    ]


def plot_base_map(ax, boundary_gdf, raster_data, raster_extent, show_clcd_legend=False):
    if raster_data is not None and raster_extent is not None:
        ax.imshow(
            raster_data,
            cmap=CLCD_CMAP,
            norm=CLCD_NORM,
            extent=raster_extent,
            origin="upper",
            alpha=0.66,
            zorder=1,
        )

    if boundary_gdf is not None:
        boundary_gdf.plot(ax=ax, facecolor="none", edgecolor="#5f5f5f", linewidth=1.0, zorder=2)
        bounds = boundary_gdf.total_bounds
        ax.set_xlim(bounds[0] - 0.05, bounds[2] + 0.05)
        ax.set_ylim(bounds[1] - 0.05, bounds[3] + 0.05)

    if show_clcd_legend:
        ax.legend(
            handles=clcd_legend_handles(),
            loc="upper left",
            bbox_to_anchor=(1.01, 0.02),
            ncol=3,
            fontsize=7.4,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#d0d0d0",
            title="CLCD",
            title_fontsize=8,
            borderpad=0.35,
            columnspacing=0.8,
            handlelength=1.0,
            handletextpad=0.4,
        )


def cf_legend_handles(tech):
    line_color = "#6ea6c9" if tech == "wind" else "#d8a85f"
    return [
        Line2D([0], [0], color=line_color, lw=0.75, alpha=0.62, label="Low CF"),
        Line2D([0], [0], color=line_color, lw=1.05, alpha=0.62, label="High CF"),
    ]


def build_boundary_mask(xx, yy, boundary_gdf):
    if boundary_gdf is None or boundary_gdf.empty:
        return np.ones_like(xx, dtype=bool)

    points = np.column_stack([xx.ravel(), yy.ravel()])
    inside = np.zeros(points.shape[0], dtype=bool)
    for geom in boundary_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for poly in geoms:
            path = MplPath(np.asarray(poly.exterior.coords))
            inside |= path.contains_points(points)
    return inside.reshape(xx.shape)


def compute_cf_mean_from_pool(gdf):
    cache_path = OUTDIR / "cf_mean_cache.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    records = []
    available_paths = sorted({int(p.name.split("_")[1]) for p in WEATHER_ROOT.glob("path_*") if p.is_dir()})
    n_nodes = len(gdf)

    for period in PERIODS:
        wind_sum = np.zeros(n_nodes, dtype=np.float64)
        solar_sum = np.zeros(n_nodes, dtype=np.float64)
        n_paths = 0

        for path_id in available_paths:
            npz_path = WEATHER_ROOT / f"path_{path_id:03d}" / f"period_{period}.npz"
            if not npz_path.exists():
                continue

            loaded = np.load(npz_path)
            wind_cf_mean = loaded["wind_cf"].mean(axis=0)
            solar_cf_mean = loaded["solar_cf"].mean(axis=0)
            wind_sum += wind_cf_mean
            solar_sum += solar_cf_mean
            n_paths += 1

        if n_paths == 0:
            continue

        wind_avg = wind_sum / n_paths
        solar_avg = solar_sum / n_paths

        for i, row in enumerate(gdf.itertuples(index=False)):
            records.append(
                {
                    "period": period,
                    "lon": row.lon,
                    "lat": row.lat,
                    "wind_cf_mean": float(wind_avg[i]),
                    "solar_cf_mean": float(solar_avg[i]),
                }
            )

    cf_df = pd.DataFrame(records)
    cf_df.to_csv(cache_path, index=False)
    return cf_df


def overlay_cf_contours(ax, gdf, cf_df, period, tech, boundary_gdf):
    cf_col = "wind_cf_mean" if tech == "wind" else "solar_cf_mean"
    line_color = "#6ea6c9" if tech == "wind" else "#d8a85f"

    cf_period = cf_df[cf_df["period"] == period]
    if cf_period.empty:
        return

    lon_min, lon_max = gdf["lon"].min(), gdf["lon"].max()
    lat_min, lat_max = gdf["lat"].min(), gdf["lat"].max()
    xi = np.linspace(lon_min, lon_max, 180)
    yi = np.linspace(lat_min, lat_max, 180)
    xx, yy = np.meshgrid(xi, yi)
    zz = griddata(
        (cf_period["lon"].values, cf_period["lat"].values),
        cf_period[cf_col].values,
        (xx, yy),
        method="linear",
    )
    if np.isnan(zz).all():
        return
    boundary_mask = build_boundary_mask(xx, yy, boundary_gdf)
    zz = np.where(boundary_mask, zz, np.nan)

    low = float(np.nanpercentile(zz, 20))
    high = float(np.nanpercentile(zz, 85))
    if high <= low:
        return
    levels = np.linspace(low, high, 5)
    linewidths = np.linspace(0.7, 1.15, len(levels))
    contour = ax.contour(
        xx,
        yy,
        zz,
        levels=levels,
        colors=line_color,
        linewidths=linewidths,
        alpha=0.62,
        zorder=2.4,
    )
    ax.clabel(
        contour,
        contour.levels,
        inline=True,
        inline_spacing=2,
        fmt="%.2f",
        fontsize=7.8,
        colors=line_color,
    )


def scatter_style(config, values):
    if config["is_capacity"]:
        sizes = np.clip(values / max(config["vmax_ref"], 1) * 320 + 40, 40, 520)
    else:
        sizes = 132
    return sizes


def plot_beautified_panels(gdf, boundary_gdf, raster_data, raster_extent, cf_df):
    wind_cap_all = np.concatenate([gdf[f"wind_cumul_mean_{p}"].values for p in PERIODS])
    solar_cap_all = np.concatenate([gdf[f"solar_cumul_mean_{p}"].values for p in PERIODS])

    metrics_config = [
        {
            "name": "Wind Build Frequency",
            "col_key": "wind_cumul_freq",
            "cmap": "YlOrRd",
            "vmin": 0,
            "vmax": 1.0,
            "cbar_label": "Build Frequency",
            "filename": "02_wind_build_frequency_panel.png",
            "is_capacity": False,
            "tech": "wind",
        },
        {
            "name": "Solar Build Frequency",
            "col_key": "solar_cumul_freq",
            "cmap": "YlOrRd",
            "vmin": 0,
            "vmax": 1.0,
            "cbar_label": "Build Frequency",
            "filename": "03_solar_build_frequency_panel.png",
            "is_capacity": False,
            "tech": "solar",
        },
        {
            "name": "Wind Capacity",
            "col_key": "wind_cumul_mean",
            "cmap": "Blues",
            "vmin": 0,
            "vmax": np.percentile(wind_cap_all, 98),
            "cbar_label": "Mean Capacity (MW)",
            "filename": "04_wind_capacity_panel.png",
            "is_capacity": True,
            "vmax_ref": np.percentile(wind_cap_all, 98),
            "tech": "wind",
        },
        {
            "name": "Solar Capacity",
            "col_key": "solar_cumul_mean",
            "cmap": "Oranges",
            "vmin": 0,
            "vmax": np.percentile(solar_cap_all, 98),
            "cbar_label": "Mean Capacity (MW)",
            "filename": "05_solar_capacity_panel.png",
            "is_capacity": True,
            "vmax_ref": np.percentile(solar_cap_all, 98),
            "tech": "solar",
        },
    ]

    for config in metrics_config:
        fig, axes = plt.subplots(1, 4, figsize=(22, 6.6), sharex=True, sharey=True)
        for idx, period in enumerate(PERIODS):
            ax = axes[idx]
            plot_base_map(ax, boundary_gdf, raster_data, raster_extent, show_clcd_legend=False)
            overlay_cf_contours(ax, gdf, cf_df, period, config["tech"], boundary_gdf)

            val_col = gdf[f"{config['col_key']}_{period}"].values
            valid_mask = np.isfinite(val_col) & (val_col > 0)
            sc = ax.scatter(
                gdf.loc[valid_mask, "lon"],
                gdf.loc[valid_mask, "lat"],
                c=val_col[valid_mask],
                cmap=config["cmap"],
                s=scatter_style(config, val_col[valid_mask]),
                vmin=config["vmin"],
                vmax=config["vmax"],
                edgecolors="#ffffff",
                linewidths=1.2,
                alpha=1.0,
                zorder=3.2,
            )
            ax.set_title(str(period), fontweight="bold", fontsize=14)
            ax.set_xlabel("Longitude" if idx == 0 else "")
            ax.set_ylabel("Latitude" if idx == 0 else "")
            ax.grid(True, linestyle="--", alpha=0.18)

        fig.subplots_adjust(right=0.88, wspace=0.08, bottom=0.09)
        cbar_ax = fig.add_axes([0.90, 0.18, 0.015, 0.65])
        cbar = fig.colorbar(sc, cax=cbar_ax)
        cbar.set_label(config["cbar_label"], fontsize=12)
        fig.legend(
            handles=clcd_legend_handles(),
            loc="lower right",
            bbox_to_anchor=(0.885, 0.035),
            ncol=3,
            fontsize=7.2,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#d0d0d0",
            title="CLCD",
            title_fontsize=8,
            borderpad=0.35,
            columnspacing=0.8,
            handlelength=1.0,
            handletextpad=0.4,
        )
        fig.legend(
            handles=cf_legend_handles(config["tech"]),
            loc="lower right",
            bbox_to_anchor=(0.885, 0.135),
            ncol=1,
            fontsize=7.2,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#d0d0d0",
            title="CF contour",
            title_fontsize=8,
            borderpad=0.35,
            handlelength=1.8,
            handletextpad=0.55,
        )
        fig.suptitle(config["name"], fontsize=18, fontweight="bold", y=0.98)
        fig.savefig(OUTDIR / config["filename"])
        plt.close(fig)


def plot_individual_2040_maps(gdf, boundary_gdf, raster_data, raster_extent, cf_df):
    wind_cap_all = np.concatenate([gdf[f"wind_cumul_mean_{p}"].values for p in PERIODS])
    solar_cap_all = np.concatenate([gdf[f"solar_cumul_mean_{p}"].values for p in PERIODS])

    configs_2040 = [
        {
            "title": "Wind Build Frequency",
            "col_key": "wind_cumul_freq_2040",
            "cmap": "YlOrRd",
            "vmin": 0,
            "vmax": 1.0,
            "cbar_label": "Build Frequency",
            "filename": "06_wind_build_frequency_2040.png",
            "is_capacity": False,
            "tech": "wind",
        },
        {
            "title": "Solar Build Frequency",
            "col_key": "solar_cumul_freq_2040",
            "cmap": "YlOrRd",
            "vmin": 0,
            "vmax": 1.0,
            "cbar_label": "Build Frequency",
            "filename": "07_solar_build_frequency_2040.png",
            "is_capacity": False,
            "tech": "solar",
        },
        {
            "title": "Wind Capacity",
            "col_key": "wind_cumul_mean_2040",
            "cmap": "Blues",
            "vmin": 0,
            "vmax": np.percentile(wind_cap_all, 98),
            "cbar_label": "Mean Capacity (MW)",
            "filename": "08_wind_capacity_2040.png",
            "is_capacity": True,
            "vmax_ref": np.percentile(wind_cap_all, 98),
            "tech": "wind",
        },
        {
            "title": "Solar Capacity",
            "col_key": "solar_cumul_mean_2040",
            "cmap": "Oranges",
            "vmin": 0,
            "vmax": np.percentile(solar_cap_all, 98),
            "cbar_label": "Mean Capacity (MW)",
            "filename": "09_solar_capacity_2040.png",
            "is_capacity": True,
            "vmax_ref": np.percentile(solar_cap_all, 98),
            "tech": "solar",
        },
    ]

    for config in configs_2040:
        fig, ax = plt.subplots(figsize=(8.6, 8.6))
        plot_base_map(ax, boundary_gdf, raster_data, raster_extent, show_clcd_legend=False)
        overlay_cf_contours(ax, gdf, cf_df, 2040, config["tech"], boundary_gdf)
        val_col = gdf[config["col_key"]].values
        valid_mask = np.isfinite(val_col) & (val_col > 0)
        if config["is_capacity"]:
            marker_sizes = np.clip(val_col[valid_mask] / max(config["vmax_ref"], 1) * 360 + 45, 45, 560)
        else:
            marker_sizes = 135

        sc = ax.scatter(
            gdf.loc[valid_mask, "lon"],
            gdf.loc[valid_mask, "lat"],
            c=val_col[valid_mask],
            cmap=config["cmap"],
            s=marker_sizes,
            vmin=config["vmin"],
            vmax=config["vmax"],
            edgecolors="#ffffff",
            linewidths=1.15,
            alpha=0.99,
            zorder=3.2,
        )
        ax.set_title(config["title"], fontweight="bold", fontsize=15, pad=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", alpha=0.18)
        fig.subplots_adjust(bottom=0.15, right=0.88)
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(config["cbar_label"], fontsize=12)
        fig.legend(
            handles=clcd_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.42, 0.035),
            ncol=3,
            fontsize=7.2,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#d0d0d0",
            title="CLCD",
            title_fontsize=8,
            borderpad=0.35,
            columnspacing=0.8,
            handlelength=1.0,
            handletextpad=0.4,
        )
        fig.legend(
            handles=cf_legend_handles(config["tech"]),
            loc="lower right",
            bbox_to_anchor=(0.89, 0.055),
            fontsize=7.2,
            frameon=True,
            framealpha=0.96,
            facecolor="white",
            edgecolor="#d0d0d0",
            title="CF contour",
            title_fontsize=8,
            borderpad=0.35,
            handlelength=1.8,
            handletextpad=0.55,
        )
        fig.savefig(OUTDIR / config["filename"])
        plt.close(fig)


def main():
    setup_style()
    print("=" * 70)
    print("Publication-style plotting")
    print("=" * 70)

    if not SUMMARY_CSV.exists():
        print(f"ERROR: Missing raw grid results: {SUMMARY_CSV}")
        return

    success = load_results()
    cap_df = build_capacity_dataframe(success)
    gdf = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded {len(gdf)} grid records and {len(success)} successful paths.")

    plot_capacity_pathways(cap_df)
    boundary_gdf, raster_data, raster_extent = clip_clcd_ly()
    cf_df = compute_cf_mean_from_pool(gdf)
    plot_beautified_panels(gdf, boundary_gdf, raster_data, raster_extent, cf_df)
    plot_individual_2040_maps(gdf, boundary_gdf, raster_data, raster_extent, cf_df)

    print(f"Saved plots to: {OUTDIR}")


if __name__ == "__main__":
    main()



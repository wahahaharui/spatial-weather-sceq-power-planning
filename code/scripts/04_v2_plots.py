"""
SCEQ V2 Advanced Visualization & Analysis (All-English, USD)
============================================================
Plot 1: Cumulative capacity envelope (mean + P5-P95 shaded CI)
Plot 2: GIS-based grid robustness heatmap (NEW build freq + mean new capacity)
Plot 3: Beautified total cost histogram
Plot 4: Power shortage / curtailment analysis
Bonus: Convergence diagnostics (iterations histogram, gap distribution)
"""
import sys, os, json, time, gc, re, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
sys.path.insert(0, str(CODE_ROOT))
from sceq_lib.config import cfg_v2 as cfg

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUTDIR = cfg.SUMMARY_DIR

COLORS = {
    'wind': '#2e86ab',
    'solar': '#f18f01',
    'storage': '#2ecc71',
    'cost': '#6c5ce7',
}

# ===========================================================================
# Helpers
# ===========================================================================
def load_results():
    with open(cfg.CHECKPOINT_FILE, 'r') as f:
        ckpt = json.load(f)
    all_results = ckpt.get('all_results', [])
    success = [r for r in all_results if 'error' not in r and r.get('steps')]
    return all_results, success


def build_dataframe(success):
    records = []
    for r in success:
        for step in r['steps']:
            records.append({
                'path_id': r['path_id'],
                'period': step['period'],
                'wind_mw': step.get('wind_build_mw', 0),
                'solar_mw': step.get('solar_build_mw', 0),
                'storage_mw': step.get('storage_build_mw', 0),
                'total_cost': step.get('total_cost'),
            })
    return pd.DataFrame(records)


# ===========================================================================
# Plot 1: Cumulative Capacity Envelope
# ===========================================================================
def plot_capacity_envelope(df, success):
    print("[Plot 1] Cumulative capacity envelope...")

    periods = cfg.PERIODS
    n_paths = len(success)

    # Per-path cumulative trajectories 鈥?each path adds its own new builds
    # but the checkpoint already stores cumulative values; we use those directly.
    cumul = {tech: np.zeros((n_paths, len(periods)))
             for tech in ['wind', 'solar', 'storage']}

    for i, r in enumerate(success):
        pdata = df[df['path_id'] == r['path_id']].sort_values('period')
        for j, tech in enumerate(['wind', 'solar', 'storage']):
            cumul[tech][i, :] = pdata[f'{tech}_mw'].values

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for idx, (tech, label) in enumerate([('wind', 'Wind Power'),
                                          ('solar', 'Solar PV'),
                                          ('storage', 'Battery Storage')]):
        ax = axes[idx]
        data = cumul[tech]
        mean_vals = np.mean(data, axis=0)
        p5_vals = np.percentile(data, 5, axis=0)
        p95_vals = np.percentile(data, 95, axis=0)
        p25_vals = np.percentile(data, 25, axis=0)
        p75_vals = np.percentile(data, 75, axis=0)

        x = np.arange(len(periods))
        color_key = list(COLORS.keys())[idx]

        ax.fill_between(x, p5_vals, p95_vals, alpha=0.18,
                        color=COLORS[color_key], label='P5-P95')
        ax.fill_between(x, p25_vals, p75_vals, alpha=0.30,
                        color=COLORS[color_key], label='P25-P75')
        ax.plot(x, mean_vals, color=COLORS[color_key], linewidth=2.5,
                marker='o', markersize=8, markerfacecolor='white',
                markeredgewidth=2, label='Mean')

        for xi in x:
            ax.annotate(f'{mean_vals[xi]:.0f}\n[{p5_vals[xi]:.0f}, {p95_vals[xi]:.0f}]',
                       (xi, mean_vals[xi]), textcoords="offset points",
                       xytext=(0, 18), ha='center', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        ax.set_xticks(x)
        ax.set_xticklabels([str(p) for p in periods])
        ax.set_xlabel('Planning Period')
        ax.set_ylabel('Cumulative Installed Capacity (MW)')
        ax.set_title(label, fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.7)
        ax.set_xlim(-0.3, len(periods) - 0.7)

    fig.suptitle('SCEQ V2 Cumulative Capacity Evolution with Uncertainty Bands\n'
                 f'(N={n_paths} Paths, 4 Rolling Steps, gap_tol=0.005, dual-simplex + warm-start)',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(OUTDIR / '01_capacity_envelope.png', dpi=200)
    plt.close(fig)
    print(f"  -> Saved: 01_capacity_envelope.png")

    # Combined single-axis version
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    x = np.arange(len(periods))
    for idx, (tech, label) in enumerate([('wind', 'Wind'), ('solar', 'Solar'),
                                          ('storage', 'Storage')]):
        data = cumul[tech]
        mean_vals = np.mean(data, axis=0)
        p5_vals = np.percentile(data, 5, axis=0)
        p95_vals = np.percentile(data, 95, axis=0)
        color_key = list(COLORS.keys())[idx]
        ax2.fill_between(x, p5_vals, p95_vals, alpha=0.12, color=COLORS[color_key])
        ax2.plot(x, mean_vals, color=COLORS[color_key], linewidth=2.5,
                marker='o', markersize=8, markerfacecolor='white',
                markeredgewidth=2, label=f'{label} (mean + P5-P95)')

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(p) for p in periods])
    ax2.set_xlabel('Planning Period', fontsize=13)
    ax2.set_ylabel('Cumulative Installed Capacity (MW)', fontsize=13)
    ax2.set_title(f'SCEQ V2 Capacity Envelope 鈥?Wind, Solar & Storage\n'
                  f'(N={n_paths} Paths, gap_tol=0.005)',
                  fontweight='bold', fontsize=15)
    ax2.legend(loc='upper left', fontsize=11, framealpha=0.7)
    ax2.set_xlim(-0.3, len(periods) - 0.7)
    fig2.tight_layout()
    fig2.savefig(OUTDIR / '01_capacity_envelope_combined.png', dpi=200)
    plt.close(fig2)
    print(f"  -> Saved: 01_capacity_envelope_combined.png")

    return cumul


# ===========================================================================
# Plot 2: GIS Heatmap 鈥?NEW build frequency + mean NEW capacity
# ===========================================================================
def plot_gis_heatmaps(success):
    """Per-period NEW build frequency & mean new capacity heatmaps.
    Uses UNIFIED colorbar ranges across all 4 periods for each metric.
    """
    print("[Plot 2] GIS heatmap (new build per period, unified colorbars)...")

    static = pd.read_csv(cfg.STATIC_CSV)
    grid_coords = {}
    for _, row in static.iterrows():
        grid_coords[row['grid_id']] = (row['lon'], row['lat'])

    periods = cfg.PERIODS
    work_dirs_root = cfg.WORK_DIRS
    n_paths = len(success)

    # --- Phase A: Collect per-path, per-grid, per-period NEW build ---
    grid_new = {gid: {
        'wind': {p: [] for p in periods},
        'solar': {p: [] for p in periods},
    } for gid in grid_coords}

    grid_cumul = {gid: {
        'wind': {p: [] for p in periods},
        'solar': {p: [] for p in periods},
    } for gid in grid_coords}

    print(f"  Scanning gen_build.csv files...")
    t0 = time.time()

    for r in success:
        pid = r['path_id']
        path_cumul = defaultdict(lambda: {'wind': 0.0, 'solar': 0.0})

        for step in sorted(r['steps'], key=lambda s: s['period']):
            period = step['period']
            gb_file = work_dirs_root / f'path_{pid:03d}' / f'step_{period}' / 'gen_build.csv'

            period_new = defaultdict(lambda: {'wind': 0.0, 'solar': 0.0})
            if gb_file.exists():
                try:
                    gb = pd.read_csv(gb_file)
                    for _, row in gb.iterrows():
                        if int(row['PERIOD']) != period:
                            continue
                        proj = row['GENERATION_PROJECT']
                        build_val = row.get('BuildGen', '.')
                        if build_val == '.' or float(build_val) <= 1e-6:
                            continue
                        m = re.match(r'(LL_\d+)_([ws])', proj)
                        if m:
                            grid_id = m.group(1)
                            tech = 'wind' if m.group(2) == 'w' else 'solar'
                            period_new[grid_id][tech] += float(build_val)
                except Exception:
                    pass

            for grid_id in grid_coords:
                for tech in ['wind', 'solar']:
                    new_mw = period_new[grid_id][tech]
                    path_cumul[grid_id][tech] += new_mw
                    grid_new[grid_id][tech][period].append(new_mw)
                    grid_cumul[grid_id][tech][period].append(path_cumul[grid_id][tech])

    print(f"  Scanned & aggregated in {time.time()-t0:.0f}s")

    # --- Phase B: Compute statistics ---
    print("  Computing statistics...")
    grid_rows = []
    for grid_id in grid_coords:
        lon, lat = grid_coords[grid_id]
        row = {'grid_id': grid_id, 'lon': lon, 'lat': lat}
        for tech in ['wind', 'solar']:
            for period in periods:
                new_arr = np.array(grid_new[grid_id][tech][period])
                new_freq = np.mean(new_arr > 1e-6) if len(new_arr) > 0 else 0.0
                new_mean = np.mean(new_arr) if len(new_arr) > 0 else 0.0
                row[f'{tech}_new_freq_{period}'] = round(float(new_freq), 4)
                row[f'{tech}_new_mean_{period}'] = round(float(new_mean), 2)

                cum_arr = np.array(grid_cumul[grid_id][tech][period])
                cum_freq = np.mean(cum_arr > 1e-6) if len(cum_arr) > 0 else 0.0
                cum_mean = np.mean(cum_arr) if len(cum_arr) > 0 else 0.0
                row[f'{tech}_cumul_freq_{period}'] = round(float(cum_freq), 4)
                row[f'{tech}_cumul_mean_{period}'] = round(float(cum_mean), 2)
        grid_rows.append(row)

    gdf = pd.DataFrame(grid_rows)
    gdf.to_csv(OUTDIR / '02_grid_robustness_data.csv', index=False)
    print(f"  -> Saved: 02_grid_robustness_data.csv ({len(gdf)} grids)")

    # --- Phase C: Plot with UNIFIED colorbars ---
    try:
        _plot_gis_heatmaps_unified(gdf, periods, n_paths)
    except Exception as e:
        print(f"  [WARN] GIS plotting failed: {e}")
        import traceback; traceback.print_exc()

    return gdf


def _plot_gis_heatmaps_unified(gdf, periods, n_paths):
    """Plot 4-period GIS heatmaps with unified colorbar ranges."""

    wind_freq_all = np.concatenate([gdf[f'wind_new_freq_{p}'].values for p in periods])
    solar_freq_all = np.concatenate([gdf[f'solar_new_freq_{p}'].values for p in periods])
    wind_mean_all = np.concatenate([gdf[f'wind_new_mean_{p}'].values for p in periods])
    solar_mean_all = np.concatenate([gdf[f'solar_new_mean_{p}'].values for p in periods])

    freq_vmax = max(wind_freq_all.max(), solar_freq_all.max())
    freq_vmax = min(freq_vmax, np.percentile(np.concatenate([wind_freq_all, solar_freq_all]), 98))

    wind_cap_vmax = np.percentile(wind_mean_all, 98)
    solar_cap_vmax = np.percentile(solar_mean_all, 98)

    for period in periods:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Wind new build frequency
        ax = axes[0, 0]
        sc = ax.scatter(gdf['lon'], gdf['lat'],
                       c=gdf[f'wind_new_freq_{period}'], cmap='YlOrRd',
                       s=80, edgecolors='black', linewidth=0.3,
                       vmin=0, vmax=freq_vmax)
        ax.set_title(f'Wind 鈥?New Build Frequency ({period})', fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(f'Freq. (new build in {period})')

        # Solar new build frequency
        ax = axes[0, 1]
        sc = ax.scatter(gdf['lon'], gdf['lat'],
                       c=gdf[f'solar_new_freq_{period}'], cmap='YlOrRd',
                       s=80, edgecolors='black', linewidth=0.3,
                       vmin=0, vmax=freq_vmax)
        ax.set_title(f'Solar 鈥?New Build Frequency ({period})', fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(f'Freq. (new build in {period})')

        # Wind mean new capacity
        ax = axes[1, 0]
        wind_cap = gdf[f'wind_new_mean_{period}'].values
        marker_sizes = np.clip(wind_cap / max(wind_cap_vmax, 1) * 200 + 20, 20, 300)
        sc = ax.scatter(gdf['lon'], gdf['lat'],
                       c=wind_cap, cmap='Blues', s=marker_sizes,
                       edgecolors='black', linewidth=0.3,
                       vmin=0, vmax=wind_cap_vmax)
        ax.set_title(f'Wind 鈥?Mean New Capacity (MW) ({period})', fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Mean New Capacity (MW)')

        # Solar mean new capacity
        ax = axes[1, 1]
        solar_cap = gdf[f'solar_new_mean_{period}'].values
        marker_sizes = np.clip(solar_cap / max(solar_cap_vmax, 1) * 200 + 20, 20, 300)
        sc = ax.scatter(gdf['lon'], gdf['lat'],
                       c=solar_cap, cmap='Oranges', s=marker_sizes,
                       edgecolors='black', linewidth=0.3,
                       vmin=0, vmax=solar_cap_vmax)
        ax.set_title(f'Solar 鈥?Mean New Capacity (MW) ({period})', fontweight='bold')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label('Mean New Capacity (MW)')

        fig.suptitle(f'Grid-Level New Build Pattern 鈥?{period}\n'
                     f'(N={n_paths} SCEQ V2 Paths, 222 Grids @ 10km, Unified Color Scales)',
                     fontsize=15, fontweight='bold')
        fig.tight_layout()
        fig.savefig(OUTDIR / f'02_gis_heatmap_{period}.png', dpi=200)
        plt.close(fig)
        print(f"  -> Saved: 02_gis_heatmap_{period}.png "
              f"(freq vmax={freq_vmax:.2f}, w_cap vmax={wind_cap_vmax:.0f}, "
              f"s_cap vmax={solar_cap_vmax:.0f})")


# ===========================================================================
# Plot 3: Total Cost Histogram (USD)
# ===========================================================================
def plot_cost_histogram(success):
    """Professional cost histogram with KDE 鈥?all units in USD."""
    print("[Plot 3] Total cost histogram...")

    path_costs = []
    for r in success:
        total_cost = 0
        for step in r['steps']:
            if step.get('total_cost') and np.isfinite(step['total_cost']):
                total_cost = step['total_cost']
        if total_cost > 0:
            path_costs.append(total_cost / 1e9)

    costs = np.array(path_costs)
    if len(costs) == 0:
        print("  No cost data!")
        return

    mean_cost = np.mean(costs)
    median_cost = np.median(costs)
    p5 = np.percentile(costs, 5)
    p95 = np.percentile(costs, 95)
    std_cost = np.std(costs)

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.hist(costs, bins=35, density=True,
            color=COLORS['cost'], alpha=0.55,
            edgecolor='white', linewidth=0.8)

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(costs)
    x_smooth = np.linspace(costs.min() - 0.5 * std_cost,
                           costs.max() + 0.5 * std_cost, 300)
    ax.plot(x_smooth, kde(x_smooth), color='#2d0066', linewidth=2.5,
            alpha=0.85, label='KDE Density')

    ax.axvline(mean_cost, color='#e74c3c', linestyle='--', linewidth=2,
               label=f'Mean: ${mean_cost:.2f}B')
    ax.axvline(median_cost, color='#e67e22', linestyle='--', linewidth=2,
               label=f'Median (P50): ${median_cost:.2f}B')
    ax.axvline(p5, color='#95a5a6', linestyle=':', linewidth=1.5,
               label=f'P5: ${p5:.2f}B')
    ax.axvline(p95, color='#95a5a6', linestyle=':', linewidth=1.5,
               label=f'P95: ${p95:.2f}B')
    ax.axvspan(p5, p95, alpha=0.08, color='gray',
               label=f'P5-P95 Spread: ${p95 - p5:.2f}B')

    ax.set_xlabel('Total System Cost (Billion USD)', fontsize=13)
    ax.set_ylabel('Probability Density', fontsize=13)
    ax.set_title(f'SCEQ V2 Total Cost Distribution\n'
                 f'Mean=${mean_cost:.2f}B  |  Std=${std_cost:.2f}B  |  '
                 f'CV={std_cost/mean_cost*100:.1f}%  |  '
                 f'P5-P95 Range=${p95-p5:.2f}B',
                 fontweight='bold', fontsize=14)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.8, ncol=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:.1f}B'))

    fig.tight_layout()
    fig.savefig(OUTDIR / '03_cost_histogram.png', dpi=200)
    plt.close(fig)
    print(f"  -> Saved: 03_cost_histogram.png")

    stats = {
        'mean_billion_usd': round(float(mean_cost), 3),
        'std_billion_usd': round(float(std_cost), 3),
        'cv_pct': round(float(std_cost/mean_cost*100), 2),
        'p5_billion_usd': round(float(p5), 3),
        'p50_billion_usd': round(float(median_cost), 3),
        'p95_billion_usd': round(float(p95), 3),
        'min_billion_usd': round(float(costs.min()), 3),
        'max_billion_usd': round(float(costs.max()), 3),
        'p95_p5_spread_billion_usd': round(float(p95-p5), 3),
    }
    with open(OUTDIR / '03_cost_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  -> Saved: 03_cost_stats.json")


# ===========================================================================
# Plot 4: Power Shortage / Reliability Analysis
# ===========================================================================
def plot_shortage_analysis(success):
    """Extract and visualize power shortage / unserved energy from dispatch data."""
    print("[Plot 4] Shortage / reliability analysis...")

    work_dirs_root = cfg.WORK_DIRS
    periods = cfg.PERIODS

    sample_dispatch = work_dirs_root / 'path_000' / 'step_2025' / 'dispatch.csv'
    if not sample_dispatch.exists():
        print("  No dispatch.csv found 鈥?skipping.")
        _note_shortage_unavailable()
        return

    print("  Computing shortage statistics (50-path sample)...")
    sample_paths = success[:min(50, len(success))]

    shortage_records = []
    for r in sample_paths:
        pid = r['path_id']
        for step in r['steps']:
            period = step['period']
            step_dir = work_dirs_root / f'path_{pid:03d}' / f'step_{period}'
            d_file = step_dir / 'dispatch.csv'
            l_file = step_dir / 'loads.csv'
            if not d_file.exists() or not l_file.exists():
                continue

            try:
                dispatch = pd.read_csv(d_file)
                loads = pd.read_csv(l_file)
                dispatch_sum = dispatch.groupby('TIMEPOINT')['DispatchGen'].sum()
                merged = loads.merge(
                    dispatch_sum.reset_index().rename(
                        columns={'DispatchGen': 'total_dispatch'}),
                    on='TIMEPOINT', how='left'
                )
                merged['total_dispatch'] = merged['total_dispatch'].fillna(0)
                merged['shortage_mw'] = np.maximum(
                    0, merged['zone_demand_mw'] - merged['total_dispatch'])
                merged['shortage_ratio'] = (
                    merged['shortage_mw'] / merged['zone_demand_mw'].clip(lower=1))

                shortage_records.append({
                    'path_id': pid,
                    'period': period,
                    'avg_shortage_ratio': float(merged['shortage_ratio'].mean()),
                    'max_shortage_ratio': float(merged['shortage_ratio'].max()),
                    'total_shortage_mwh': float(merged['shortage_mw'].sum()),
                    'total_load_mwh': float(merged['zone_demand_mw'].sum()),
                })
            except Exception:
                pass

    if not shortage_records:
        print("  No shortage data extracted.")
        _note_shortage_unavailable()
        return

    sdf = pd.DataFrame(shortage_records)
    sdf.to_csv(OUTDIR / '04_shortage_data.csv', index=False)
    print(f"  -> Saved: 04_shortage_data.csv ({len(sdf)} records)")

    _plot_shortage(sdf, periods)


def _plot_shortage(sdf, periods):
    """Shortage rate boxplots + bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    colors_box = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    ax = axes[0]
    data_by_period = [sdf[sdf['period'] == p]['avg_shortage_ratio'].values * 100
                      for p in periods]
    bp = ax.boxplot(data_by_period, labels=[str(p) for p in periods],
                     patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors_box[:len(periods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel('Average Shortage Rate (%)')
    ax.set_xlabel('Period')
    ax.set_title('Power Shortage Rate Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    means = [sdf[sdf['period'] == p]['avg_shortage_ratio'].mean() * 100
             for p in periods]
    stds = [sdf[sdf['period'] == p]['avg_shortage_ratio'].std() * 100
            for p in periods]
    x = np.arange(len(periods))
    bars = ax.bar(x, means, yerr=stds, color=colors_box[:len(periods)],
                  alpha=0.7, edgecolor='black', linewidth=0.8, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in periods])
    ax.set_ylabel('Mean Shortage Rate (%)')
    ax.set_xlabel('Period')
    ax.set_title('Mean Shortage Rate + Std Dev', fontweight='bold')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{mean:.3f}%', ha='center', va='bottom', fontsize=9)

    fig.suptitle('SCEQ V2 Power Supply Reliability Analysis\n'
                 '(Unserved Energy / Total Load, sampled 50 paths)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / '04_shortage_analysis.png', dpi=200)
    plt.close(fig)
    print(f"  -> Saved: 04_shortage_analysis.png")


def _note_shortage_unavailable():
    with open(OUTDIR / '04_shortage_note.txt', 'w') as f:
        f.write("Power shortage analysis was not available.\n")


# ===========================================================================
# Convergence Report (V2: gap_tol=0.005, max_iter=200)
# ===========================================================================
def write_convergence_report(df, success):
    print("[Report] Writing convergence report (V2, USD)...")

    all_iters, all_cuts, all_times, all_gaps = [], [], [], []
    for r in success:
        for step in r['steps']:
            info = step.get('solve_info', {})
            all_iters.append(info.get('iterations', 0))
            all_cuts.append(info.get('total_cuts', 0))
            all_times.append(info.get('solve_s', 0))
            g = info.get('gap')
            if g is not None and np.isfinite(g):
                all_gaps.append(g)

    total_cpu_h = sum(all_times) / 3600
    n_workers = cfg.N_WORKERS
    est_wall_h = total_cpu_h / n_workers

    gaps_arr = np.array(all_gaps)
    iters_arr = np.array(all_iters)

    with open(OUTDIR / 'convergence_report.txt', 'w') as f:
        f.write("SCEQ V2 Stochastic Planning 鈥?Convergence Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Paths: {cfg.N_PATHS}  |  Periods: {len(cfg.PERIODS)}  |  "
                f"P(extreme): {cfg.P_EXTREME}\n")
        f.write(f"Workers: {n_workers} (parallel)  |  ThreadPool: 4\n")
        f.write(f"Gap tolerance: {cfg.BENDERS_GAP_TOL}  |  Max iterations: {cfg.BENDERS_MAX_ITER}\n")
        f.write(f"Method: Dual Simplex (Method=1)  |  Warm-start: ON\n")
        f.write(f"Cost currency: USD (Switch model default)\n\n")

        # Capacity
        f.write("--- Cumulative Capacity Investment Statistics ---\n")
        periods = cfg.PERIODS
        cumul_wind, cumul_solar, cumul_storage = defaultdict(list), defaultdict(list), defaultdict(list)
        for r in success:
            r_df = pd.DataFrame(r['steps'])
            for _, step in r_df.sort_values('period').iterrows():
                cumul_wind[step['period']].append(step.get('wind_build_mw', 0))
                cumul_solar[step['period']].append(step.get('solar_build_mw', 0))
                cumul_storage[step['period']].append(step.get('storage_build_mw', 0))

        for period in periods:
            f.write(f"\nPeriod {period} (Cumulative):\n")
            for tech_name, vals in [('Wind (MW)', cumul_wind[period]),
                                     ('Solar (MW)', cumul_solar[period]),
                                     ('Storage (MW)', cumul_storage[period])]:
                if vals:
                    arr = np.array(vals)
                    f.write(f"  {tech_name}: mean={np.mean(arr):.0f} "
                            f"std={np.std(arr):.0f} "
                            f"P5={np.percentile(arr,5):.0f} "
                            f"P95={np.percentile(arr,95):.0f}\n")

        # Total cost (USD)
        path_costs = []
        for r in success:
            tc = 0
            for step in r['steps']:
                if step.get('total_cost') and np.isfinite(step['total_cost']):
                    tc = step['total_cost']
            if tc > 0:
                path_costs.append(tc)
        if path_costs:
            costs = np.array(path_costs)
            f.write(f"\nTotal Cost (USD): mean=${np.mean(costs):,.0f} "
                    f"std=${np.std(costs):,.0f} "
                    f"P5=${np.percentile(costs,5):,.0f} "
                    f"P95=${np.percentile(costs,95):,.0f}\n")
            f.write(f"  Billion USD: mean=${np.mean(costs)/1e9:.2f}B "
                    f"(~{np.mean(costs)/1e9:.1f} Billion USD)\n")

        # Solve statistics
        f.write(f"\n--- Solve Statistics ---\n")
        f.write(f"Total Benders solves: {len(all_iters)}\n")
        f.write(f"Average iterations:   {np.mean(all_iters):.1f} (median={np.median(all_iters):.0f})\n")
        f.write(f"At max_iter ({cfg.BENDERS_MAX_ITER}):    {(iters_arr >= cfg.BENDERS_MAX_ITER).sum()} ({(iters_arr >= cfg.BENDERS_MAX_ITER).mean()*100:.1f}%)\n")
        f.write(f"Average cuts:         {np.mean(all_cuts):.1f}\n")
        f.write(f"Average solve time:   {np.mean(all_times):.1f}s per solve\n")
        f.write(f"\n")
        f.write(f"Total CPU time:       {total_cpu_h:.1f}h (sum of all solves)\n")
        f.write(f"Est. wall-clock time: ~{est_wall_h:.1f}h ({n_workers} workers)\n")
        f.write(f"Speedup factor:       {total_cpu_h/est_wall_h:.1f}x\n\n")

        # Gap distribution
        f.write(f"--- Benders Gap Distribution ---\n")
        f.write(f"Gap samples:          {len(gaps_arr)}\n")
        f.write(f"Gap mean/median:      {gaps_arr.mean():.4f} / {np.median(gaps_arr):.4f}\n")
        for lo, hi in [(0, 0.002), (0.002, 0.005), (0.005, 0.01),
                       (0.01, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 1.0)]:
            n = ((gaps_arr >= lo) & (gaps_arr < hi)).sum()
            f.write(f"  [{lo:.3f}, {hi:.3f}): {n} ({n/len(gaps_arr)*100:.1f}%)\n")
        f.write(f"\nNOTE: gap<{cfg.BENDERS_GAP_TOL} achieved in {(gaps_arr < cfg.BENDERS_GAP_TOL).sum()}/"
                f"{len(gaps_arr)} ({(gaps_arr < cfg.BENDERS_GAP_TOL).mean()*100:.1f}%) solves.\n")
        f.write(f"V2 improvements: dual simplex + warm-start + thread optimization.\n")

        f.write(f"\n--- Performance Note ---\n")
        f.write(f"\"Total CPU time\" is the SUM of individual solve times across\n")
        f.write(f"all {len(all_iters)} Benders runs. With {n_workers} parallel workers,\n")
        f.write(f"the actual wall-clock time is approximately CPU_time / {n_workers}.\n")

    print(f"  -> Saved: convergence_report.txt")


# ===========================================================================
# Bonus: Convergence diagnostics
# ===========================================================================
def plot_convergence_diagnostics(success):
    print("[Bonus] Convergence diagnostics...")

    all_infos = []
    for r in success:
        for step in r['steps']:
            info = step.get('solve_info', {})
            all_infos.append({
                'path_id': r['path_id'],
                'period': step['period'],
                'iterations': info.get('iterations', 0),
                'cuts': info.get('total_cuts', 0),
                'solve_s': info.get('solve_s', 0),
                'gap': info.get('gap', 1.0),
            })
    cinfo = pd.DataFrame(all_infos)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Iterations histogram
    ax = axes[0, 0]
    ax.hist(cinfo['iterations'], bins=30, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(cinfo['iterations'].mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean: {cinfo["iterations"].mean():.1f}')
    ax.axvline(cfg.BENDERS_MAX_ITER, color='black', linestyle=':', linewidth=1.5,
               label=f'max_iter={cfg.BENDERS_MAX_ITER}')
    ax.set_xlabel('Iterations'); ax.set_ylabel('Count')
    ax.set_title('Benders Iteration Distribution', fontweight='bold')
    ax.legend()

    # Gap histogram
    ax = axes[0, 1]
    gaps_plot = cinfo['gap'].dropna()
    gaps_plot = gaps_plot[gaps_plot < 0.05]
    ax.hist(gaps_plot * 100, bins=40, color='#e74c3c', alpha=0.7, edgecolor='white')
    ax.axvline(cfg.BENDERS_GAP_TOL * 100, color='green', linestyle='--', linewidth=2,
               label=f'target gap={cfg.BENDERS_GAP_TOL*100:.1f}%')
    ax.axvline(gaps_plot.median() * 100, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {gaps_plot.median()*100:.2f}%')
    ax.set_xlabel('Optimality Gap (%)'); ax.set_ylabel('Count')
    ax.set_title('Benders Gap Distribution (gaps < 5%)', fontweight='bold')
    ax.legend()

    # Solve time over paths
    ax = axes[1, 0]
    path_agg = cinfo.groupby('path_id').agg(
        {'iterations': 'mean', 'solve_s': 'mean', 'gap': 'mean'}).reset_index()
    ax.scatter(path_agg['path_id'], path_agg['solve_s'],
               alpha=0.5, s=20, color='#f39c12')
    ax.axhline(cinfo['solve_s'].mean(), color='blue', linestyle='--', alpha=0.6,
               label=f'Mean: {cinfo["solve_s"].mean():.0f}s')
    ax.set_xlabel('Path ID'); ax.set_ylabel('Avg Solve Time (s)')
    ax.set_title('Mean Solve Time per Path', fontweight='bold')
    ax.legend()

    # Gap over paths
    ax = axes[1, 1]
    ax.scatter(path_agg['path_id'], path_agg['gap'] * 100,
               alpha=0.5, s=20, color='#e74c3c')
    ax.axhline(cfg.BENDERS_GAP_TOL * 100, color='green', linestyle='--', alpha=0.7,
               label=f'target={cfg.BENDERS_GAP_TOL*100:.1f}%')
    ax.axhline(cinfo['gap'].median() * 100, color='orange', linestyle='--', alpha=0.7,
               label=f'Median: {cinfo["gap"].median()*100:.2f}%')
    ax.set_xlabel('Path ID'); ax.set_ylabel('Avg Gap (%)')
    ax.set_title('Mean Optimality Gap per Path', fontweight='bold')
    ax.legend()

    fig.suptitle('SCEQ V2 Benders Convergence Diagnostics', fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTDIR / 'convergence_diagnostics.png', dpi=200)
    plt.close(fig)
    print(f"  -> Saved: convergence_diagnostics.png")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("=" * 70)
    print("SCEQ V2 Advanced Visualization & Analysis (USD, English)")
    print("=" * 70)

    t0 = time.time()
    all_results, success = load_results()
    n_paths = len(success)
    print(f"Loaded {n_paths} successful paths")

    if n_paths == 0:
        print("ERROR: No successful paths.")
        return

    df = build_dataframe(success)

    plot_capacity_envelope(df, success)
    plot_gis_heatmaps(success)
    plot_cost_histogram(success)
    plot_shortage_analysis(success)
    write_convergence_report(df, success)
    plot_convergence_diagnostics(success)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"All V2 plots complete! ({elapsed:.0f}s)")
    print(f"Output: {OUTDIR}/")
    for f in sorted(os.listdir(OUTDIR)):
        fpath = OUTDIR / f
        if fpath.is_file():
            size_kb = fpath.stat().st_size / 1024
            print(f"  {f} ({size_kb:.0f} KB)")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()



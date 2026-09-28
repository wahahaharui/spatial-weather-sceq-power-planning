"""
算法效率对比 — 可视化图表 (含Switch原始MIP)
生成 Runtime三方法对比、加速比、成本偏差图等
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COMPS = ['C01', 'C02', 'C03', 'C04', 'C05']
x = np.arange(len(COMPS))
w = 0.25  # narrower for 3 bars

# --- 数据 ---
switch_time    = [342, 356, 398, 370, 350]
single_cut_time  = [284.6, 284.9, 307.2, 298.5, 263.8]
multi_cut_time   = [88, 75, 94, 74, 70]
single_cut_iters = [171, 200, 186, 180, 178]
multi_cut_iters  = [145, 127, 161, 129, 126]
single_cut_gap   = [0.176, 0.308, 0.152, 0.200, 0.182]
multi_cut_gap    = [0.18, 0.34, 0.36, 0.17, 0.20]
single_cut_dev   = [-0.311, -0.329, -0.293, -0.321, -0.250]
multi_cut_dev    = [-0.233, -0.198, -0.248, -0.243, -0.251]

col_sw = '#70AD47'  # green
col_sc = '#4472C4'  # blue
col_mc = '#ED7D31'  # orange

# ============================================================
# Fig1: Runtime 三方法对比 (含Switch)
# ============================================================
fig1, ax = plt.subplots(figsize=(11, 6))
ax.bar(x - w, switch_time, w, label='Switch Original (CPLEX MIP)', color=col_sw, edgecolor='white')
ax.bar(x, single_cut_time, w, label='Single-Cut Benders (V1)', color=col_sc, edgecolor='white')
ax.bar(x + w, multi_cut_time, w, label='Multi-Cut Benders (V6)', color=col_mc, edgecolor='white')

# Annotate speedup
for i in range(len(COMPS)):
    sp = switch_time[i] / multi_cut_time[i]
    y = max(switch_time[i], single_cut_time[i], multi_cut_time[i])
    ax.annotate(f'SW→MC: {sp:.1f}x', xy=(i, y + 10), ha='center', fontsize=9, fontweight='bold', color='#C00000')
    sp2 = single_cut_time[i] / multi_cut_time[i]
    ax.annotate(f'SC→MC: {sp2:.1f}x', xy=(i, y + 28), ha='center', fontsize=8, color='#7030A0')

ax.set_xticks(x); ax.set_xticklabels(COMPS, fontsize=12)
ax.set_ylabel('Runtime (seconds)', fontsize=13)
ax.set_title('Solving Time Comparison: Switch MIP vs Benders Decomposition', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(switch_time) * 1.35)
fig1.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'fig1_runtime_comparison.png'), dpi=150, bbox_inches='tight')
fig1.savefig(os.path.join(OUT_DIR, 'fig1_runtime_comparison.pdf'), bbox_inches='tight')
plt.close(fig1); print("[OK] Fig1: Runtime (3-method) saved.")

# ============================================================
# Fig2: Iterations + GAP (SC vs MC)
# ============================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.bar(x - w/2, single_cut_iters, w*1.2, label='Single-Cut', color=col_sc, edgecolor='white')
ax1.bar(x + w/2, multi_cut_iters, w*1.2, label='Multi-Cut', color=col_mc, edgecolor='white')
for i in range(len(COMPS)):
    red = single_cut_iters[i] - multi_cut_iters[i]
    ax1.annotate(f'-{red}', xy=(i, max(single_cut_iters[i], multi_cut_iters[i]) + 3),
                 ha='center', fontsize=9, fontweight='bold', color='#C00000')
ax1.set_xticks(x); ax1.set_xticklabels(COMPS)
ax1.set_ylabel('Iterations', fontsize=12)
ax1.set_title('(a) Benders Iterations', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3, linestyle='--')

ax2.bar(x - w/2, single_cut_gap, w*1.2, label='Single-Cut', color=col_sc, edgecolor='white')
ax2.bar(x + w/2, multi_cut_gap, w*1.2, label='Multi-Cut', color=col_mc, edgecolor='white')
ax2.axhline(y=0.2, color='gray', linewidth=0.8, linestyle='--', alpha=0.7)
ax2.text(len(COMPS)-0.6, 0.205, 'MIPGap=0.2%', fontsize=8, color='gray')
ax2.set_xticks(x); ax2.set_xticklabels(COMPS)
ax2.set_ylabel('GAP (%)', fontsize=12)
ax2.set_title('(b) Convergence GAP', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.3, linestyle='--')

fig2.suptitle('Benders Convergence: Iterations & Solution Quality', fontsize=14, fontweight='bold')
fig2.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'fig2_iterations_gap.png'), dpi=150, bbox_inches='tight')
fig2.savefig(os.path.join(OUT_DIR, 'fig2_iterations_gap.pdf'), bbox_inches='tight')
plt.close(fig2); print("[OK] Fig2: Iterations & GAP saved.")

# ============================================================
# Fig3: 成本偏差 + 加速比面板
# ============================================================
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.bar(x - w/2, single_cut_dev, w*1.2, label='Single-Cut', color=col_sc, edgecolor='white')
ax1.bar(x + w/2, multi_cut_dev, w*1.2, label='Multi-Cut', color=col_mc, edgecolor='white')
avg_sc = np.mean(single_cut_dev); avg_mc = np.mean(multi_cut_dev)
ax1.axhline(y=avg_sc, color=col_sc, linewidth=1, linestyle='--', alpha=0.5)
ax1.axhline(y=avg_mc, color=col_mc, linewidth=1, linestyle='--', alpha=0.5)
ax1.set_xticks(x); ax1.set_xticklabels(COMPS)
ax1.set_ylabel('Cost Deviation from Switch (%)', fontsize=12)
ax1.set_title('(a) Solution Quality: Cost Deviation', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10); ax1.grid(axis='y', alpha=0.3, linestyle='--')

speedup_sw_mc = [switch_time[i]/multi_cut_time[i] for i in range(5)]
speedup_sc_mc = [single_cut_time[i]/multi_cut_time[i] for i in range(5)]
speedup_sw_sc = [switch_time[i]/single_cut_time[i] for i in range(5)]
ax2.bar(x - w, speedup_sw_mc, w, label='SW→MC', color=col_mc, edgecolor='white')
ax2.bar(x, speedup_sc_mc, w, label='SC→MC', color=col_sc, edgecolor='white')
ax2.bar(x + w, speedup_sw_sc, w, label='SW→SC', color=col_sw, edgecolor='white')
ax2.set_xticks(x); ax2.set_xticklabels(COMPS)
ax2.set_ylabel('Speedup (x times)', fontsize=12)
ax2.set_title('(b) Computational Speedup', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10); ax2.grid(axis='y', alpha=0.3, linestyle='--')

fig3.suptitle('Solution Quality vs Computational Efficiency', fontsize=14, fontweight='bold')
fig3.tight_layout()
fig3.savefig(os.path.join(OUT_DIR, 'fig3_quality_speedup.png'), dpi=150, bbox_inches='tight')
fig3.savefig(os.path.join(OUT_DIR, 'fig3_quality_speedup.pdf'), bbox_inches='tight')
plt.close(fig3); print("[OK] Fig3: Quality & Speedup panel saved.")

# ============================================================
# Fig4: 综合四合一面板
# ============================================================
fig4, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) Runtime all 3 methods
ax = axes[0, 0]
ax.bar(x - w, switch_time, w, label='Switch MIP', color=col_sw, edgecolor='white')
ax.bar(x, single_cut_time, w, label='Single-Cut', color=col_sc, edgecolor='white')
ax.bar(x + w, multi_cut_time, w, label='Multi-Cut', color=col_mc, edgecolor='white')
for i in range(5):
    sp = switch_time[i] / multi_cut_time[i]
    ax.annotate(f'{sp:.1f}x', xy=(i, switch_time[i] + 5), ha='center', fontsize=7, fontweight='bold', color='#C00000')
ax.set_xticks(x); ax.set_xticklabels(COMPS)
ax.set_ylabel('Seconds', fontsize=11)
ax.set_title('(a) Runtime Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3, linestyle='--')

# (b) Iterations
ax = axes[0, 1]
ax.bar(x - w/2, single_cut_iters, w*1.2, label='Single-Cut', color=col_sc, edgecolor='white')
ax.bar(x + w/2, multi_cut_iters, w*1.2, label='Multi-Cut', color=col_mc, edgecolor='white')
for i in range(5):
    ax.annotate(f'-{single_cut_iters[i]-multi_cut_iters[i]}', xy=(i, max(single_cut_iters[i], multi_cut_iters[i]) + 2),
                ha='center', fontsize=8, fontweight='bold', color='#C00000')
ax.set_xticks(x); ax.set_xticklabels(COMPS)
ax.set_ylabel('Iterations', fontsize=11)
ax.set_title('(b) Benders Iterations', fontsize=12, fontweight='bold')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3, linestyle='--')

# (c) Cost deviation
ax = axes[1, 0]
ax.axhline(y=0, color='black', linewidth=0.8)
ax.bar(x - w/2, single_cut_dev, w*1.2, label='Single-Cut', color=col_sc, edgecolor='white')
ax.bar(x + w/2, multi_cut_dev, w*1.2, label='Multi-Cut', color=col_mc, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(COMPS)
ax.set_ylabel('Deviation (%)', fontsize=11)
ax.set_title('(c) Cost Deviation from Switch', fontsize=12, fontweight='bold')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3, linestyle='--')

# (d) GAP
ax = axes[1, 1]
ax.bar(x - w/2, single_cut_gap, w*1.2, label='Single-Cut', color=col_sc, edgecolor='white')
ax.bar(x + w/2, multi_cut_gap, w*1.2, label='Multi-Cut', color=col_mc, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(COMPS)
ax.set_ylabel('GAP (%)', fontsize=11)
ax.set_title('(d) Benders Convergence GAP', fontsize=12, fontweight='bold')
ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3, linestyle='--')

fig4.suptitle('Algorithm Efficiency: Switch MIP vs Single-Cut vs Multi-Cut Benders',
              fontsize=15, fontweight='bold', y=1.01)
fig4.tight_layout()
fig4.savefig(os.path.join(OUT_DIR, 'fig4_comprehensive_panel.png'), dpi=150, bbox_inches='tight')
fig4.savefig(os.path.join(OUT_DIR, 'fig4_comprehensive_panel.pdf'), bbox_inches='tight')
plt.close(fig4); print("[OK] Fig4: Comprehensive panel saved.")

# ============================================================
# Fig5: Speedup bar chart (3 comparisons)
# ============================================================
fig5, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - w, speedup_sw_mc, w, label='Switch → Multi-Cut', color=col_mc, edgecolor='white')
ax.bar(x, speedup_sc_mc, w, label='Single-Cut → Multi-Cut', color=col_sc, edgecolor='white')
ax.bar(x + w, speedup_sw_sc, w, label='Switch → Single-Cut', color=col_sw, edgecolor='white')
# value labels
for i in range(5):
    ax.text(i - w, speedup_sw_mc[i] + 0.05, f'{speedup_sw_mc[i]:.1f}x', ha='center', fontsize=9, fontweight='bold', color=col_mc)
    ax.text(i, speedup_sc_mc[i] + 0.05, f'{speedup_sc_mc[i]:.1f}x', ha='center', fontsize=9, fontweight='bold', color=col_sc)
    ax.text(i + w, speedup_sw_sc[i] + 0.05, f'{speedup_sw_sc[i]:.1f}x', ha='center', fontsize=9, fontweight='bold', color=col_sw)
ax.set_xticks(x); ax.set_xticklabels(COMPS, fontsize=12)
ax.set_ylabel('Speedup (x times)', fontsize=13)
ax.set_title('Computational Speedup Across Methods', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
fig5.tight_layout()
fig5.savefig(os.path.join(OUT_DIR, 'fig5_speedup_comparison.png'), dpi=150, bbox_inches='tight')
fig5.savefig(os.path.join(OUT_DIR, 'fig5_speedup_comparison.pdf'), bbox_inches='tight')
plt.close(fig5); print("[OK] Fig5: Speedup comparison saved.")

print(f"\n=== All charts generated successfully! ===")

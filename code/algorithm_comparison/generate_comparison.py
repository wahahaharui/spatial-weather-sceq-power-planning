"""
算法效率对比分析: Switch原始MIP vs Single-Cut Benders vs Multi-Cut Benders
生成对比表格、统计数据和可视化图表
数据来源:
  - Switch: 0607_comp_0X_outputs/total_cost.txt + compare_switch_vs_benders.py (runtime)
  - Single-Cut: benders_fixed/batch_comparison.json + batch_results.json
  - Multi-Cut: benders_fixed_v6/output_comp_0X/total_cost.txt + V6_README.md
"""
import os, json
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPS = ['comp_01', 'comp_02', 'comp_03', 'comp_04', 'comp_05']

# ============================================================
# 1. 数据汇总
# ============================================================

# Switch 原始 MIP (CPLEX 12.8, mipgap=0.002, threads=8)
switch_cost = {
    'comp_01': 16680006757.82,
    'comp_02': 16607982022.49,
    'comp_03': 16556304489.10,
    'comp_04': 16550988975.44,
    'comp_05': 16744047432.19,
}
# Runtime from model_tests/compare_switch_vs_benders.py:369 (approximate from CPLEX logs)
switch_time = {
    'comp_01': 342, 'comp_02': 356, 'comp_03': 398, 'comp_04': 370, 'comp_05': 350,
}

# Single-Cut Benders (V1): benders_fixed/batch_comparison.json + batch_results.json
single_cut = {
    'comp_01': {'cost': 16628105983.87, 'time_s': 284.6, 'iters': 171, 'gap_pct': 0.176},
    'comp_02': {'cost': 16553339453.78, 'time_s': 284.9, 'iters': 200, 'gap_pct': 0.308},
    'comp_03': {'cost': 16507717789.08, 'time_s': 307.2, 'iters': 186, 'gap_pct': 0.152},
    'comp_04': {'cost': 16497906272.24, 'time_s': 298.5, 'iters': 180, 'gap_pct': 0.200},
    'comp_05': {'cost': 16702207198.52, 'time_s': 263.8, 'iters': 178, 'gap_pct': 0.182},
}

# Multi-Cut Benders (V6): benders_fixed_v6/output_comp_0X/total_cost.txt + V6_README
multi_cut = {
    'comp_01': {'cost': 16641115003.11, 'time_s': 88, 'iters': 145, 'gap_pct': 0.18},
    'comp_02': {'cost': 16575027776.50, 'time_s': 75, 'iters': 127, 'gap_pct': 0.34},
    'comp_03': {'cost': 16515207892.29, 'time_s': 94, 'iters': 161, 'gap_pct': 0.36},
    'comp_04': {'cost': 16510801597.83, 'time_s': 74, 'iters': 129, 'gap_pct': 0.17},
    'comp_05': {'cost': 16702071431.22, 'time_s': 70, 'iters': 126, 'gap_pct': 0.20},
}

# ============================================================
# 2. 求解器参数配置对比
# ============================================================
solver_configs = {
    'Switch (CPLEX MIP)': {
        '求解器': 'CPLEX 12.8.0',
        '求解方式': '单次 MIP (含 Binary DR)',
        'MIP Gap / 收敛容差': '0.002 (0.2%)',
        '线程数': '8',
        '模型类型': 'MIP (Binary DR)',
        '变量规模': '~530,000 (reduced)',
        '约束规模': '~526,000 (reduced)',
        '子问题结构': 'N/A (一体化求解)',
        '求解算法': 'Branch-and-Cut (CPLEX 默认)',
    },
    'Single-Cut Benders (V1)': {
        '求解器': 'Gurobi (Persistent LP)',
        '求解方式': 'Benders 分解 — 单割 (Aggregate Cut)',
        'MIP Gap / 收敛容差': 'gap_tol=0.002 (0.2%)',
        '线程数': 'Master: 8, Sub: 1 each',
        '模型类型': 'LP (DR 连续松弛)',
        '变量规模': 'Master ~5,000, 每Sub ~2,000-4,000',
        '约束规模': 'Master ~3,000 + Cuts',
        '子问题结构': '4 个全周期 LP (288 tp/个)',
        '求解算法': 'Primal Simplex (Sub), Dual Simplex (Master)',
    },
    'Multi-Cut Benders (V6)': {
        '求解器': 'Gurobi (Persistent LP)',
        '求解方式': 'Benders 分解 — 多割 (Multi-Cut + Hybrid Dual-θ)',
        'MIP Gap / 收敛容差': 'gap_tol=0.002 (0.2%)',
        '线程数': 'Master: 8, 48 Segments×ThreadPool(8)',
        '模型类型': 'LP (DR 连续松弛)',
        '变量规模': 'Master ~6,600, 每Seg ~500-2,000',
        '约束规模': 'Master ~80 + Cuts, 每Seg ~1,000-3,000',
        '子问题结构': '48 个独立 Segment LP (24 tp/个)',
        '求解算法': 'Primal Simplex (Sub), 并行 8 workers',
    },
}

# ============================================================
# 3. 计算汇总指标
# ============================================================
def compute_metrics():
    rows = []
    for comp in COMPS:
        sc = single_cut[comp]
        mc = multi_cut[comp]
        sw_c = switch_cost[comp]
        sw_t = switch_time[comp]

        sc_dev_pct = (sc['cost'] - sw_c) / sw_c * 100
        mc_dev_pct = (mc['cost'] - sw_c) / sw_c * 100

        rows.append({
            '场景': comp,
            'Switch_Cost_B': sw_c / 1e9,
            'Switch_Time_s': sw_t,
            'SC_Cost_B': sc['cost'] / 1e9,
            'SC_Dev_pct': sc_dev_pct,
            'SC_Time_s': sc['time_s'],
            'SC_Iters': sc['iters'],
            'SC_GAP_pct': sc['gap_pct'],
            'MC_Cost_B': mc['cost'] / 1e9,
            'MC_Dev_pct': mc_dev_pct,
            'MC_Time_s': mc['time_s'],
            'MC_Iters': mc['iters'],
            'MC_GAP_pct': mc['gap_pct'],
            'Speedup_SWvsMC': sw_t / mc['time_s'],
            'Speedup_SWvsSC': sw_t / sc['time_s'],
            'Speedup_SCvsMC': sc['time_s'] / mc['time_s'],
        })

    df = pd.DataFrame(rows)

    # 平均值行
    avg = {}
    for col in df.columns:
        if col == '场景':
            avg[col] = 'AVERAGE'
        elif 'Speedup_' in col:
            # 使用 sum ratio 而非 mean of ratios
            if 'SWvsMC' in col:
                avg[col] = df['Switch_Time_s'].sum() / df['MC_Time_s'].sum()
            elif 'SWvsSC' in col:
                avg[col] = df['Switch_Time_s'].sum() / df['SC_Time_s'].sum()
            elif 'SCvsMC' in col:
                avg[col] = df['SC_Time_s'].sum() / df['MC_Time_s'].sum()
        else:
            avg[col] = df[col].mean()
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df

df = compute_metrics()

# ============================================================
# 4. Markdown 报告
# ============================================================
def gen_md(df):
    dn = df[df['场景'] != 'AVERAGE']
    md = []
    md.append("# 算法求解效率对比分析\n")
    md.append(f"**生成日期**: 2026-07-28  \n")
    md.append(f"**数据来源**: Switch (CPLEX MIP) vs Single-Cut Benders (V1) vs Multi-Cut Benders (V6)\n")
    md.append("---\n")

    # 表1: 综合 Runtime 对比
    md.append("## 表1: 求解效率综合对比\n")
    md.append("| 场景 | Switch时间 | SC时间 | MC时间 | SW→MC加速 | SC→MC加速 | SW→SC加速 |")
    md.append("|:---|:---|:---|:---|:---|:---|:---|")
    for _, r in df.iterrows():
        b = '**' if r['场景'] == 'AVERAGE' else ''
        md.append(f"| {b}{r['场景']}{b} | {b}{r['Switch_Time_s']:.0f}s{b} | {b}{r['SC_Time_s']:.1f}s{b} | {b}{r['MC_Time_s']:.0f}s{b} | {b}{r['Speedup_SWvsMC']:.1f}×{b} | {b}{r['Speedup_SCvsMC']:.1f}×{b} | {b}{r['Speedup_SWvsSC']:.1f}×{b} |")
    md.append("")
    md.append("**注**: SW=Switch原始MIP, SC=Single-Cut Benders, MC=Multi-Cut Benders (V6).\n")

    # 表2: 迭代 & GAP 对比
    md.append("## 表2: 迭代次数与收敛精度对比\n")
    md.append("| 场景 | SC迭代 | SC-GAP% | MC迭代 | MC-GAP% | 迭代减少 |")
    md.append("|:---|:---|:---|:---|:---|:---|")
    for _, r in df.iterrows():
        b = '**' if r['场景'] == 'AVERAGE' else ''
        reduction = r['SC_Iters'] - r['MC_Iters']
        md.append(f"| {b}{r['场景']}{b} | {b}{r['SC_Iters']:.0f}{b} | {b}{r['SC_GAP_pct']:.3f}{b} | {b}{r['MC_Iters']:.0f}{b} | {b}{r['MC_GAP_pct']:.3f}{b} | {b}-{reduction:.0f}{b} |")
    md.append("")
    md.append(f"**注**: Switch 原始 MIP 不涉及迭代, MIPGap=0.2%.\n")

    # 表3: 成本偏差
    md.append("## 表3: 成本偏差分析 (相对于Switch参考解)\n")
    md.append("| 场景 | Switch (B¥) | SC (B¥) | SC偏差% | MC (B¥) | MC偏差% | |MC| vs |SC| |")
    md.append("|:---|:---|:---|:---|:---|:---|:---|")
    for _, r in df.iterrows():
        if r['场景'] == 'AVERAGE':
            continue
        sc_d = (r['SC_Cost_B'] - r['Switch_Cost_B']) * 1000  # M¥
        mc_d = (r['MC_Cost_B'] - r['Switch_Cost_B']) * 1000
        delta = abs(mc_d) - abs(sc_d)
        md.append(f"| {r['场景']} | {r['Switch_Cost_B']:.3f} | {r['SC_Cost_B']:.3f} | {r['SC_Dev_pct']:+.3f} | {r['MC_Cost_B']:.3f} | {r['MC_Dev_pct']:+.3f} | {delta:+.0f} M¥ |")

    avg_sc_d = dn['SC_Dev_pct'].mean()
    avg_mc_d = dn['MC_Dev_pct'].mean()
    md.append(f"| **平均** | **{dn['Switch_Cost_B'].mean():.3f}** | — | **{avg_sc_d:+.3f}** | — | **{avg_mc_d:+.3f}** | **{abs(avg_sc_d)-abs(avg_mc_d):.3f} pp** |")
    md.append("")
    md.append(f"- SC (Single-Cut) 平均偏差: **{avg_sc_d:+.3f}%** (系统性偏低, 因 DR 连续松弛)")
    md.append(f"- MC (Multi-Cut V6) 平均偏差: **{avg_mc_d:+.3f}%** (更接近Switch, 因 BaseloadLevel/CarbonAlloc 上移)")
    md.append("")

    # 表4: 求解器配置
    md.append("## 表4: 求解器与参数配置对比\n")
    md.append("| 参数 | Switch (CPLEX MIP) | Single-Cut Benders | Multi-Cut Benders (V6) |")
    md.append("|:---|:---|:---|:---|")
    config_keys = list(solver_configs['Switch (CPLEX MIP)'].keys())
    for k in config_keys:
        s_sw = solver_configs['Switch (CPLEX MIP)'][k]
        s_sc = solver_configs['Single-Cut Benders (V1)'][k]
        s_mc = solver_configs['Multi-Cut Benders (V6)'][k]
        md.append(f"| {k} | {s_sw} | {s_sc} | {s_mc} |")
    md.append("")

    # 表5: 汇总
    md.append("## 表5: 关键指标平均汇总\n")
    md.append("| 指标 | Switch | Single-Cut (V1) | Multi-Cut (V6) | 最优 |")
    md.append("|:---|:---|:---|:---|:---|")
    a_sw_t = dn['Switch_Time_s'].mean()
    a_sc_t = dn['SC_Time_s'].mean()
    a_mc_t = dn['MC_Time_s'].mean()
    a_sc_i = dn['SC_Iters'].mean()
    a_mc_i = dn['MC_Iters'].mean()
    a_sc_g = dn['SC_GAP_pct'].mean()
    a_mc_g = dn['MC_GAP_pct'].mean()
    md.append(f"| 平均求解时间 (s) | {a_sw_t:.0f} | {a_sc_t:.1f} | {a_mc_t:.1f} | **MC ({a_mc_t:.0f}s)** |")
    md.append(f"| 加速比 vs Switch | 1.0× | {a_sw_t/a_sc_t:.1f}× | **{a_sw_t/a_mc_t:.1f}×** | **MC** |")
    md.append(f"| 加速比 vs Single-Cut | — | 1.0× | **{a_sc_t/a_mc_t:.1f}×** | **MC** |")
    md.append(f"| 平均迭代次数 | N/A | {a_sc_i:.0f} | {a_mc_i:.0f} | **MC (-{a_sc_i-a_mc_i:.0f})** |")
    md.append(f"| 平均 GAP (%) | 0.200* | {a_sc_g:.3f} | {a_mc_g:.3f} | 同量级 |")
    md.append(f"| 平均 |Cost偏差| (%) | 0 (基准) | {abs(avg_sc_d):.3f} | {abs(avg_mc_d):.3f} | **MC** |")
    md.append("")
    md.append("\\* Switch 的 0.200% 是 CPLEX MIPGap 参数设置, 非迭代收敛 GAP.\n")

    return "\n".join(md)

md_content = gen_md(df)

# 保存文件
for fname, content, mode in [
    ('comparison_report.md', md_content, 'w'),
]:
    with open(os.path.join(OUT_DIR, fname), mode, encoding='utf-8') as f:
        f.write(content)

df.to_csv(os.path.join(OUT_DIR, 'comparison_results.csv'), index=False, encoding='utf-8-sig')

output_json = {
    'summary': {
        'avg_switch_time_s': round(df[df['场景']!='AVERAGE']['Switch_Time_s'].mean(), 0),
        'avg_single_cut_time_s': round(df[df['场景']!='AVERAGE']['SC_Time_s'].mean(), 1),
        'avg_multi_cut_time_s': round(df[df['场景']!='AVERAGE']['MC_Time_s'].mean(), 1),
        'speedup_sw_vs_mc': round(df[df['场景']!='AVERAGE']['Switch_Time_s'].sum()/df[df['场景']!='AVERAGE']['MC_Time_s'].sum(), 1),
        'speedup_sc_vs_mc': round(df[df['场景']!='AVERAGE']['SC_Time_s'].sum()/df[df['场景']!='AVERAGE']['MC_Time_s'].sum(), 1),
        'speedup_sw_vs_sc': round(df[df['场景']!='AVERAGE']['Switch_Time_s'].sum()/df[df['场景']!='AVERAGE']['SC_Time_s'].sum(), 1),
        'avg_single_cut_iters': round(df[df['场景']!='AVERAGE']['SC_Iters'].mean(), 0),
        'avg_multi_cut_iters': round(df[df['场景']!='AVERAGE']['MC_Iters'].mean(), 0),
        'avg_single_cut_gap_pct': round(df[df['场景']!='AVERAGE']['SC_GAP_pct'].mean(), 3),
        'avg_multi_cut_gap_pct': round(df[df['场景']!='AVERAGE']['MC_GAP_pct'].mean(), 3),
        'avg_single_cut_dev_pct': round(df[df['场景']!='AVERAGE']['SC_Dev_pct'].mean(), 3),
        'avg_multi_cut_dev_pct': round(df[df['场景']!='AVERAGE']['MC_Dev_pct'].mean(), 3),
    },
    'per_comp': {c: {
        'switch_cost': switch_cost[c], 'switch_time_s': switch_time[c],
        'single_cut': single_cut[c], 'multi_cut': multi_cut[c],
    } for c in COMPS},
    'solver_configs': solver_configs,
}
with open(os.path.join(OUT_DIR, 'comparison_data.json'), 'w', encoding='utf-8') as f:
    json.dump(output_json, f, indent=2, ensure_ascii=False)

# 打印结论
dn = df[df['场景'] != 'AVERAGE']
print("[OK] Markdown report saved.")
print("[OK] CSV data saved.")
print("[OK] JSON data saved.")
print(f"\n=== 关键结论 (平均值) ===")
print(f"  Switch 平均求解时间:       {dn['Switch_Time_s'].mean():.0f}s")
print(f"  Single-Cut 平均求解时间:   {dn['SC_Time_s'].mean():.1f}s")
print(f"  Multi-Cut 平均求解时间:    {dn['MC_Time_s'].mean():.0f}s")
print(f"  Switch → Multi-Cut 加速:  {dn['Switch_Time_s'].sum()/dn['MC_Time_s'].sum():.1f}×")
print(f"  Single-Cut → Multi-Cut:   {dn['SC_Time_s'].sum()/dn['MC_Time_s'].sum():.1f}×")
print(f"  Single-Cut 平均迭代:      {dn['SC_Iters'].mean():.0f} 轮")
print(f"  Multi-Cut 平均迭代:       {dn['MC_Iters'].mean():.0f} 轮")
print(f"  SC 平均 |Cost偏差|:       {abs(dn['SC_Dev_pct'].mean()):.3f}%")
print(f"  MC 平均 |Cost偏差|:       {abs(dn['MC_Dev_pct'].mean()):.3f}%")
print(f"  SC 平均 GAP:              {dn['SC_GAP_pct'].mean():.3f}%")
print(f"  MC 平均 GAP:              {dn['MC_GAP_pct'].mean():.3f}%")

"""
Benders 分解 V7 — V6 + Barrier subs + Thread opt + Warm-start
==============================================================================
与 V5 的核心区别:
  - BaseloadLevel[g,p]: 基荷平坦出力从子问题上移至 Master
  - CarbonAlloc[p,s]:   碳排放配额从单一年度总量上移至 Master 动态分配
  - 子问题: 48 个独立 Segment LP (每 segment = 1 典型日 = 24 时段)
  - 每个 segment 自己返回有效次梯度 (不再从全周期 LP 人为分解)
  - 消除 V4/V5 的 "跨迭代 per-segment max" 导致的 LB 虚高问题

架构:
  Master (LP):
    投资变量 + theta_p + theta_ps + BaseloadLevel + CarbonAlloc
    约束: BaseloadLevel <= GenCap*avail, Σ_s CarbonAlloc <= CarbonCap
  Subproblem × 48 (LP, 独立):
    每 segment (p,s) 独立求解 → 返回 ∂obj_s/∂(GenCap, BaseloadLevel, CarbonAlloc, ...)
  Benders cuts:
    Per-segment cut on θ_ps + Aggregate cut on θ_p (Hybrid Dual-θ)
"""

import os, sys, time, logging, io
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# ============================================================
# 日志配置
# ============================================================
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(LOG_DIR, 'output_benders_v6'), exist_ok=True)

logger = logging.getLogger('benders')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(ch)
fh = logging.FileHandler(os.path.join(LOG_DIR, 'output_benders_v6', 'benders_log.txt'), mode='w')
fh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.addHandler(fh)

# ============================================================
# 数据加载
# ============================================================
from benders_fixed_v6.data_loader import load_all

# ============================================================
# 财务函数
# ============================================================
def crf(ir, t): return 1/t if ir==0 else ir/(1-(1+ir)**-t)
def uspv(dr, t): return t if dr==0 else (1-(1+dr)**-t)/dr
def fpv(dr, t): return (1+dr)**-t

# 配置
SEGMENTS_PER_PERIOD = 12  # 每周期 segment 数 (每个 = 1 典型日)
THETA_PS_LB = -1e8        # per-segment theta 下界 (允许负值, 消除 kinked penalty)


# ============================================================
# 构建主模型 (Master LP, 持久化)
# ============================================================
def build_persistent_master(data):
    """构建持久化 Master LP: 投资变量 + BaseloadLevel + CarbonAlloc + theta + Benders cuts"""
    m = gp.Model('Benders_Master_V6')
    m.Params.OutputFlag = 0
    m.Params.Threads = 2   # V7: reduced for multi-worker parallelism
    m.Params.MIPGap = 0.002

    P = data['PERIODS']
    G = data['gen']
    ba = data['bring_annual']
    INTEREST = data['INTEREST_RATE']
    S = SEGMENTS_PER_PERIOD

    # ---- 投资变量 (同 V5) ----
    BuildGen = {}
    for (g, p) in data['NEW_GEN_BLD_YRS']:
        BuildGen[(g, p)] = m.addVar(lb=0, name=f'Build_{g}_{p}')

    SuspendGen = {}; GEN_BLD_SUSPEND = []
    for g in data['GENERATORS']:
        can_susp = G[g].get('can_suspend', False)
        can_ret = G[g].get('can_retire_early', False)
        if not (can_susp or can_ret):
            continue
        for (g2, by) in list(data['pred_build'].keys()):
            if g2 != g: continue
            for p in P:
                if data['is_alive'](g, by, p):
                    SuspendGen[(g, by, p)] = m.addVar(lb=0, name=f'Susp_{g}_{by}_{p}')
                    GEN_BLD_SUSPEND.append((g, by, p))
        for p in P:
            if (g, p) in BuildGen:
                for pp in P:
                    if pp >= p and data['is_alive'](g, p, pp):
                        SuspendGen[(g, p, pp)] = m.addVar(lb=0, name=f'SuspN_{g}_{p}_{pp}')
                        GEN_BLD_SUSPEND.append((g, p, pp))

    GenCap = {}
    for g in data['GENERATORS']:
        for p in P:
            aby = [by for (g2, by) in data['pred_build'] if g2 == g and data['is_alive'](g, by, p)]
            pred = sum(data['pred_build'].get((g, by), 0) for by in aby)
            be = gp.quicksum(BuildGen.get((g, pp), 0) for pp in P if pp <= p and data['is_alive'](g, pp, p))
            se = gp.quicksum(SuspendGen.get((g, by, p), 0) for (g2, by, pp) in GEN_BLD_SUSPEND if g2 == g and pp == p)
            GenCap[(g, p)] = pred + be - se

    BuildCogen = {}
    for g in data['FUEL_GENS']:
        for p in P: BuildCogen[(g, p)] = m.addVar(lb=0, name=f'BC_{g}_{p}')
    CogenCap = {}
    for g in data['FUEL_GENS']:
        for p in P: CogenCap[(g, p)] = gp.quicksum(BuildCogen.get((g, pp), 0) for pp in P if pp <= p)

    BuildStorEnergy = {}
    for g in data['STORAGE_GENS']:
        for p in P: BuildStorEnergy[(g, p)] = m.addVar(lb=0, name=f'BSE_{g}_{p}')
    StorEnergyCap = {}
    for g in data['STORAGE_GENS']:
        for p in P: StorEnergyCap[(g, p)] = gp.quicksum(BuildStorEnergy.get((g, pp), 0) for pp in P if pp <= p)

    BuildLocalTD = {}
    for z in data['LOAD_ZONES']:
        for p in P: BuildLocalTD[(z, p)] = m.addVar(lb=0, name=f'BTD_{z}_{p}')
    LocalTDCap = {}
    for z in data['LOAD_ZONES']:
        ex = data['lz_data'].get(z, {}).get('existing_td', 0)
        for p in P: LocalTDCap[(z, p)] = ex + gp.quicksum(BuildLocalTD.get((z, pp), 0) for pp in P if pp <= p)

    BuildTx = {}
    for tx in data['tx_data']:
        for p in P: BuildTx[(tx, p)] = m.addVar(lb=0, name=f'BTx_{tx}_{p}')
    TxCap = {}
    for tx, td in data['tx_data'].items():
        for p in P: TxCap[(tx, p)] = td['exist'] + gp.quicksum(BuildTx.get((tx, pp), 0) for pp in P if pp <= p)

    # ---- 基荷平坦出力: BaseloadLevel[g,p] (NEW in V6) ----
    # 基荷机组: is_baseload && not H2 (对齐 switch_gurobi.py 排除 coal 的逻辑?
    # 暂不排除 coal, 保持与 benders V1-V5 一致)
    BASELOAD_GENS = [g for g in data['GENERATORS']
                     if G[g]['is_baseload'] and g not in data['H2_GENS']]
    BaseloadLevel = {}
    for g in BASELOAD_GENS:
        for p in P:
            BaseloadLevel[(g, p)] = m.addVar(lb=0, name=f'BL_{g}_{p}')

    # 基荷出力约束: BaseloadLevel <= GenCap * avail
    for g in BASELOAD_GENS:
        for p in P:
            m.addConstr(BaseloadLevel[(g, p)] <= GenCap[(g, p)] * G[g]['avail'],
                       name=f'BaseloadCap_{g}_{p}')

    # ---- 碳排放动态分配: CarbonAlloc[p,s] (NEW in V6) ----
    CarbonAlloc = {}
    for p in P:
        if p in data['carbon_cap']:
            for s in range(S):
                CarbonAlloc[(p, s)] = m.addVar(lb=0, name=f'CA_{p}_{s}')

    # 碳配额约束: Σ_s CarbonAlloc[p,s] <= CarbonCap[p]
    for p in P:
        if p in data['carbon_cap']:
            m.addConstr(gp.quicksum(CarbonAlloc[(p, s)] for s in range(S))
                       <= data['carbon_cap'][p], name=f'CarbonBudget_{p}')

    # ---- Theta — Hybrid Dual-θ ----
    theta_p = {}
    theta_ps = {}
    for p in P:
        theta_p[p] = m.addVar(lb=0, name=f'theta_p_{p}')
        for s in range(S):
            theta_ps[(p, s)] = m.addVar(lb=THETA_PS_LB, name=f'theta_ps_{p}_{s}')

    m.update()

    # 链接约束: theta_p >= Σ_s theta_ps
    for p in P:
        m.addConstr(theta_p[p] >= gp.quicksum(theta_ps[(p, s)] for s in range(S)),
                   name=f'LinkTheta_{p}')

    # ---- 投资约束 (同 V5) ----
    for (g, by, p) in GEN_BLD_SUSPEND:
        bv = data['pred_build'].get((g, by), 0) if by not in P else 0
        bvar = BuildGen.get((g, by))
        tot = bv
        if bvar is not None: tot += bvar
        m.addConstr(SuspendGen[(g, by, p)] <= tot)

    for (g, by, p) in GEN_BLD_SUSPEND:
        if G[g].get('can_retire_early', False) and not G[g].get('can_suspend', False):
            prevs = [pp for pp in P if pp < p]
            if prevs:
                ps = SuspendGen.get((g, by, max(prevs)), None)
                if ps is not None: m.addConstr(SuspendGen[(g, by, p)] >= ps)

    for g in data['CAP_LIMITED']:
        for p in P: m.addConstr(GenCap[(g, p)] <= G[g]['cap_limit'])

    # ---- 目标函数 ----
    obj = gp.LinExpr()
    for (g, p), var in BuildGen.items():
        ov = data['get_cost'](g, p, data['gen_overnight'])
        annual_cap = (ov + G[g]['connect_cost']) * crf(INTEREST, G[g]['max_age'])
        alive_npv = sum(ba[pp] for pp in P if pp >= p and data['is_alive'](g, p, pp))
        obj += var * annual_cap * alive_npv
    for (g, p), var in BuildGen.items():
        fom = data['get_cost'](g, p, data['gen_fixed_om'])
        for pp in P:
            if pp >= p and data['is_alive'](g, p, pp):
                obj += (var - SuspendGen.get((g, p, pp), 0)) * fom * ba[pp]
    for g in data['GENERATORS']:
        for p in P:
            for by in [b for (g2, b) in data['pred_build'] if g2 == g and data['is_alive'](g, b, p)]:
                pv = data['pred_build'].get((g, by), 0)
                if pv <= 0: continue
                fom = data['gen_fixed_om'].get((g, by), data['get_cost'](g, p, data['gen_fixed_om']))
                ov = data['gen_overnight'].get((g, by), data['get_cost'](g, p, data['gen_overnight']))
                ac = (ov + G[g]['connect_cost']) * crf(INTEREST, G[g]['max_age'])
                obj += pv * ac * ba[p]
                obj += (pv - SuspendGen.get((g, by, p), 0)) * fom * ba[p]
    for g in data['FUEL_GENS']:
        for p in P: obj += CogenCap[(g, p)] * data['COGEN_FIXED_COST'] * ba[p]
    for (g, p), var in BuildStorEnergy.items():
        ec = data['get_cost'](g, p, data['gen_store_en_cost'], default=0)
        alive_npv = sum(ba[pp] for pp in P if pp >= p and data['is_alive'](g, p, pp))
        obj += var * ec * crf(INTEREST, G[g]['max_age']) * alive_npv
    for z in data['LOAD_ZONES']:
        tc = data['lz_data'].get(z, {}).get('td_cost', 0)
        for p in P: obj += LocalTDCap[(z, p)] * tc * ba[p]
    tcrf = crf(INTEREST, 20)
    for tx, td in data['tx_data'].items():
        ta = td['cost_per_mw_km'] * td['length'] * (tcrf + 0.03)
        for p in P: obj += TxCap[(tx, p)] * ta * ba[p]
    for p in P:
        obj += theta_p[p]

    m.setObjective(obj, GRB.MINIMIZE)

    # 存储引用
    m._BuildGen = BuildGen; m._SuspendGen = SuspendGen
    m._GenCap = GenCap; m._BuildCogen = BuildCogen; m._CogenCap = CogenCap
    m._BuildStorEnergy = BuildStorEnergy; m._StorEnergyCap = StorEnergyCap
    m._BuildLocalTD = BuildLocalTD; m._LocalTDCap = LocalTDCap
    m._BuildTx = BuildTx; m._TxCap = TxCap
    m._theta_p = theta_p; m._theta_ps = theta_ps
    m._BaseloadLevel = BaseloadLevel   # NEW
    m._CarbonAlloc = CarbonAlloc       # NEW
    m._BASELOAD_GENS = BASELOAD_GENS   # NEW
    m._GEN_BLD_SUSPEND = GEN_BLD_SUSPEND; m._n_cuts = 0

    logger.info(f"[Master V6] 构建完成: {m.NumVars:,} 变量, {m.NumConstrs:,} 约束 "
               f"(Baseload={len(BaseloadLevel)}, CarbonAlloc={len(CarbonAlloc)})")
    return m


# ============================================================
# 构建独立 Segment 子问题 (LP, 持久化)
# ============================================================
def build_segment_subproblem(data, period, seg_idx):
    """
    构建单个 Segment 子问题 LP (V6 核心重构)

    参数:
      data:   输入数据
      period: 投资周期
      seg_idx: segment 索引 (0..SEGMENTS_PER_PERIOD-1)

    每个 segment 包含 ~24 个时间点 (= 1 个典型日)
    子问题通过显式链接约束接收 master 决策:
      K_gen[g]      -> GenCap*[g,p]
      K_baseload[g] -> BaseloadLevel*[g,p]   (NEW)
      K_carbon       -> CarbonAlloc*[p,s]     (NEW)
      K_stor_en, K_cogen, K_ltd, K_tx        (同 V5)
    """
    p = period
    tps_all = data['tps_in_period'][p]
    S = SEGMENTS_PER_PERIOD
    seg_size = max(1, len(tps_all) // S)
    start = seg_idx * seg_size
    end = start + seg_size if seg_idx < S - 1 else len(tps_all)
    seg_tps = tps_all[start:end]

    G = data['gen']
    bp = data['bring_annual'][p]
    twy = data['tp_wt_yr']

    m = gp.Model(f'Sub_{p}_s{seg_idx}')
    m.Params.OutputFlag = 0
    m.Params.Threads = 1
    m.Params.Method = 1      # V7: dual simplex (fast re-opt + precise duals)
    m.Params.Crossover = 0

    # ============ 显式链接变量 K_* ============
    # GenCap 链接
    K_gen = {}; link_gen = {}
    for g in data['GENERATORS']:
        K_gen[g] = m.addVar(lb=-GRB.INFINITY, name=f'Kgen_{g}')
        link_gen[g] = m.addConstr(K_gen[g] == 0.0, name=f'LinkGen_{g}')

    # BaseloadLevel 链接 (NEW)
    K_baseload = {}; link_baseload = {}
    BASELOAD_GENS = [g for g in data['GENERATORS']
                     if G[g]['is_baseload'] and g not in data['H2_GENS']]
    for g in BASELOAD_GENS:
        K_baseload[g] = m.addVar(lb=-GRB.INFINITY, name=f'Kbl_{g}')
        link_baseload[g] = m.addConstr(K_baseload[g] == 0.0, name=f'LinkBL_{g}')

    # CarbonAlloc 链接 (NEW) — 单变量
    K_carbon = None; link_carbon = None
    if p in data['carbon_cap']:
        K_carbon = m.addVar(lb=-GRB.INFINITY, name='Kcarbon')
        link_carbon = m.addConstr(K_carbon == 0.0, name='LinkCarbon')

    # 其他链接 (同 V5)
    K_cogen = {}; link_cogen = {}
    for g in data['FUEL_GENS']:
        K_cogen[g] = m.addVar(lb=-GRB.INFINITY, name=f'Kcgn_{g}')
        link_cogen[g] = m.addConstr(K_cogen[g] == 0.0, name=f'LinkCgn_{g}')

    K_stor_en = {}; link_stor_en = {}
    for g in data['STORAGE_GENS']:
        K_stor_en[g] = m.addVar(lb=-GRB.INFINITY, name=f'Kstor_{g}')
        link_stor_en[g] = m.addConstr(K_stor_en[g] == 0.0, name=f'LinkStor_{g}')

    K_ltd = {}; link_ltd = {}
    for z in data['LOAD_ZONES']:
        K_ltd[z] = m.addVar(lb=-GRB.INFINITY, name=f'Kltd_{z}')
        link_ltd[z] = m.addConstr(K_ltd[z] == 0.0, name=f'LinkLTD_{z}')

    K_tx = {}; link_tx = {}
    for tx in data['tx_data']:
        K_tx[tx] = m.addVar(lb=-GRB.INFINITY, name=f'Ktx_{tx}')
        link_tx[tx] = m.addConstr(K_tx[tx] == 0.0, name=f'LinkTx_{tx}')

    # ============ 调度变量 (仅 segment 内时间点) ============
    DispatchGen = {}
    for g in data['GENERATORS']:
        if g in data['H2_GENS']: continue
        for t in seg_tps:
            DispatchGen[(g, t)] = m.addVar(lb=0, name=f'D_{g}_{t}')

    FuelUse = {}
    for g in data['FUEL_GENS']:
        for t in seg_tps:
            if (g, t) in DispatchGen:
                FuelUse[(g, t)] = m.addVar(lb=0, name=f'F_{g}_{t}')

    ChargeStor = {}; SOC = {}
    for g in data['STORAGE_GENS']:
        for t in seg_tps:
            ChargeStor[(g, t)] = m.addVar(lb=0, name=f'C_{g}_{t}')
            SOC[(g, t)] = m.addVar(lb=0, name=f'S_{g}_{t}')

    DispatchCogen = {}
    for g in data['FUEL_GENS']:
        for t in seg_tps:
            if (g, t) in DispatchGen:
                DispatchCogen[(g, t)] = m.addVar(lb=0, name=f'DC_{g}_{t}')

    DR_act = {}; ShiftUp = {}; ShiftDown = {}; RecovDem = {}
    for z in data['LOAD_ZONES']:
        for t in seg_tps:
            DR_act[(z, t)] = m.addVar(lb=0, ub=1, name=f'DR_{z}_{t}')
            ShiftUp[(z, t)] = m.addVar(lb=0, name=f'SU_{z}_{t}')
            ShiftDown[(z, t)] = m.addVar(lb=0, name=f'SD_{z}_{t}')
            RecovDem[(z, t)] = m.addVar(lb=-GRB.INFINITY, name=f'RD_{z}_{t}')

    DispatchTx = {}
    for z1 in data['LOAD_ZONES']:
        for z2 in data['LOAD_ZONES']:
            if z1 != z2:
                for t in seg_tps:
                    DispatchTx[(z1, z2, t)] = m.addVar(lb=0, name=f'TX_{z1}_{z2}_{t}')

    WithdrawCG = {}
    for z in data['LOAD_ZONES']:
        for t in seg_tps: WithdrawCG[(z, t)] = m.addVar(lb=0, name=f'WD_{z}_{t}')

    Unserved = {}
    for z in data['LOAD_ZONES']:
        for t in seg_tps: Unserved[(z, t)] = m.addVar(lb=0, name=f'US_{z}_{t}')

    RunElec = {}; LiqH2 = {}
    for z in data['LOAD_ZONES']:
        for t in seg_tps:
            RunElec[(z, t)] = m.addVar(lb=0, name=f'RE_{z}_{t}')
            LiqH2[(z, t)] = m.addVar(lb=0, name=f'LH_{z}_{t}')

    m.update()

    # ============ 调度约束 ============
    # 调度上限: DispatchGen <= K_gen * avail * CF
    disp_ul = {}
    for (g, t), var in DispatchGen.items():
        avail = G[g]['avail']
        if G[g]['is_vre']:
            cf = data['get_cf'](g, t)
            disp_ul[(g, t)] = m.addConstr(var <= K_gen[g] * avail * cf)
        else:
            disp_ul[(g, t)] = m.addConstr(var <= K_gen[g] * avail)

    # 基荷平坦约束 (NEW): DispatchGen[g,t] == K_baseload[g] + BaseloadDev
    # K_baseload 链接到 master 的 BaseloadLevel[g,p]
    # BaseloadDev 松弛变量确保 segment 始终可行 (高惩罚成本, 类似 Unserved)
    BL_PENALTY = 5000.0  # 基荷偏离惩罚 ($/MWh, 高于 Unserved=2000 确保优先使用 Unserved)
    BaseloadDevUp = {}; BaseloadDevDown = {}
    for g in BASELOAD_GENS:
        for t in seg_tps:
            if (g, t) in DispatchGen:
                BaseloadDevUp[(g, t)] = m.addVar(lb=0, name=f'BLdevU_{g}_{t}')
                BaseloadDevDown[(g, t)] = m.addVar(lb=0, name=f'BLdevD_{g}_{t}')
                m.addConstr(DispatchGen[(g, t)] == K_baseload[g]
                           + BaseloadDevUp[(g, t)] - BaseloadDevDown[(g, t)])

    # 燃料
    for (g, t), var in FuelUse.items():
        m.addConstr(var == DispatchGen[(g, t)] * G[g]['heat_rate'])

    # 中央电网平衡
    for z in data['LOAD_ZONES']:
        for t in seg_tps:
            central = gp.quicksum(DispatchGen.get((g, t), 0) for g in data['GENS_BY_ZONE'][z]
                                  if (g, t) in DispatchGen and g not in data['STORAGE_GENS'])
            sn = gp.quicksum(DispatchGen.get((g, t), 0) - ChargeStor.get((g, t), 0)
                            for g in data['STORAGE_GENS'] if G[g]['zone'] == z)
            cd = gp.quicksum(DispatchCogen.get((g, t), 0) for g in data['FUEL_GENS'] if G[g]['zone'] == z)
            tn = sum((DispatchTx.get((z2, z, t), 0) * 0.95 - DispatchTx.get((z, z2, t), 0))
                    for z2 in data['LOAD_ZONES'] if z2 != z)
            m.addConstr(central + sn + cd + tn ==
                       WithdrawCG.get((z, t), 0) + RunElec.get((z, t), 0) + LiqH2.get((z, t), 0) * 0.01)

    # 配电平衡
    for z in data['LOAD_ZONES']:
        for t in seg_tps:
            m.addConstr(WithdrawCG.get((z, t), 0) * 0.947 + Unserved.get((z, t), 0) ==
                       data['zone_load'][z].get(t, 0) + ShiftUp.get((z, t), 0)
                       - ShiftDown.get((z, t), 0) + RecovDem.get((z, t), 0))

    # 本地TD容量
    ltd_bound = {}
    for z in data['LOAD_ZONES']:
        for t in seg_tps:
            ltd_bound[(z, t)] = m.addConstr(WithdrawCG[(z, t)] <= K_ltd[z], name=f'LTDB_{z}_{t}')

    # 储能约束 (在 segment 时间点内, SOC 循环)
    charge_bound = {}; soc_bound = {}
    for g in data['STORAGE_GENS']:
        eff = G[g]['storage_eff']; ratio = G[g]['store_ratio']
        for ts_n in data['TIMESERIES']:
            stps = sorted([t_ for t_ in data['tps_in_ts'][ts_n] if t_ in seg_tps])
            if not stps: continue
            for i, t in enumerate(stps):
                charge_bound[(g, t)] = m.addConstr(ChargeStor[(g, t)] <= K_gen[g] * ratio,
                                                   name=f'CHB_{g}_{t}')
                dur = data['ts_dur'].get(ts_n, 1.0)
                prev = SOC[(g, stps[-1])] if i == 0 else SOC[(g, stps[i-1])]
                m.addConstr(SOC[(g, t)] == prev + (ChargeStor[(g, t)] * eff
                            - DispatchGen.get((g, t), 0)) * dur)
                soc_bound[(g, t)] = m.addConstr(SOC[(g, t)] <= K_stor_en[g], name=f'SOCB_{g}_{t}')

    # DR 约束 (仅 segment 内时间点)
    for z in data['LOAD_ZONES']:
        for t in seg_tps:
            up = data['dr_up_lim'].get((z, t), 0); down = data['dr_down_lim'].get((z, t), 0)
            if up > 0:
                m.addConstr(ShiftUp[(z, t)] <= DR_act[(z, t)] * up)
                m.addConstr(ShiftDown[(z, t)] <= DR_act[(z, t)] * down)
        for ts_n in data['TIMESERIES']:
            stps = sorted([t_ for t_ in data['tps_in_ts'][ts_n] if t_ in seg_tps])
            if not stps: continue
            for i, t in enumerate(stps):
                rc = data['dr_recov_time'].get((z, t), 2.0)
                ns = max(1, int(rc / data['ts_dur'].get(ts_n, 1.0)))
                re = sum((ShiftUp.get((z, stps[i-j]), 0) - ShiftDown.get((z, stps[i-j]), 0))
                        * max(0, 1.0 - j * data['ts_dur'].get(ts_n, 1.0) / rc)
                        for j in range(1, ns+1) if i-j >= 0)
                m.addConstr(RecovDem[(z, t)] == re)
            m.addConstr(gp.quicksum(ShiftUp.get((z, tp), 0) - ShiftDown.get((z, tp), 0)
                       + RecovDem.get((z, tp), 0) for tp in stps) == 0)
            for i in range(len(stps)-5):
                m.addConstr(gp.quicksum(DR_act[(z, tp)] for tp in stps[i:i+6]) <= 1)
            ul = data['dr_up_lim'].get((z, stps[0]), 0)
            if ul > 0: m.addConstr(gp.quicksum(ShiftUp.get((z, tp), 0) for tp in stps) <= ul)

    # 热电联产
    cogen_bound = {}
    for g in data['FUEL_GENS']:
        for t in seg_tps:
            if (g, t) not in DispatchCogen: continue
            cogen_bound[(g, t)] = m.addConstr(DispatchCogen[(g, t)] <= K_cogen[g], name=f'CGB_{g}_{t}')
            m.addConstr(DispatchCogen[(g, t)] * data['COGEN_HEAT_RATE']
                       <= DispatchGen.get((g, t), 0) * (G[g]['heat_rate'] - 3.412))

    # 输电容量
    tx_bound = {}
    for z1, z2, t in DispatchTx:
        for tx, td in data['tx_data'].items():
            if td['lz1'] == z1 and td['lz2'] == z2:
                tx_bound[(tx, t)] = m.addConstr(DispatchTx[(z1, z2, t)] <= K_tx[tx], name=f'TXB_{tx}_{t}')

    # 碳排放 (NEW): per-segment 独立碳预算 + 松弛变量
    carbon_bound = None; CarbonSlack = None
    if p in data['carbon_cap'] and K_carbon is not None:
        CARBON_SLACK_COST = 10000.0  # 碳超排惩罚 ($/tCO2, 高于正常碳价)
        CarbonSlack = m.addVar(lb=0, name='CSlack')
        em = gp.quicksum(FuelUse.get((g, t), 0) * data['co2_intensity'].get(G[g]['energy_source'], 0.1006)
                         * data['tp_weight'].get(t, 1.0) / data['period_len_yr'][p]
                         for g in data['FUEL_GENS'] for t in seg_tps if (g, t) in FuelUse)
        carbon_bound = m.addConstr(em <= K_carbon + CarbonSlack, name='CarbonSeg')

    # 氢能 (简化)
    for z in data['LOAD_ZONES']:
        for t in seg_tps:
            m.addConstr(RunElec[(z, t)] <= 1e6)

    # ============ 目标函数 (纯运营成本) ============
    obj = gp.LinExpr()
    _fuel_coeff = {}
    for (g, t), var in FuelUse.items():
        fuel = G[g]['energy_source']; zone = G[g]['zone']
        price = data['fuel_price'].get((fuel, p, zone),
                                       data['fuel_price'].get((fuel, data['PERIODS'][0], zone), 3.0))
        coeff = price * twy[t] * bp
        obj += var * coeff
        _fuel_coeff[(g, t)] = coeff
    for (g, t), var in DispatchGen.items():
        vom = G[g]['vom']
        if vom > 0: obj += var * vom * twy[t] * bp
    for (z, t), var in Unserved.items():
        obj += var * data['UNSERVED_COST'] * twy[t] * bp
    for (z, t), var in RunElec.items():
        obj += var * 0.01 * twy[t] * bp
    # Baseload 偏离惩罚 (高成本, 确保仅在必要时使用)
    for (g, t), var in BaseloadDevUp.items():
        obj += var * BL_PENALTY * twy[t] * bp
    for (g, t), var in BaseloadDevDown.items():
        obj += var * BL_PENALTY * twy[t] * bp
    # 碳超排惩罚
    if CarbonSlack is not None:
        obj += CarbonSlack * CARBON_SLACK_COST * bp

    m.setObjective(obj, GRB.MINIMIZE)
    m.update()

    # 存储引用
    m._K_gen = K_gen; m._link_gen = link_gen
    m._K_baseload = K_baseload; m._link_baseload = link_baseload
    m._K_carbon = K_carbon; m._link_carbon = link_carbon
    m._K_cogen = K_cogen; m._link_cogen = link_cogen
    m._K_stor_en = K_stor_en; m._link_stor_en = link_stor_en
    m._K_ltd = K_ltd; m._link_ltd = link_ltd
    m._K_tx = K_tx; m._link_tx = link_tx
    m._DR_act = DR_act
    m._fuel_coeff = _fuel_coeff
    m._seg_tps = seg_tps
    m._period = p; m._seg_idx = seg_idx
    m._BASELOAD_GENS = BASELOAD_GENS
    m._DispatchGen = DispatchGen
    m._ChargeStor = ChargeStor
    m._FuelUse = FuelUse
    m._Unserved = Unserved
    m._BaseloadDevUp = BaseloadDevUp
    m._BaseloadDevDown = BaseloadDevDown
    m._CarbonSlack = CarbonSlack

    return m


# ============================================================
# 更新 Segment 子问题链接约束 RHS
# ============================================================
def update_segment_rhs(sub, master_sol, period, seg_idx):
    """修改 segment 子问题的链接约束 RHS, 注入 master 解"""
    p = period

    # GenCap
    for g, constr in sub._link_gen.items():
        constr.RHS = master_sol['GenCap'].get((g, p), 0.0)

    # BaseloadLevel (NEW)
    for g, constr in sub._link_baseload.items():
        constr.RHS = master_sol['BaseloadLevel'].get((g, p), 0.0)

    # CarbonAlloc (NEW) — 单变量, 不需要按 g 遍历
    if sub._link_carbon is not None:
        sub._link_carbon.RHS = master_sol['CarbonAlloc'].get((p, seg_idx), 0.0)

    # Others
    for g, constr in sub._link_cogen.items():
        constr.RHS = master_sol['CogenCap'].get((g, p), 0.0)
    for g, constr in sub._link_stor_en.items():
        constr.RHS = master_sol['StorEnergyCap'].get((g, p), 0.0)
    for z, constr in sub._link_ltd.items():
        constr.RHS = master_sol['LocalTDCap'].get((z, p), 0.0)
    for tx, constr in sub._link_tx.items():
        constr.RHS = master_sol['TxCap'].get((tx, p), 0.0)

    sub.update()


# ============================================================
# 求解 Segment 子问题 + 提取对偶
# ============================================================
def solve_segment_subproblem(sub, period, seg_idx, data):
    """求解单个 segment 子问题并提取所有链接约束对偶 (有效次梯度)"""
    sub.optimize()
    if sub.Status != GRB.OPTIMAL:
        return {'obj': float('inf'), 'status': sub.Status}

    # 提取链接约束对偶 (这些都是有效次梯度, 因为 segment LP 完全独立)
    gen_duals = {g: c.Pi for g, c in sub._link_gen.items() if abs(c.Pi) > 1e-10}
    baseload_duals = {g: c.Pi for g, c in sub._link_baseload.items() if abs(c.Pi) > 1e-10}
    carbon_dual = sub._link_carbon.Pi if sub._link_carbon is not None else 0.0
    cogen_duals = {g: c.Pi for g, c in sub._link_cogen.items() if abs(c.Pi) > 1e-10}
    stor_duals = {g: c.Pi for g, c in sub._link_stor_en.items() if abs(c.Pi) > 1e-10}
    ltd_duals = {z: c.Pi for z, c in sub._link_ltd.items() if abs(c.Pi) > 1e-10}
    tx_duals = {tx: c.Pi for tx, c in sub._link_tx.items() if abs(c.Pi) > 1e-10}

    # DR fractional 统计
    dr_frac = sum(1 for v in sub._DR_act.values() if 0.001 < v.X < 0.999)
    dr_total = len(sub._DR_act)

    return {
        'obj': sub.ObjVal,
        'duals': {
            'GenCap': gen_duals,
            'BaseloadLevel': baseload_duals,   # NEW
            'CarbonAlloc': carbon_dual,         # NEW
            'CogenCap': cogen_duals,
            'StorEnergyCap': stor_duals,
            'LocalTDCap': ltd_duals,
            'TxCap': tx_duals,
        },
        'dr_frac': dr_frac,
        'dr_total': dr_total,
        'status': sub.Status,
    }


# ============================================================
# 提取主模型解
# ============================================================
def extract_master_sol(master, data):
    P = data['PERIODS']; S = SEGMENTS_PER_PERIOD
    sol = {}
    sol['GenCap'] = {(g, p): master._GenCap[(g, p)].getValue()
                     for g in data['GENERATORS'] for p in P if (g, p) in master._GenCap}
    sol['BaseloadLevel'] = {(g, p): master._BaseloadLevel[(g, p)].X
                           for (g, p) in master._BaseloadLevel}
    sol['CarbonAlloc'] = {(p, s): master._CarbonAlloc[(p, s)].X
                         for (p, s) in master._CarbonAlloc}
    sol['CogenCap'] = {(g, p): master._CogenCap[(g, p)].getValue() for g in data['FUEL_GENS'] for p in P}
    sol['StorEnergyCap'] = {(g, p): master._StorEnergyCap[(g, p)].getValue() for g in data['STORAGE_GENS'] for p in P}
    sol['LocalTDCap'] = {(z, p): master._LocalTDCap[(z, p)].getValue() for z in data['LOAD_ZONES'] for p in P}
    sol['TxCap'] = {(tx, p): master._TxCap[(tx, p)].getValue() for tx in data['tx_data'] for p in P}
    sol['theta_p'] = {p: master._theta_p[p].X for p in P}
    sol['theta_ps'] = {(p, s): master._theta_ps[(p, s)].X for p in P for s in range(S)}
    sol['invest_cost'] = master.ObjVal - sum(sol['theta_p'].values())
    sol['master_obj'] = master.ObjVal
    return sol


# ============================================================
# 添加 Benders Cuts
# ============================================================
def _add_cut_terms(master, period, duals, ref_vals):
    """共用: 从 duals + ref_vals 构造 cut expression"""
    GenCap = master._GenCap
    BaseloadLevel = master._BaseloadLevel
    CarbonAlloc = master._CarbonAlloc
    CogenCap = master._CogenCap
    StorEnergyCap = master._StorEnergyCap
    LocalTDCap = master._LocalTDCap
    TxCap = master._TxCap

    const = 0.0
    lin = gp.LinExpr()

    for g, lam in duals.get('GenCap', {}).items():
        ref = ref_vals['GenCap'].get((g, period), 0)
        const -= lam * ref
        if (g, period) in GenCap: lin += lam * GenCap[(g, period)]

    for g, lam in duals.get('BaseloadLevel', {}).items():
        ref = ref_vals['BaseloadLevel'].get((g, period), 0)
        const -= lam * ref
        if (g, period) in BaseloadLevel: lin += lam * BaseloadLevel[(g, period)]

    ca_dual = duals.get('CarbonAlloc', 0.0)
    ca_seg = duals.get('_ca_seg', None)  # set by caller for per-segment cut
    if ca_seg is not None and abs(ca_dual) > 1e-10:
        ref = ref_vals['CarbonAlloc'].get((period, ca_seg), 0)
        const -= ca_dual * ref
        lin += ca_dual * CarbonAlloc[(period, ca_seg)]

    for g, lam in duals.get('CogenCap', {}).items():
        ref = ref_vals['CogenCap'].get((g, period), 0)
        const -= lam * ref
        lin += lam * CogenCap[(g, period)]

    for g, lam in duals.get('StorEnergyCap', {}).items():
        ref = ref_vals['StorEnergyCap'].get((g, period), 0)
        const -= lam * ref
        lin += lam * StorEnergyCap[(g, period)]

    for z, lam in duals.get('LocalTDCap', {}).items():
        ref = ref_vals['LocalTDCap'].get((z, period), 0)
        const -= lam * ref
        lin += lam * LocalTDCap[(z, period)]

    for tx, lam in duals.get('TxCap', {}).items():
        ref = ref_vals['TxCap'].get((tx, period), 0)
        const -= lam * ref
        lin += lam * TxCap[(tx, period)]

    return const, lin


def add_benders_cut_single(master, period, total_sub_obj, agg_duals, ref_vals, data):
    """Aggregate (Single) Cut on theta_p — period 级别汇总"""
    theta_p = master._theta_p[period]
    const = total_sub_obj
    cut_lin = gp.LinExpr()

    # 累加所有 segment 的对偶 (agg_duals 是 per-category 的 sum)
    for g, lam in agg_duals.get('GenCap', {}).items():
        ref = ref_vals['GenCap'].get((g, period), 0)
        const -= lam * ref
        if (g, period) in master._GenCap: cut_lin += lam * master._GenCap[(g, period)]

    for g, lam in agg_duals.get('BaseloadLevel', {}).items():
        ref = ref_vals['BaseloadLevel'].get((g, period), 0)
        const -= lam * ref
        if (g, period) in master._BaseloadLevel: cut_lin += lam * master._BaseloadLevel[(g, period)]

    for (ps, lam) in agg_duals.get('CarbonAlloc', {}).items():
        # ps = (period, seg) tuple
        ref = ref_vals['CarbonAlloc'].get(ps, 0)
        const -= lam * ref
        cut_lin += lam * master._CarbonAlloc[ps]

    for g, lam in agg_duals.get('CogenCap', {}).items():
        ref = ref_vals['CogenCap'].get((g, period), 0)
        const -= lam * ref
        cut_lin += lam * master._CogenCap[(g, period)]

    for g, lam in agg_duals.get('StorEnergyCap', {}).items():
        ref = ref_vals['StorEnergyCap'].get((g, period), 0)
        const -= lam * ref
        cut_lin += lam * master._StorEnergyCap[(g, period)]

    for z, lam in agg_duals.get('LocalTDCap', {}).items():
        ref = ref_vals['LocalTDCap'].get((z, period), 0)
        const -= lam * ref
        cut_lin += lam * master._LocalTDCap[(z, period)]

    for tx, lam in agg_duals.get('TxCap', {}).items():
        ref = ref_vals['TxCap'].get((tx, period), 0)
        const -= lam * ref
        cut_lin += lam * master._TxCap[(tx, period)]

    master.addConstr(theta_p >= const + cut_lin, f'BendersCutAgg_{period}_{master._n_cuts}')
    master._n_cuts += 1


def add_benders_cut_multi(master, period, seg_idx, seg_obj, seg_duals, ref_vals, data):
    """Per-Segment (Multi) Cut on theta_ps[(period, seg_idx)]"""
    theta = master._theta_ps[(period, seg_idx)]

    const = seg_obj
    cut_lin = gp.LinExpr()

    for g, lam in seg_duals.get('GenCap', {}).items():
        ref = ref_vals['GenCap'].get((g, period), 0)
        const -= lam * ref
        if (g, period) in master._GenCap: cut_lin += lam * master._GenCap[(g, period)]

    for g, lam in seg_duals.get('BaseloadLevel', {}).items():
        ref = ref_vals['BaseloadLevel'].get((g, period), 0)
        const -= lam * ref
        if (g, period) in master._BaseloadLevel: cut_lin += lam * master._BaseloadLevel[(g, period)]

    ca_dual = seg_duals.get('CarbonAlloc', 0.0)
    if abs(ca_dual) > 1e-10 and (period, seg_idx) in master._CarbonAlloc:
        ref = ref_vals['CarbonAlloc'].get((period, seg_idx), 0)
        const -= ca_dual * ref
        cut_lin += ca_dual * master._CarbonAlloc[(period, seg_idx)]

    for g, lam in seg_duals.get('CogenCap', {}).items():
        ref = ref_vals['CogenCap'].get((g, period), 0)
        const -= lam * ref
        cut_lin += lam * master._CogenCap[(g, period)]

    for g, lam in seg_duals.get('StorEnergyCap', {}).items():
        ref = ref_vals['StorEnergyCap'].get((g, period), 0)
        const -= lam * ref
        cut_lin += lam * master._StorEnergyCap[(g, period)]

    for z, lam in seg_duals.get('LocalTDCap', {}).items():
        ref = ref_vals['LocalTDCap'].get((z, period), 0)
        const -= lam * ref
        cut_lin += lam * master._LocalTDCap[(z, period)]

    for tx, lam in seg_duals.get('TxCap', {}).items():
        ref = ref_vals['TxCap'].get((tx, period), 0)
        const -= lam * ref
        cut_lin += lam * master._TxCap[(tx, period)]

    master.addConstr(theta >= const + cut_lin, f'BendersCut_{period}s{seg_idx}_{master._n_cuts}')
    master._n_cuts += 1


# ============================================================
# 导出装机与调度 CSV 输出
# ============================================================
def export_results(data, master, seg_subs, master_sol, best_iter, out_dir):
    """从最优解导出 gen_build.csv, gen_cap.csv, dispatch.csv, summary.csv"""
    import pandas as pd
    P = data['PERIODS']; S = SEGMENTS_PER_PERIOD
    G = data['gen']
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"\n{'─'*80}")
    logger.info(f"导出装机与调度输出 (最优解 @ Iter {best_iter}) → {out_dir}")
    logger.info(f"{'─'*80}")

    # ---- gen_build.csv ----
    build_rows = []
    for (g, p), var in master._BuildGen.items():
        val = var.X
        build_rows.append({
            'GENERATION_PROJECT': g, 'PERIOD': p,
            'gen_tech': G[g]['tech'], 'gen_load_zone': G[g]['zone'],
            'gen_energy_source': G[g]['energy_source'],
            'BuildGen': round(val, 4) if val > 1e-6 else '.'
        })
    pd.DataFrame(build_rows).to_csv(os.path.join(out_dir, 'gen_build.csv'), index=False)
    build_mw = sum(var.X for var in master._BuildGen.values() if var.X > 1e-6)
    logger.info(f"  gen_build.csv: {len([r for r in build_rows if r['BuildGen']!='.'])} projects, {build_mw:,.0f} MW")

    # ---- gen_cap.csv ----
    cap_rows = []
    for g in data['GENERATORS']:
        for p in P:
            aby = [by for (g2, by) in data['pred_build'] if g2 == g and data['is_alive'](g, by, p)]
            pred = sum(data['pred_build'].get((g, by), 0) for by in aby)
            build_sum = sum(master._BuildGen[(g, pp)].X if (g, pp) in master._BuildGen else 0
                          for pp in P if pp <= p and data['is_alive'](g, pp, p))
            susp_sum = sum(master._SuspendGen[(g2, by, pp)].X
                          for (g2, by, pp) in master._GEN_BLD_SUSPEND if g2 == g and pp == p)
            cap_rows.append({
                'GENERATION_PROJECT': g, 'PERIOD': p,
                'gen_tech': G[g]['tech'], 'gen_load_zone': G[g]['zone'],
                'gen_energy_source': G[g]['energy_source'],
                'GenCapacity': round(pred + build_sum - susp_sum, 4),
                'SuspendGen_total': round(susp_sum, 4)
            })
    pd.DataFrame(cap_rows).to_csv(os.path.join(out_dir, 'gen_cap.csv'), index=False)
    logger.info(f"  gen_cap.csv: {len(cap_rows)} entries")

    # ---- dispatch.csv (汇总所有 segment 子问题) ----
    ms = extract_master_sol(master, data)
    disp_rows = []
    for p in P:
        for s in range(S):
            sub = seg_subs[(p, s)]
            update_segment_rhs(sub, ms, p, s)
            sub.optimize()
            if sub.Status == GRB.OPTIMAL:
                for (g, t), var in sub._DispatchGen.items():
                    if var.X > 1e-6:
                        disp_rows.append({
                            'GENERATION_PROJECT': g, 'TIMEPOINT': t,
                            'PERIOD': data['tp_period'][t],
                            'DispatchGen': round(var.X, 4)
                        })
                for (g, t), var in sub._ChargeStor.items():
                    if var.X > 1e-6:
                        disp_rows.append({
                            'GENERATION_PROJECT': g, 'TIMEPOINT': t,
                            'PERIOD': data['tp_period'][t],
                            'DispatchGen': round(-var.X, 4)
                        })
    pd.DataFrame(disp_rows).to_csv(os.path.join(out_dir, 'dispatch.csv'), index=False)
    logger.info(f"  dispatch.csv: {len(disp_rows):,} non-zero entries")

    # ---- summary.csv ----
    total_cost = master.ObjVal
    with open(os.path.join(out_dir, 'summary.csv'), 'w') as f:
        f.write(f'scenario,total_cost,solve_time_s\nBenders_V6,{total_cost},-1\n')
    with open(os.path.join(out_dir, 'total_cost.txt'), 'w') as f:
        f.write(f'{total_cost}\n')
    logger.info(f"  summary.csv + total_cost.txt")
    logger.info(f"  全部输出已保存到: {out_dir}/")


# ============================================================
# Benders 主循环
# ============================================================
def run_benders(data, max_iter=50, gap_tol=1e-3, output_dir=None,
                warm_start_sol=None):
    """V7: added warm_start_sol for cross-step cut seeding.
    warm_start_sol = master_sol dict from previous SCEQ step (or None).
    """
    P = data['PERIODS']; S = SEGMENTS_PER_PERIOD

    # ---- 构建持久化模型 ----
    t0 = time.time()
    master = build_persistent_master(data)

    # 构建 48 个 segment 子问题 (4 periods × 12 segments)
    seg_subs = {}
    for p in P:
        for s in range(S):
            seg_subs[(p, s)] = build_segment_subproblem(data, p, s)

    build_time = time.time() - t0
    n_subs = len(seg_subs)
    logger.info(f"\n模型构建完成 ({build_time:.1f}s): Master + {n_subs} segment 子问题\n")

    total_cuts = 0  # V7: initialize before warm-start

    # ---- V7: Warm-start from previous SCEQ step ----
    if warm_start_sol is not None:
        logger.info(">>> Warm-start: injecting cuts from previous step's solution...")
        ws_t0 = time.time()
        ws_new_cuts = 0
        # Update subproblems with warm-start investment decisions
        for p in P:
            for s in range(S):
                update_segment_rhs(seg_subs[(p, s)], warm_start_sol, p, s)
        # Solve all subproblems
        def _ws_solve_one(p, s):
            sub = seg_subs[(p, s)]
            return (p, s), solve_segment_subproblem(sub, p, s, data)
        ws_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:  # V2.1: 8→4 for multi-worker
            futures = [executor.submit(_ws_solve_one, p, s) for p in P for s in range(S)]
            for future in as_completed(futures):
                (p, s), result = future.result()
                ws_results[(p, s)] = result
        # Build and inject cuts
        ref_vals = {
            'GenCap': warm_start_sol['GenCap'],
            'BaseloadLevel': warm_start_sol['BaseloadLevel'],
            'CarbonAlloc': warm_start_sol['CarbonAlloc'],
            'CogenCap': warm_start_sol['CogenCap'],
            'StorEnergyCap': warm_start_sol['StorEnergyCap'],
            'LocalTDCap': warm_start_sol['LocalTDCap'],
            'TxCap': warm_start_sol['TxCap'],
        }
        ws_theta_p = warm_start_sol.get('theta_p', {})
        ws_theta_ps = warm_start_sol.get('theta_ps', {})
        for p in P:
            total_p_obj = sum(ws_results[(p, s)]['obj'] for s in range(S))
            # Aggregate duals
            agg_gen = defaultdict(float); agg_bl = defaultdict(float)
            agg_ca = {}; agg_cogen = defaultdict(float)
            agg_stor = defaultdict(float); agg_ltd = defaultdict(float); agg_tx = defaultdict(float)
            for s in range(S):
                r = ws_results[(p, s)]
                d = r['duals']
                for g, lam in d.get('GenCap', {}).items(): agg_gen[g] += lam
                for g, lam in d.get('BaseloadLevel', {}).items(): agg_bl[g] += lam
                if abs(d.get('CarbonAlloc', 0.0)) > 1e-10: agg_ca[(p, s)] = d['CarbonAlloc']
                for g, lam in d.get('CogenCap', {}).items(): agg_cogen[g] += lam
                for g, lam in d.get('StorEnergyCap', {}).items(): agg_stor[g] += lam
                for z, lam in d.get('LocalTDCap', {}).items(): agg_ltd[z] += lam
                for tx, lam in d.get('TxCap', {}).items(): agg_tx[tx] += lam
            agg_duals = {'GenCap': dict(agg_gen), 'BaseloadLevel': dict(agg_bl),
                         'CarbonAlloc': agg_ca, 'CogenCap': dict(agg_cogen),
                         'StorEnergyCap': dict(agg_stor), 'LocalTDCap': dict(agg_ltd),
                         'TxCap': dict(agg_tx)}
            gap_agg = total_p_obj - ws_theta_p.get(p, 0)
            if gap_agg > 1e-3:
                add_benders_cut_single(master, p, total_p_obj, agg_duals, ref_vals, data)
                ws_new_cuts += 1; total_cuts += 1
            for s in range(S):
                r = ws_results[(p, s)]
                gap_ps = r['obj'] - ws_theta_ps.get((p, s), 0)
                if gap_ps > 1e-3:
                    add_benders_cut_multi(master, p, s, r['obj'], r['duals'], ref_vals, data)
                    ws_new_cuts += 1; total_cuts += 1
        ws_time = time.time() - ws_t0
        logger.info(f">>> Warm-start complete: {ws_new_cuts} initial cuts injected "
                    f"({ws_time:.1f}s)")

    # ---- 迭代状态 ----
    LB = -float('inf'); UB = float('inf')
    total_cuts_ws = total_cuts  # remember cuts from warm-start
    total_cuts = 0; gap = 1.0
    best_sol = None
    dr_frac_total = (0, 0)
    convergence_log = []

    # ---- 打印表头 ----
    header = f"{'='*110}\n" \
             f"{'Iter':>4} | {'Master_Obj':>14} | {'Sub_Ops_Cost':>14} | {'UB':>18} | {'LB':>18} | {'GAP':>9} | {'Cuts':>5} | {'Time':>7}\n" \
             f"{'-'*110}"
    logger.info(header)

    for iteration in range(1, max_iter + 1):
        it_t0 = time.time()

        # Step 1: 求解 Master
        master.optimize()
        if master.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            logger.error(f"Master status={master.Status}, 终止")
            break

        master_sol = extract_master_sol(master, data)
        LB = master.ObjVal
        invest = master_sol['invest_cost']

        # 构建参考点
        ref_vals = {
            'GenCap': master_sol['GenCap'],
            'BaseloadLevel': master_sol['BaseloadLevel'],
            'CarbonAlloc': master_sol['CarbonAlloc'],
            'CogenCap': master_sol['CogenCap'],
            'StorEnergyCap': master_sol['StorEnergyCap'],
            'LocalTDCap': master_sol['LocalTDCap'],
            'TxCap': master_sol['TxCap'],
        }

        # Step 2: 更新所有 segment 子问题 RHS 并并行求解
        def solve_one_seg(p, s):
            sub = seg_subs[(p, s)]
            update_segment_rhs(sub, master_sol, p, s)
            return (p, s), solve_segment_subproblem(sub, p, s, data)

        seg_results = {}
        with ThreadPoolExecutor(max_workers=4) as executor:  # V2.1: 8→4 for multi-worker
            futures = [executor.submit(solve_one_seg, p, s) for p in P for s in range(S)]
            for future in as_completed(futures):
                (p, s), result = future.result()
                seg_results[(p, s)] = result

        # 检查所有 segment 是否最优
        if any(r['status'] != GRB.OPTIMAL for r in seg_results.values()):
            bad = [(ps, r['status']) for ps, r in seg_results.items() if r['status'] != GRB.OPTIMAL]
            logger.error(f"Segment subproblem(s) non-optimal: {bad[:5]}")
            break

        total_sub = sum(r['obj'] for r in seg_results.values())

        # Step 3: 更新上界
        UB_cand = invest + total_sub
        if UB_cand < UB:
            UB = UB_cand
            best_sol = {'iter': iteration, 'master': master_sol, 'seg_results': seg_results}

        # Step 4: 按 period 汇总 + 生成 Benders cuts
        new_cuts = 0
        for p in P:
            # 汇总该 period 下所有 segment 的对偶
            agg_gen = defaultdict(float)
            agg_bl = defaultdict(float)
            agg_ca = defaultdict(float)  # key: (p, s) tuple
            agg_cogen = defaultdict(float)
            agg_stor = defaultdict(float)
            agg_ltd = defaultdict(float)
            agg_tx = defaultdict(float)
            total_p_obj = 0.0

            for s in range(S):
                r = seg_results[(p, s)]
                total_p_obj += r['obj']
                d = r['duals']

                for g, lam in d.get('GenCap', {}).items():
                    agg_gen[g] += lam
                for g, lam in d.get('BaseloadLevel', {}).items():
                    agg_bl[g] += lam
                ca = d.get('CarbonAlloc', 0.0)
                if abs(ca) > 1e-10:
                    agg_ca[(p, s)] = ca
                for g, lam in d.get('CogenCap', {}).items():
                    agg_cogen[g] += lam
                for g, lam in d.get('StorEnergyCap', {}).items():
                    agg_stor[g] += lam
                for z, lam in d.get('LocalTDCap', {}).items():
                    agg_ltd[z] += lam
                for tx, lam in d.get('TxCap', {}).items():
                    agg_tx[tx] += lam

            agg_duals = {
                'GenCap': dict(agg_gen),
                'BaseloadLevel': dict(agg_bl),
                'CarbonAlloc': dict(agg_ca),
                'CogenCap': dict(agg_cogen),
                'StorEnergyCap': dict(agg_stor),
                'LocalTDCap': dict(agg_ltd),
                'TxCap': dict(agg_tx),
            }

            # —— 4a: Aggregate Single-Cut on theta_p ——
            gap_agg = total_p_obj - master_sol['theta_p'][p]
            if gap_agg > 1e-3:
                add_benders_cut_single(master, p, total_p_obj, agg_duals, ref_vals, data)
                new_cuts += 1; total_cuts += 1

            # —— 4b: Per-Segment Multi-Cut on theta_ps ——
            for s in range(S):
                r = seg_results[(p, s)]
                gap_ps = r['obj'] - master_sol['theta_ps'].get((p, s), 0)
                if gap_ps > 1e-3:
                    add_benders_cut_multi(master, p, s, r['obj'], r['duals'], ref_vals, data)
                    new_cuts += 1; total_cuts += 1

        # DR fractional 统计
        dr_f = sum(r.get('dr_frac', 0) for r in seg_results.values())
        dr_t = sum(r.get('dr_total', 0) for r in seg_results.values())
        if iteration == max_iter:
            dr_frac_total = (dr_f, dr_t)

        # Step 5: 收敛检查
        gap = (UB - LB) / (abs(UB) + 1e-10)
        it_time = time.time() - it_t0
        convergence_log.append({'iter': iteration, 'master_obj': master.ObjVal,
                                'sub_cost': total_sub, 'UB': UB, 'LB': LB,
                                'gap': gap, 'cuts': new_cuts, 'time': it_time})

        logger.info(f"{iteration:>4} | {master.ObjVal:>14,.0f} | {total_sub:>14,.0f} | "
                   f"{UB:>18,.0f} | {LB:>18,.0f} | {gap:>8.4%} | {total_cuts:>5} | {it_time:>6.1f}")

        if gap < gap_tol:
            logger.info(f"\n*** Benders 收敛! GAP={gap:.4%} < {gap_tol:.2%} at iteration {iteration} ***")
            break
        if new_cuts == 0:
            logger.info(f"\n*** Benders 收敛! 无新增割平面 at iteration {iteration} ***")
            break

    total_cuts += total_cuts_ws  # include warm-start cuts
    total_time = time.time() - t0

    # ============================================================
    # 最终报告
    # ============================================================
    logger.info(f"\n{'='*110}")
    logger.info(f"BENDERS 分解 V6 (独立 Segment 子问题) 最终报告")
    logger.info(f"{'='*110}")
    logger.info(f"  总耗时:           {total_time:,.1f} s")
    logger.info(f"  总迭代:           {iteration} 轮")
    logger.info(f"  累积 Cut 数:      {total_cuts}")
    logger.info(f"  最终 UB (总成本):  {UB:,.2f}")
    logger.info(f"  最终 LB:           {LB:,.2f}")
    logger.info(f"  最终 GAP:          {gap:.6%}")

    # 容量规划
    if best_sol:
        ms = best_sol['master']
        logger.info(f"\n{'─'*80}")
        logger.info(f"发电容量规划 (最优解 @ Iter {best_sol['iter']})")
        logger.info(f"{'─'*80}")
        logger.info(f"{'技术':<10} {'2025':>12} {'2030':>12} {'2035':>12} {'2040':>12}")
        for tech in ['coal', 'wind', 'solar', 'storage']:
            if tech == 'storage':
                tech_gens = data['STORAGE_GENS']
            else:
                tech_gens = [g for g in data['GENERATORS'] if data['gen'][g]['energy_source'] == tech]
            row = f"{tech:<10}"
            for p in P:
                cap = sum(ms['GenCap'].get((g, p), 0) for g in tech_gens if (g, p) in ms['GenCap'])
                row += f" {cap:>11,.0f}"
            logger.info(row)
        row = f"{'stor(MWh)':<10}"
        for p in P:
            cap = sum(ms['StorEnergyCap'].get((g, p), 0) for g in data['STORAGE_GENS'])
            row += f" {cap:>11,.0f}"
        logger.info(row)

        # 碳排放分配
        ca = ms.get('CarbonAlloc', {})
        if ca:
            logger.info(f"\n{'─'*80}")
            logger.info(f"碳排放分配 CarbonAlloc[p,s] (最优解)")
            logger.info(f"{'─'*80}")
            for p in P:
                total_ca = sum(ca.get((p, s), 0) for s in range(S))
                cc = data['carbon_cap'].get(p, float('inf'))
                logger.info(f"  Period {p}: Σ_s CA = {total_ca:,.0f} / cap = {cc:,.0f} tCO2/yr")
                for s in range(S):
                    v = ca.get((p, s), 0)
                    if v > 1e-3:
                        logger.info(f"    seg {s:>2}: {v:>14,.0f}")

    # DR 松弛分析
    if dr_frac_total[1] > 0:
        f, t = dr_frac_total
        logger.info(f"\n{'─'*80}")
        logger.info(f"DR 松弛分析 (最终迭代)")
        logger.info(f"{'─'*80}")
        logger.info(f"  DR 变量总数:    {t}")
        logger.info(f"  Fractional 数:  {f} ({f/t*100:.1f}%)")

    # 导出结果
    if best_sol:
        out = output_dir if output_dir else os.path.join(LOG_DIR, 'output_benders_v6')
        export_results(data, master, seg_subs, ms, best_sol['iter'], out)

    # ── 释放 Gurobi 模型资源 (防止顺序多场景运行时环境退化) ──
    master.dispose()
    for sub in seg_subs.values():
        sub.dispose()
    seg_subs.clear()

    logger.info(f"\n{'='*110}\n")

    return {
        'UB': UB, 'LB': LB, 'iterations': iteration, 'total_cuts': total_cuts,
        'total_time': total_time, 'best_sol': best_sol, 'convergence': convergence_log,
        'dr_frac': dr_frac_total
    }


# ============================================================
# 主入口
# ============================================================
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Benders V6 — Independent Segment Subproblems')
    ap.add_argument('--inputs-dir', default=None, help='Custom inputs directory')
    ap.add_argument('--outputs-dir', default=None, help='Custom outputs directory')
    ap.add_argument('--max-iter', type=int, default=100, help='Max Benders iterations')
    ap.add_argument('--gap-tol', type=float, default=1e-3, help='Convergence gap tolerance')
    args = ap.parse_args()

    if args.inputs_dir:
        INPUTS_DIR = args.inputs_dir
    else:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        INPUTS_DIR = os.path.join(BASE_DIR, 'inputs_data_add_storage')

    out_dir = args.outputs_dir if args.outputs_dir else None

    logger.info("=" * 80)
    logger.info("BENDERS 分解 V6 — 独立 Segment 子问题 + BaseloadLevel/CarbonAlloc")
    logger.info(f"  Inputs:  {INPUTS_DIR}")
    if out_dir: logger.info(f"  Outputs: {out_dir}")
    logger.info(f"  Max iter: {args.max_iter}, Gap tol: {args.gap_tol}")
    logger.info(f"  SEGMENTS: {SEGMENTS_PER_PERIOD}, THETA_PS_LB: {THETA_PS_LB}")
    logger.info("=" * 80)

    data = load_all(INPUTS_DIR)
    result = run_benders(data, max_iter=args.max_iter, gap_tol=args.gap_tol, output_dir=out_dir)

"""Shared data loading for Benders decomposition (reuses switch_gurobi.py logic)."""
import os, logging, pandas as pd
from collections import defaultdict

logger = logging.getLogger('benders.data')

def sf(val, default=0.0):
    if val is None or val == '' or val == '.': return default
    try: return float(val)
    except: return default

def load_all(inputs_dir):
    """Load all input data. Returns a dict with all model data."""
    data = {}

    # Time structure
    tp_df = pd.read_csv(os.path.join(inputs_dir, 'timepoints.csv'))
    ts_df = pd.read_csv(os.path.join(inputs_dir, 'timeseries.csv'))
    per_df = pd.read_csv(os.path.join(inputs_dir, 'periods.csv'))
    data['TIMEPOINTS'] = sorted(tp_df['timepoint_id'].tolist())
    data['PERIODS'] = sorted(per_df['INVESTMENT_PERIOD'].astype(int).tolist())
    tp_to_ts = dict(zip(tp_df['timepoint_id'], tp_df['timeseries']))
    ts_to_period = dict(zip(ts_df['TIMESERIES'], ts_df['ts_period'].astype(int)))
    data['tp_period'] = {tp: ts_to_period[tp_to_ts[tp]] for tp in data['TIMEPOINTS']}
    ts_weight = dict(zip(ts_df['TIMESERIES'], ts_df['ts_scale_to_period']))
    data['tp_weight'] = {tp: ts_weight[tp_to_ts[tp]] for tp in data['TIMEPOINTS']}
    tps_in_period = {p: [t for t in data['TIMEPOINTS'] if data['tp_period'][t]==p] for p in data['PERIODS']}
    data['tps_in_period'] = tps_in_period
    period_start = dict(zip(per_df['INVESTMENT_PERIOD'].astype(int), per_df['period_start'].astype(int)))
    period_end = dict(zip(per_df['INVESTMENT_PERIOD'].astype(int), per_df['period_end'].astype(int)))
    data['period_start'] = period_start
    data['period_len_yr'] = {p: period_end[p]-period_start[p] for p in data['PERIODS']}
    data['TIMESERIES'] = sorted(ts_df['TIMESERIES'].tolist())
    tps_in_ts = defaultdict(list)
    for tp in data['TIMEPOINTS']:
        tps_in_ts[tp_to_ts[tp]].append(tp)
    data['tps_in_ts'] = dict(tps_in_ts)
    data['ts_dur'] = dict(zip(ts_df['TIMESERIES'], ts_df['ts_duration_of_tp']))

    # Generator data
    gi_df = pd.read_csv(os.path.join(inputs_dir, 'gen_info.csv'))
    gbc_df = pd.read_csv(os.path.join(inputs_dir, 'gen_build_costs.csv'))
    gbp_df = pd.read_csv(os.path.join(inputs_dir, 'gen_build_predetermined.csv'))
    gcf_df = pd.read_csv(os.path.join(inputs_dir, 'variable_capacity_factors.csv'))

    GENERATORS = gi_df['GENERATION_PROJECT'].tolist()
    gen = {}
    for _, r in gi_df.iterrows():
        g = r['GENERATION_PROJECT']
        es = str(r.get('gen_energy_source','')).strip().lower()
        gen[g] = dict(
            tech=r.get('gen_tech',''), zone=r.get('gen_load_zone',''),
            energy_source=es,
            is_vre=sf(r.get('gen_is_variable',0))>0 or es in ('wind','solar','wnd','pv'),
            is_baseload=sf(r.get('gen_is_baseload',0))>0,
            cap_limit=sf(r.get('gen_capacity_limit_mw',0)),
            heat_rate=sf(r.get('gen_full_load_heat_rate',0)),
            vom=sf(r.get('gen_variable_om',0)),
            max_age=int(sf(r.get('gen_max_age',30))),
            forced_outage=sf(r.get('gen_forced_outage_rate',0)),
            sched_outage=sf(r.get('gen_scheduled_outage_rate',0)),
            storage_eff=sf(r.get('gen_storage_efficiency',0)),
            store_ratio=sf(r.get('gen_store_to_release_ratio',0)),
            connect_cost=sf(r.get('gen_connect_cost_per_mw',0)),
            can_suspend=sf(r.get('gen_can_suspend',0))>0,
            can_retire_early=sf(r.get('gen_can_retire_early',0))>0,
        )
        gen[g]['avail'] = ((1-gen[g]['forced_outage'])*(1-gen[g]['sched_outage'])
                           if gen[g]['is_baseload'] else 1-gen[g]['forced_outage'])

    data['GENERATORS'] = GENERATORS
    data['gen'] = gen
    data['H2_GENS'] = [g for g in GENERATORS if 'H2' in g or gen[g]['energy_source']=='hydrogen']
    data['FUEL_GENS'] = [g for g in GENERATORS if gen[g]['heat_rate']>0 and g not in data['H2_GENS']]
    data['STORAGE_GENS'] = [g for g in GENERATORS if gen[g]['storage_eff']>0]
    data['CAP_LIMITED'] = [g for g in GENERATORS if gen[g]['cap_limit']>0]
    data['VARIABLE_GENS'] = [g for g in GENERATORS if gen[g]['is_vre']]
    data['COAL_GENS'] = [g for g in data['FUEL_GENS'] if gen[g]['energy_source']=='coal']
    GENS_BY_ZONE = defaultdict(list)
    for g in GENERATORS: GENS_BY_ZONE[gen[g]['zone']].append(g)
    data['GENS_BY_ZONE'] = dict(GENS_BY_ZONE)

    # Build costs
    gen_overnight = {}; gen_fixed_om = {}; gen_store_en_cost = {}
    for _, r in gbc_df.iterrows():
        g = r['GENERATION_PROJECT']; by = int(r['build_year'])
        gen_overnight[(g,by)] = sf(r.get('gen_overnight_cost',0))
        gen_fixed_om[(g,by)] = sf(r.get('gen_fixed_om',0))
        gen_store_en_cost[(g,by)] = sf(r.get('gen_storage_energy_overnight_cost', 0))
    data['gen_overnight'] = gen_overnight
    data['gen_fixed_om'] = gen_fixed_om
    data['gen_store_en_cost'] = gen_store_en_cost

    def get_cost(gen_name, period, lookup, default=0.0):
        p_start = period_start.get(period, period)
        avail = sorted([by for (g,by) in lookup if g==gen_name])
        if not avail: return default
        best = max([by for by in avail if by<=p_start], default=None)
        if best is None: best = avail[0]
        return lookup.get((gen_name,best), default)
    data['get_cost'] = get_cost

    # NEW_GEN_BLD_YRS
    NEW_GEN_BLD_YRS = set()
    gen_bld_yrs = defaultdict(set)
    for _, r in gbc_df.iterrows():
        g = r['GENERATION_PROJECT']; by = int(r['build_year'])
        gen_bld_yrs[g].add(by)
    for g in GENERATORS:
        for by in gen_bld_yrs.get(g, set()):
            if by in data['PERIODS']:
                NEW_GEN_BLD_YRS.add((g, by))
    data['NEW_GEN_BLD_YRS'] = NEW_GEN_BLD_YRS

    # Predetermined builds
    pred_build = defaultdict(float)
    for _, r in gbp_df.iterrows():
        pred_build[(r['GENERATION_PROJECT'], int(r['build_year']))] = sf(r.get('build_gen_predetermined',0))
    data['pred_build'] = pred_build

    # is_alive
    def is_alive(g, build_year, period):
        online = period_start.get(build_year, build_year) if build_year in data['PERIODS'] else build_year
        retirement = online + gen[g]["max_age"]
        return online <= period_start.get(period, period) < retirement
    data['is_alive'] = is_alive

    # Capacity factors
    cf_data = defaultdict(dict)
    for _, r in gcf_df.iterrows():
        cf_data[r['GENERATION_PROJECT']][int(r['timepoint'])] = sf(r.get('gen_max_capacity_factor',0))
    data['cf_data'] = cf_data

    def get_cf(g, t): return cf_data[g].get(t, 1.0) if g in cf_data else 1.0
    data['get_cf'] = get_cf

    # Load
    loads_df = pd.read_csv(os.path.join(inputs_dir, 'loads.csv'))
    LOAD_ZONES = sorted(loads_df['LOAD_ZONE'].unique())
    data['LOAD_ZONES'] = LOAD_ZONES
    zone_load = defaultdict(lambda: defaultdict(float))
    for _, r in loads_df.iterrows():
        zone_load[r['LOAD_ZONE']][int(r['TIMEPOINT'])] = sf(r.get('zone_demand_mw',0))
    data['zone_load'] = zone_load

    # Fuel cost
    fuel_cost_df = pd.read_csv(os.path.join(inputs_dir, 'fuel_cost.csv'))
    fuel_price = {}
    for _, r in fuel_cost_df.iterrows():
        period = int(r.get('period', data['PERIODS'][0]))
        fuel = str(r.get('fuel','')).strip().lower()
        zone = str(r.get('load_zone','')).strip()
        fuel_price[(fuel, period, zone)] = sf(r.get('fuel_cost',0))
    data['fuel_price'] = fuel_price

    # Financials
    fin_df = pd.read_csv(os.path.join(inputs_dir, 'financials.csv'))
    for _, r in fin_df.iterrows():
        data['INTEREST_RATE'] = sf(r.get('interest_rate', 0.07))
        data['DISCOUNT_RATE'] = sf(r.get('discount_rate', 0.05))
        data['BASE_YEAR'] = int(sf(r.get('base_financial_year', 2020)))

    # Load zones
    lz_df = pd.read_csv(os.path.join(inputs_dir, 'load_zones.csv'))
    lz_data = {}
    for _, r in lz_df.iterrows():
        lz_data[r['LOAD_ZONE']] = dict(
            existing_td=sf(r.get('existing_local_td',0)),
            td_cost=sf(r.get('local_td_annual_cost_per_mw',0)))
    data['lz_data'] = lz_data

    # Lost load
    llc_df = pd.read_csv(os.path.join(inputs_dir, 'lost_load_cost.csv'))
    data['UNSERVED_COST'] = 2000.0
    for _, r in llc_df.iterrows():
        data['UNSERVED_COST'] = sf(r.get('unserved_load_penalty', 2000))

    # DR data
    dr_df = pd.read_csv(os.path.join(inputs_dir, 'dr_data.csv'))
    dr_up_lim = defaultdict(float); dr_down_lim = defaultdict(float)
    for _, r in dr_df.iterrows():
        z, t = r['LOAD_ZONE'], int(r['TIMEPOINT'])
        dr_up_lim[(z,t)] = sf(r.get('dr_shift_up_limit',0))
        dr_down_lim[(z,t)] = sf(r.get('dr_shift_down_limit',0))
    data['dr_up_lim'] = dr_up_lim
    data['dr_down_lim'] = dr_down_lim

    # DR recovery time
    dr_recov_time = defaultdict(float)
    df = pd.read_csv(os.path.join(inputs_dir, 'dr_response_recovery_data.csv'))
    for _, r in df.iterrows():
        dr_recov_time[(r['LOAD_ZONE'], int(r['TIMEPOINT']))] = sf(r.get('recovery_time', 2.0))
    data['dr_recov_time'] = dr_recov_time

    # Transmission
    tx_df = pd.read_csv(os.path.join(inputs_dir, 'transmission_lines.csv'))
    tx_data = {}
    for _, r in tx_df.iterrows():
        tx_data[r['TRANSMISSION_LINE']] = dict(
            lz1=r.get('trans_lz1',''), lz2=r.get('trans_lz2',''),
            exist=sf(r.get('existing_trans_cap',0)),
            length=sf(r.get('trans_length_km',0)),
            eff=sf(r.get('trans_efficiency',0.95)),
            cost_per_mw_km=sf(r.get('trans_capital_cost_per_mw_km',1000)))
    data['tx_data'] = tx_data

    # Cogen
    data['COGEN_HEAT_RATE'] = 20.0
    data['COGEN_FIXED_COST'] = 150000.0
    cogen_df = pd.read_csv(os.path.join(inputs_dir, 'cogen.csv'))
    for _, r in cogen_df.iterrows():
        data['COGEN_HEAT_RATE'] = sf(r.get('cogen_heat_rate', 20.0))
        data['COGEN_FIXED_COST'] = sf(r.get('cogen_fixed_cost', 150000.0))

    # Hydrogen
    h2_df = pd.read_csv(os.path.join(inputs_dir, 'hydrogen.csv'))
    h2_export_df = pd.read_csv(os.path.join(inputs_dir, 'hydrogen_export.csv'))
    h2_param = {}
    for _, r in h2_df.iterrows():
        h2_param = dict(elec_cap=sf(r.get('hydrogen_electrolyzer_capital_cost_per_mw',0)),
            elec_fom=sf(r.get('hydrogen_electrolyzer_fixed_cost_per_mw_year',0)),
            elec_kg_per_mwh=sf(r.get('hydrogen_electrolyzer_kg_per_mwh',17.8)),
            elec_life=int(sf(r.get('hydrogen_electrolyzer_life_years',20))),
            liq_cap=sf(r.get('hydrogen_liquefier_capital_cost_per_kg_per_hour',0)),
            liq_mwh_per_kg=sf(r.get('hydrogen_liquefier_mwh_per_kg',0.01)),
            liq_life=int(sf(r.get('hydrogen_liquefier_life_years',30))),
            tank_cap_cost=sf(r.get('liquid_hydrogen_tank_capital_cost_per_kg',0)),
            tank_life=int(sf(r.get('liquid_hydrogen_tank_life_years',40))))
        break
    data['h2_param'] = h2_param
    h2_export = {}
    for _, r in h2_export_df.iterrows():
        h2_export[(r['LOAD_ZONE'], r.get('DATE', r.get('TIMESERIES','')))] = sf(r.get('export_hydrogen',0))
    data['h2_export'] = h2_export

    # Carbon
    co2_intensity = {}
    for _, r in pd.read_csv(os.path.join(inputs_dir, 'fuels.csv')).iterrows():
        co2_intensity[str(r.get('fuel','')).strip().lower()] = sf(r.get('co2_intensity', 0))
    data['co2_intensity'] = co2_intensity
    carbon_cap = {}
    for _, r in pd.read_csv(os.path.join(inputs_dir, 'carbon_policies.csv')).iterrows():
        p = int(r.get('PERIOD', data['PERIODS'][0]))
        cap = sf(r.get('carbon_cap_tco2_per_yr', float('inf')))
        if cap < 1e10: carbon_cap[p] = cap
    data['carbon_cap'] = carbon_cap

    # Financial functions
    def crf(ir, t): return 1/t if ir==0 else ir/(1-(1+ir)**-t)
    def uspv(dr, t): return t if dr==0 else (1-(1+dr)**-t)/dr
    def fpv(dr, t): return (1+dr)**-t
    data['crf'] = crf
    data['uspv'] = uspv
    data['fpv'] = fpv

    bring_annual = {}
    for p in data['PERIODS']:
        bring_annual[p] = (uspv(data['DISCOUNT_RATE'], data['period_len_yr'][p]) *
                           fpv(data['DISCOUNT_RATE'], data['period_start'][p]-data['BASE_YEAR']))
    data['bring_annual'] = bring_annual

    tp_wt_yr = {t: data['tp_weight'][t]/data['period_len_yr'][data['tp_period'][t]] for t in data['TIMEPOINTS']}
    data['tp_wt_yr'] = tp_wt_yr

    logger.info(f"Data loaded: {len(GENERATORS)} gens, {len(data['TIMEPOINTS'])} tp, "
                f"{len(data['PERIODS'])} periods, {len(NEW_GEN_BLD_YRS)} new build entries")

    return data

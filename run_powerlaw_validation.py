#!/usr/bin/env python3
"""
run_powerlaw_validation
========================

One validation run of the calibrated power-law community network
(basePars_community_powerlaw.py's community_pars, from calibrate_community_powerlaw.py),
capturing:

  1. Network diagnostics -- saved to NETWORK_STATS_PATH (.npz):
       - annual_degree_years / annual_degree_values: pooled mean annual distinct-partner
         count (excluding singles, same definition as calibrate_community_powerlaw.py),
         one point per calendar year across the whole run.
       - annual_degree_dist_end: full distribution (counts, index k = k partners, last
         index = MAX_DEGREE_FOR_DIST+ overflow) of distinct partners in the FINAL 1-year
         window, among everyone active at some point in that window (includes zero).
       - instant_degree_dist_end: same shape, but for the instantaneous (snapshot) degree
         at the very last timestep, not a windowed count.
       - duration_short_months / duration_long_months: completed-partnership duration
         distributions (months) by layer -- see partnership_durations_by_type()'s
         docstring for the left/right-censoring caveats.
       - long_duration_summary: printed + stored scalars (n completed, n still active/
         censored at sim end, mean/median of completed durations) specifically for the
         long layer, since its ~55-year calibrated mean duration means most 'l' ties
         formed during a 75-year run are still ongoing (censored) at the end, not
         completed -- see the printed summary for how much of the distribution that is.

  2. Epidemic outputs -- saved to EPI_STATS_PATH (.csv), the same way run_sim.py does
     (sim.to_df(date_index=True) -> csv). This is a single seed (not run_sim.py's 5-run
     MultiSim) -- it exists to sanity-check the calibrated network end-to-end as an
     epidemic model, not to produce production results.
"""
import pathlib

import numpy as np
import sciris as sc

import hpvsim_working as hpv

from basePars_community_powerlaw import base_pars_geno
from calibrate_default_poisson import (
    _ActiveTracker, degree_from_edges, union_edges_window, active_union_window,
)

OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'
NETWORK_STATS_PATH = 'network_stats_community_powerlaw.npz'
EPI_STATS_PATH = 'powerlaw.csv'
SEEDS = 0

MAX_DEGREE_FOR_DIST = 20  # overflow bin at this value, matching natsal_analysis_working.ipynb's convention


def degree_histogram(deg, max_degree=MAX_DEGREE_FOR_DIST):
    ''' counts[k] = number of people with degree k, for k=0..max_degree-1, plus an overflow bin at max_degree '''
    return np.bincount(np.clip(deg, 0, max_degree), minlength=max_degree + 1)


def annual_degree_timeseries(nh, act, n, steps_per_year, t_end):
    '''
    Pooled mean annual distinct-partner count (excluding singles -- same definition used
    throughout calibrate_default_poisson.py/calibrate_community_powerlaw.py), evaluated
    once per calendar year across the run.
    '''
    years_idx, values = [], []
    t = steps_per_year
    while t <= t_end:
        edges = union_edges_window(nh, t, steps_per_year, 1)
        deg = degree_from_edges(edges, n)
        f_union, m_union = active_union_window(act, t, steps_per_year, 1)
        active = np.array(sorted(f_union | m_union), dtype=np.int64)
        if active.size:
            active_deg = deg[active]
            partnered = active_deg[active_deg >= 1]
            mean_deg = float(partnered.mean()) if partnered.size else float('nan')
        else:
            mean_deg = float('nan')
        years_idx.append(t)
        values.append(mean_deg)
        t += steps_per_year
    return np.array(years_idx), np.array(values)


def partnership_durations_by_type(nh, sim, layer_map):
    '''
    Completed-partnership duration distributions (months), split by layer, built by
    matching each removed edge's eid back to the delta in which it was first added. Only
    edges added strictly AFTER t=0 are used -- edges present in nh.initial_snapshot were
    formed during CommunityNetworkBackend's pre-sim burn-in, whose true formation time
    isn't recorded (see project discussion), so including them would left-censor the
    distribution toward short durations. This means the reported distribution is itself
    right-censored the other way: partnerships formed late in the run that are still
    ongoing at sim end are excluded from the arrays, but counted in `still_active`.
    '''
    added = {}  # eid -> (t_formed, layer_code), for edges added at t > 0
    for t in sorted(nh.deltas.keys()):
        delta = nh.deltas[t]
        for eid, layer in zip(delta.added_edges.eid, delta.added_edges.layer):
            added[int(eid)] = (t, int(layer))

    durations = {lkey: [] for lkey in layer_map}
    for t in sorted(nh.deltas.keys()):
        delta = nh.deltas[t]
        for eid in delta.removed_edges.eid:
            info = added.pop(int(eid), None)
            if info is None:
                continue  # part of the initial (pre-burn) snapshot -- true formation time unknown
            t_formed, layer = info
            dur_months = (sim.yearvec[t] - sim.yearvec[t_formed]) * 12.0
            durations[layer_map[layer]].append(dur_months)

    # Whatever's left in `added` is still active at sim end -- right-censored, excluded
    # from the arrays above but counted here for context.
    still_active = {lkey: 0 for lkey in layer_map}
    for t_formed, layer in added.values():
        still_active[layer_map[layer]] += 1

    return {lkey: np.array(v) for lkey, v in durations.items()}, still_active


def compute_network_stats(sim):
    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    n = len(sim.people)
    t_end = sim.npts - 1
    steps_per_year = int(round(1 / sim['dt']))
    layer_map = nh.layer_map  # e.g. ['s', 'l']

    print('Computing annual mean-degree time series...')
    years_idx, annual_degree_values = annual_degree_timeseries(nh, act, n, steps_per_year, t_end)

    print('Computing final-year and instantaneous degree distributions...')
    annual_edges_end = union_edges_window(nh, t_end, steps_per_year, 1)
    annual_deg_end = degree_from_edges(annual_edges_end, n)
    f_union, m_union = active_union_window(act, t_end, steps_per_year, 1)
    active_end = np.array(sorted(f_union | m_union), dtype=np.int64)
    annual_degree_dist_end = degree_histogram(annual_deg_end[active_end])

    instant_edges_end = nh.edges_at(t_end)
    instant_deg_end = degree_from_edges(instant_edges_end, n)
    active_now = np.where(sim.people.alive & sim.people.is_active)[0]
    instant_degree_dist_end = degree_histogram(instant_deg_end[active_now])

    print('Computing partnership-duration distributions by type...')
    durations, still_active = partnership_durations_by_type(nh, sim, layer_map)

    long_key = 'l' if 'l' in durations else layer_map[-1]
    long_dur = durations.get(long_key, np.empty(0))
    long_summary = dict(
        n_completed=int(long_dur.size),
        n_still_active_censored=int(still_active.get(long_key, 0)),
        mean_completed_months=float(long_dur.mean()) if long_dur.size else float('nan'),
        median_completed_months=float(np.median(long_dur)) if long_dur.size else float('nan'),
    )
    print(f"  long ('{long_key}') partnerships: {long_summary['n_completed']} completed "
          f"(mean={long_summary['mean_completed_months']:.1f} mo, "
          f"median={long_summary['median_completed_months']:.1f} mo), "
          f"{long_summary['n_still_active_censored']} still active/censored at sim end")

    stats = dict(
        annual_degree_years=sim.yearvec[years_idx],
        annual_degree_values=annual_degree_values,
        annual_degree_dist_end=annual_degree_dist_end,
        instant_degree_dist_end=instant_degree_dist_end,
        duration_short_months=durations.get('s', np.empty(0)),
        duration_long_months=durations.get('l', np.empty(0)),
        long_duration_n_completed=long_summary['n_completed'],
        long_duration_n_censored=long_summary['n_still_active_censored'],
        long_duration_mean_months=long_summary['mean_completed_months'],
        long_duration_median_months=long_summary['median_completed_months'],
    )
    return stats


def main():
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    pars = sc.mergedicts(base_pars_geno, dict(
        rand_seed=SEED,
        analyzers=[hpv.network_history(), _ActiveTracker()],
    ))
    sim = hpv.Sim(pars, label='community power-law (calibrated)')
    print('Created HPVsim simulation.')
    sim.run()
    print('Sim run complete.')

    stats = compute_network_stats(sim)
    network_stats_path = outdir / NETWORK_STATS_PATH
    np.savez_compressed(network_stats_path, **stats)
    print(f'Network stats saved to: {network_stats_path}')

    try:
        temp_df = sim.to_df(date_index=True)
        temp_df['Seed'] = SEED
        epi_stats_path = outdir / EPI_STATS_PATH
        temp_df.to_csv(epi_stats_path, index=True)
        print(f'Epidemic stats saved to: {epi_stats_path}')
    except Exception as e:
        print(f'Could not save epidemic results to df: {e}')

    print('Done.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
run_community_validation
=========================

One test run of the Gamma community network (basePars_community.py: ethnicity as the four
communities, Natsal partnership-end community sizes, symmetric ethnicity mixing matrix,
gamma_shape=2), the same way run_powerlaw_validation.py does for the power-law variant:

  1. Network diagnostics -- saved to NETWORK_STATS_PATH (.npz):
       - the same degree/duration objects run_powerlaw_validation.py stores (annual
         mean-degree time series, final-window annual and instantaneous degree
         distributions, completed-partnership durations by layer), computed with that
         file's helpers so the two networks stay directly comparable;
       - the five pooled Natsal targets (calibrate_default_poisson.TARGETS) plus
         cv_degree_annual (calibrate_community_powerlaw.TARGETS), measured with the same
         definitions the calibration scripts use, printed target-vs-measured;
       - community diagnostics specific to this variant: realised community sizes vs
         community_probs, the realised symmetric community joint vs the (IPF-rescaled)
         Natsal ethnicity joint that community_mixing was built from, the row-normalised
         realised mixing, the within-community edge fraction, and mean annual degree by
         community -- the last one checks the "IPF onto the margins leaves every community
         with the same mean degree" claim in basePars_community.py.

  2. Epidemic outputs -- saved to EPI_STATS_PATH (.csv) via sim.to_df(date_index=True),
     as run_sim.py does. Single seed, and at N_AGENTS below rather than base_pars'
     production 200k, so this is a smoke test of the configuration end-to-end, not a
     results run.

Community mixing is measured off CommunityNetworkBackend's own internal state via
community_testing._MixingTracker (community membership is backend-internal bookkeeping and
is not part of network_history's NetworkDelta -- see community_network.py).
"""
import pathlib

import numpy as np
import sciris as sc

import hpvsim_working as hpv

from basePars_community import (
    base_pars_geno, ETHNICITIES, community_probs, eth_joint, _rescale_to_margins,
)
from calibrate_default_poisson import (
    TARGETS as _POISSON_TARGETS, _ActiveTracker, degree_from_edges, union_edges_window,
    active_union_window, WINDOW_LONG_YEARS,
)
from community_testing import _MixingTracker
import hpvsim_working.age_community_bipartite_network_model as acbnm

# IMPORTANT: nothing power-law-related may be imported from here. powerlaw.py monkeypatches
# age_community_bipartite_network_model._sample_side_theta to its Pareto sampler at import
# time (powerlaw.py's _install_powerlaw_theta()), and that patch is global and permanent for
# the process -- so importing powerlaw.py, basePars_community_powerlaw.py,
# calibrate_community_powerlaw.py or run_powerlaw_validation.py (even only for a helper
# function or a target figure) silently turns this Gamma run into a power-law run, and with
# gamma_shape=2 it doesn't even run: the Pareto sampler requires alpha > 2. The three
# measurement helpers below are therefore duplicated from run_powerlaw_validation.py rather
# than imported from it, and the guard in main() checks the sampler is still the Gamma one.
POWERLAW_PATCH_NAME = '_sample_side_theta_powerlaw'

# calibrate_default_poisson.py's five pooled Natsal targets, plus cv_degree_annual (copied
# from calibrate_community_powerlaw.TARGETS, which can't be imported -- see above; the value
# is Natsal-3's pooled CV of annual partner count among the partnered, and the cell that
# computes it is in natsal_analysis_working.ipynb). gamma_shape is this network's only knob
# on that CV, which is what makes it the relevant extra target here.
TARGETS = dict(
    mean_degree_annual=_POISSON_TARGETS['mean_degree_annual'],
    mean_degree_5yr=_POISSON_TARGETS['mean_degree_5yr'],
    p_single=_POISSON_TARGETS['p_single'],
    p_long=_POISSON_TARGETS['p_long'],
    p_short=_POISSON_TARGETS['p_short'],
    cv_degree_annual=1.831,
)

OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

N_AGENTS = 10_000   # test run; base_pars' production value is 200k
SEED = 0

LKEY_SHORT, LKEY_LONG = 's', 'l'
MAX_DEGREE_FOR_DIST = 20  # overflow bin, matching natsal_analysis_working.ipynb's convention


def shape_tag(shape):
    return str(shape).replace('.', 'p')


# ---------------------------------------------------------------------------
# Degree/duration helpers -- duplicated from run_powerlaw_validation.py (see the
# monkeypatch note above for why they can't be imported); keep the two in sync.
# ---------------------------------------------------------------------------

def degree_histogram(deg, max_degree=MAX_DEGREE_FOR_DIST):
    ''' counts[k] = number of people with degree k, k=0..max_degree-1, plus an overflow bin '''
    return np.bincount(np.clip(deg, 0, max_degree), minlength=max_degree + 1)


def annual_degree_timeseries(nh, act, n, steps_per_year, t_end):
    '''
    Pooled mean annual distinct-partner count (excluding singles -- the definition used
    throughout the calibration scripts), evaluated once per calendar year across the run.
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
    Completed-partnership durations (months) by layer, matching each removed edge's eid back
    to the delta that added it. Only edges added strictly after t=0 count: the t=0 snapshot
    was formed during the backend's own pre-sim burn-in, with no recorded formation time, so
    including it would left-censor the distribution toward short durations. Partnerships
    still live at the end of the run are right-censored and excluded from the arrays, but
    counted in the returned `still_active`.
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
                continue  # part of the initial (pre-burn) snapshot -- formation time unknown
            t_formed, layer = info
            durations[layer_map[layer]].append((sim.yearvec[t] - sim.yearvec[t_formed]) * 12.0)

    still_active = {lkey: 0 for lkey in layer_map}
    for _t_formed, layer in added.values():
        still_active[layer_map[layer]] += 1

    return {lkey: np.array(v) for lkey, v in durations.items()}, still_active


def target_stats(sim, nh, act):
    '''
    The five pooled Natsal targets plus cv_degree_annual, measured exactly as
    calibrate_default_poisson.run_and_measure() and
    calibrate_community_powerlaw.pooled_degree_stats_excl_singles() measure them:
    p_long/p_short/p_single from the instantaneous snapshot at the last timestep, the
    degree means/CV from unions over the final 1-year and 5-year windows, excluding people
    with zero partners in the window.
    '''
    n = len(sim.people)
    t_end = sim.npts - 1
    steps_per_year = int(round(1 / sim['dt']))

    edges_now = nh.edges_at(t_end)
    active_f, active_m = act.active_female[t_end], act.active_male[t_end]
    deg_long = degree_from_edges(edges_now, n, lkey=LKEY_LONG)
    deg_short = degree_from_edges(edges_now, n, lkey=LKEY_SHORT)
    deg_total = deg_long + deg_short  # disjoint edge sets, so this is exact

    long_pool = np.concatenate([deg_long[active_f] >= 1, deg_long[active_m] >= 1])
    short_pool = np.concatenate([deg_short[active_f] >= 1, deg_short[active_m] >= 1])
    single_pool = np.concatenate([deg_total[active_f] == 0, deg_total[active_m] == 0])

    out = dict(
        p_long=float(long_pool.mean()) if long_pool.size else float('nan'),
        p_short=float(short_pool.mean()) if short_pool.size else float('nan'),
        p_single=float(single_pool.mean()) if single_pool.size else float('nan'),
    )

    for label, n_years in (('annual', 1), ('5yr', WINDOW_LONG_YEARS)):
        edges = union_edges_window(nh, t_end, steps_per_year, n_years)
        deg = degree_from_edges(edges, n)
        f_union, m_union = active_union_window(act, t_end, steps_per_year, n_years)
        active = np.array(sorted(f_union | m_union), dtype=np.int64)
        partnered = deg[active][deg[active] >= 1] if active.size else np.empty(0)
        mean = float(partnered.mean()) if partnered.size else float('nan')
        out[f'mean_degree_{label}'] = mean
        if label == 'annual':
            out['cv_degree_annual'] = (float(partnered.std() / mean)
                                       if partnered.size and mean > 0 else float('nan'))

    # Standing fraction of *edges* that are long, which is what community_pars'
    # frac_long targets (p_long above is the fraction of *people* in a long partnership).
    n_long = sum(1 for _f, _m, lkey in edges_now.values() if lkey == LKEY_LONG)
    out['standing_long_edge_frac'] = n_long / len(edges_now) if edges_now else float('nan')
    return out


def community_stats(sim, nh, act, mt):
    '''
    Community sizes, community x community mixing and mean degree by community. Sizes and
    per-person community come from the backend's live node arrays at the end of the run;
    the mixing counts are _MixingTracker's, accumulated over every timestep of the run.
    '''
    n = len(sim.people)
    t_end = sim.npts - 1
    steps_per_year = int(round(1 / sim['dt']))
    n_comm = len(ETHNICITIES)
    state = sim.network_backend._state

    # uid -> community, for everyone currently holding a U (female) or V (male) node
    comm_of = np.full(n, -1, dtype=np.int64)
    comm_of[state['u_uid']] = state['u_comm']
    comm_of[state['v_uid']] = state['v_comm']
    has_comm = comm_of >= 0
    sizes = np.bincount(comm_of[has_comm], minlength=n_comm).astype(float)
    realised_shares = sizes / sizes.sum()

    # Realised mixing. Mcomm[i, j] counts female-community i x male-community j edges,
    # pooled over all timesteps; symmetrising gives the genderless joint over partnership
    # ends, the same convention as the notebook's symmetric matrix.
    M = mt.Mcomm.astype(float)
    row_totals = M.sum(axis=1, keepdims=True)
    realised_row_cond = np.divide(M, row_totals, out=np.zeros_like(M), where=row_totals > 0)
    M_sym = M + M.T
    realised_joint = M_sym / M_sym.sum() if M_sym.sum() else M_sym
    target_joint = _rescale_to_margins(eth_joint, community_probs)
    within_frac = float(np.trace(M) / M.sum()) if M.sum() else float('nan')

    # Mean annual degree by community over the final year: the check that IPF-ing the
    # joint onto community_probs left no degree gradient between communities.
    edges = union_edges_window(nh, t_end, steps_per_year, 1)
    deg = degree_from_edges(edges, n)
    f_union, m_union = active_union_window(act, t_end, steps_per_year, 1)
    active = np.array(sorted(f_union | m_union), dtype=np.int64)
    mean_deg_by_comm = np.full(n_comm, np.nan)
    mean_deg_excl_singles_by_comm = np.full(n_comm, np.nan)
    for c in range(n_comm):
        members = active[comm_of[active] == c]
        if not members.size:
            continue
        member_deg = deg[members]
        mean_deg_by_comm[c] = float(member_deg.mean())
        partnered = member_deg[member_deg >= 1]
        if partnered.size:
            mean_deg_excl_singles_by_comm[c] = float(partnered.mean())

    return dict(
        community_realised_shares=realised_shares,
        community_target_shares=np.asarray(community_probs, dtype=float),
        community_joint_realised=realised_joint,
        community_joint_target=target_joint,
        community_row_conditional=realised_row_cond,
        community_within_edge_frac=within_frac,
        community_mean_annual_degree=mean_deg_by_comm,
        community_mean_annual_degree_excl_singles=mean_deg_excl_singles_by_comm,
    )


def compute_network_stats(sim):
    ''' Mirrors run_powerlaw_validation.compute_network_stats, plus the community panels '''
    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    mt = sim.get_analyzer('mixing_tracker')
    n = len(sim.people)
    t_end = sim.npts - 1
    steps_per_year = int(round(1 / sim['dt']))
    layer_map = nh.layer_map

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

    print('Computing target statistics and community diagnostics...')
    measured = target_stats(sim, nh, act)
    comm = community_stats(sim, nh, act, mt)

    stats = dict(
        annual_degree_years=sim.yearvec[years_idx],
        annual_degree_values=annual_degree_values,
        annual_degree_dist_end=annual_degree_dist_end,
        instant_degree_dist_end=instant_degree_dist_end,
        duration_short_months=durations.get(LKEY_SHORT, np.empty(0)),
        duration_long_months=durations.get(LKEY_LONG, np.empty(0)),
        duration_short_n_censored=still_active.get(LKEY_SHORT, 0),
        duration_long_n_censored=still_active.get(LKEY_LONG, 0),
        **{f'measured_{k}': v for k, v in measured.items()},
        **comm,
    )
    return stats, measured, comm, durations, still_active


def report(sim, measured, comm, durations, still_active):
    cp = sim['community_pars']
    print('\n=== Targets vs measured (final-window definitions from the calibration scripts) ===')
    for key in ('mean_degree_annual', 'mean_degree_5yr', 'p_single', 'p_long', 'p_short',
                'cv_degree_annual'):
        target = TARGETS.get(key)
        value = measured[key]
        ratio = value / target if target else float('nan')
        print(f'  {key:<20} target={target!s:<8} measured={value:8.3f}  ratio={ratio:5.2f}')

    print('\n=== Network-input parameters vs realised ===')
    print(f"  standing long edge fraction  input frac_long={cp['frac_long']:.3f}  "
          f"measured={measured['standing_long_edge_frac']:.3f}")
    for lkey, mean_par in ((LKEY_SHORT, 'D_mean_short'), (LKEY_LONG, 'D_mean_long')):
        dur = durations.get(lkey, np.empty(0))
        label = 'short' if lkey == LKEY_SHORT else 'long'
        if dur.size:
            print(f"  {label:<5} duration (months)      input {mean_par}={cp[mean_par]:.1f}  "
                  f"measured mean={dur.mean():8.1f} median={np.median(dur):8.1f}  "
                  f"n_completed={dur.size} n_censored={still_active.get(lkey, 0)}")
        else:
            print(f"  {label:<5} duration (months)      input {mean_par}={cp[mean_par]:.1f}  "
                  f"no completed partnerships observed (n_censored={still_active.get(lkey, 0)})")
    print(f"  gamma shape={cp['gamma_shape']} -> propensity CV = 1/sqrt(shape) = "
          f"{1 / np.sqrt(cp['gamma_shape']):.3f}")

    print('\n=== Communities (ethnicity) ===')
    print(f"  {'community':<10} {'target share':>13} {'realised':>10} "
          f"{'mean ann. deg':>14} {'excl. singles':>14}")
    for i, eth in enumerate(ETHNICITIES):
        print(f"  {eth:<10} {comm['community_target_shares'][i]:13.4f} "
              f"{comm['community_realised_shares'][i]:10.4f} "
              f"{comm['community_mean_annual_degree'][i]:14.3f} "
              f"{comm['community_mean_annual_degree_excl_singles'][i]:14.3f}")
    print(f"  within-community edge fraction: {comm['community_within_edge_frac']:.4f} "
          f"(target = trace of the joint = {np.trace(comm['community_joint_target']):.4f})")

    print('\n  Symmetric community joint -- target (IPF-rescaled Natsal) vs realised:')
    header = '  ' + ' ' * 10 + ''.join(f'{eth:>20}' for eth in ETHNICITIES)
    print(header)
    for i, eth in enumerate(ETHNICITIES):
        cells = ''.join(f"{comm['community_joint_target'][i, j]:9.4f}/"
                        f"{comm['community_joint_realised'][i, j]:<10.4f}"
                        for j in range(len(ETHNICITIES)))
        print(f'  {eth:<10}{cells}')
    max_dev = float(np.max(np.abs(comm['community_joint_realised'] - comm['community_joint_target'])))
    print(f'  max |realised - target| over the 16 cells: {max_dev:.4f}')
    return


def main():
    if acbnm._sample_side_theta.__name__ == POWERLAW_PATCH_NAME:
        raise RuntimeError(
            'age_community_bipartite_network_model._sample_side_theta has been replaced by '
            "powerlaw.py's Pareto sampler, so this would not be a Gamma run. Something in "
            'the import graph pulled in powerlaw.py -- see the note at the top of this file.'
        )

    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    pars = sc.mergedicts(base_pars_geno, dict(
        n_agents=N_AGENTS,
        rand_seed=SEED,
        community_results=True,  # also store network + epidemic results stratified by ethnicity
        analyzers=[hpv.network_history(), _ActiveTracker(), _MixingTracker()],
    ))
    tag = shape_tag(pars['community_pars']['gamma_shape'])
    network_stats_path = outdir / f'network_stats_community_gamma{tag}.npz'
    epi_stats_path = outdir / f'community_gamma{tag}_test.csv'

    sim = hpv.Sim(pars, label=f'community network (ethnicity, gamma shape '
                              f"{pars['community_pars']['gamma_shape']})")
    print(f"Created HPVsim simulation: n_agents={N_AGENTS}, seed={SEED}, "
          f"gamma_shape={pars['community_pars']['gamma_shape']}.")
    sim.run()
    print('Sim run complete.')

    stats, measured, comm, durations, still_active = compute_network_stats(sim)
    np.savez_compressed(network_stats_path, **stats)
    print(f'Network stats saved to: {network_stats_path}')

    report(sim, measured, comm, durations, still_active)

    try:
        temp_df = sim.to_df(date_index=True, by_community=True)
        temp_df['Seed'] = SEED
        temp_df.to_csv(epi_stats_path, index=True)
        print(f'\nEpidemic stats saved to: {epi_stats_path}')
    except Exception as e:
        print(f'\nCould not save epidemic results to df: {e}')

    print('Done.')


if __name__ == '__main__':
    main()

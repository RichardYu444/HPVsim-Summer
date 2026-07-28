#!/usr/bin/env python3
"""
calibrate_community_powerlaw
=============================

Calibration harness for the age+community bipartite network (powerlaw.py's power-law
variant of hpvsim_working/community_network.py's CommunityNetworkBackend), built the same
way as calibrate_default_poisson.py calibrates HPVsim's *default* network backend: same
pooled Natsal-style targets, same measurement machinery (a network_history analyzer plus
an active-population tracker, windowed edge unions for the 1yr/5yr distinct-partner
counts, instantaneous snapshots for the relationship-status proportions), and the same
simple proportional (fixed-point) knob search -- see calibrate_default_poisson.py's
docstring for the exact target definitions and the general calibration strategy; only
what differs is covered below.

Five knobs instead of six, because CommunityNetworkBackend's partnership formation is
structurally different from 'default's layer_probs/m_partners/f_partners machinery (see
powerlaw.py / recreation.py docstrings for the full rationale):

    'default' (calibrate_default_poisson.py)        'community' (this script)
    ------------------------------------------      --------------------------------
    m_scale (layer_probs['m'] scale)                 frac_long -- ONE split knob. The
    c_scale (layer_probs['c'] scale)                  community backend has a single
                                                       continuous formation stream divided
                                                       short/long by frac_long; there are
                                                       no two independent per-layer
                                                       participation scales to tune
                                                       p_long and p_short separately, so
                                                       (as with calibrate_default_poisson's
                                                       own p_single) p_short is reported
                                                       each round for comparison but is not
                                                       given its own dedicated knob.
    m_partners_c_par1 (male mean degree)             mean_partners_per_year -- ONE knob.
    f_partners_c_par1 (female mean degree)            Bipartite edges couple the two
                                                       sides 1-1, so there is no separate
                                                       male/female mean-degree knob the
                                                       way 'default' structurally needs
                                                       one (poisson1 vs poisson).
    dur_pship_c_par1 (casual duration)               D_mean_short (short-tie duration)
    dur_pship_m_par1 (marital duration)              D_mean_long  (long-tie duration)
    (no equivalent -- 'default's poisson/                gamma_shape -- the power-law tail
     poisson1 partner-count distributions have             index (alpha, see powerlaw.py).
     a fixed, non-tunable dispersion once par1 is           Community's Gamma/power-law
     set)                                                   propensity heterogeneity is a
                                                             genuinely free extra knob, so it
                                                             gets its own target: the
                                                             coefficient of variation (CV) of
                                                             the annual distinct-partner count
                                                             among partnered people (see
                                                             TARGETS['cv_degree_annual']
                                                             below). Smaller alpha
                                                             -> heavier tail -> higher CV, so
                                                             the update is inverted the same
                                                             way duration is.

Everything else -- n_communities, age mixing, acts/age_act_pars/condoms (mirrored from
'default', see powerlaw.py), all disease/genotype pars -- stays at powerlaw.py's own
defaults, same "don't touch things without a calibration target" spirit as
calibrate_default_poisson.py.

TARGETS is its own dict here (rather than the imported one used directly), for the same
clarity reason calibrate_default_poisson.py keeps all its target figures in one place --
but it's built FROM that file's TARGETS, plus one extra entry (cv_degree_annual, see
below), so mean_degree_annual/mean_degree_5yr/p_single/p_long/p_short stay identical
between the two networks -- see calibrate_default_poisson.py for the Natsal sourcing of
those five. IMPORTANT: as there, TARGETS['mean_degree_5yr'] must be filled in with a real
pooled Natsal figure before this script will run (calibrate_default_poisson.py raises if
its own copy isn't; this script reuses that same guard on its own TARGETS).
TARGETS['cv_degree_annual'] is sourced from Natsal-3 (see its definition below for the
computation and natsal_analysis_working.ipynb for the cell that reproduces it).
"""

import numpy as np
import sciris as sc

import hpvsim_working as hpv

from powerlaw import make_sim, COMMUNITY_PARS
from calibrate_default_poisson import (
    TARGETS as _POISSON_TARGETS, _ActiveTracker, degree_from_edges, union_edges_window,
    active_union_window, clipped_ratio,
    N_AGENTS, LOCATION, START, DT, BURN_IN_YEARS, MEASURE_YEARS, WINDOW_LONG_YEARS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEED = 1
CALIBRATE_ITERS = 6
MAX_STEP = 2.0        # clip each round's rescale factor to [1/MAX_STEP, MAX_STEP]

LKEY_SHORT, LKEY_LONG = 's', 'l'   # community's short/long <-> default's casual/marital

# The Pareto sampler (powerlaw.py's _sample_side_theta_powerlaw) requires alpha > 2 for
# finite variance; keep a safety margin away from that pole so a bad round can't produce
# an invalid knob value.
GAMMA_SHAPE_FLOOR = 2.05

# Same five pooled targets as calibrate_default_poisson.py (copied from its TARGETS dict,
# not just referenced, so this file's own TARGETS is self-contained), plus one extra:
#
#   cv_degree_annual: pooled coefficient of variation (weighted std/mean) of the number of
#   distinct heterosexual partners in the past year (Natsal-3's het1yr), among respondents
#   reporting >=1 -- the same "excluding singles" restriction mean_degree_annual/p_single
#   use elsewhere in this project. It has no 'default'-network counterpart (poisson/
#   poisson1 have a fixed, non-tunable dispersion once their mean is set) -- it exists
#   here because gamma_shape/alpha is a genuinely free extra knob for the community
#   network's power-law propensity heterogeneity, and needs its own target to pin down.
#
#   Computed in natsal_analysis_working.ipynb (the cell directly after the existing
#   main_degree_col/summary_table cell, reusing that notebook's own het_degree frame and
#   weighted_mean/weighted_var helpers): per sex, restrict to het1yr_clean >= 1, take the
#   weighted mean and variance, and divide (CV = sqrt(var)/mean); pool the two sexes as an
#   unweighted mean, matching mean_degree_annual/p_single's own pooling convention. That
#   gives CV_female=2.173 (n=6591), CV_male=1.488 (n=4749), pooled=1.831. Note this is
#   driven by a small number of very high reported counts (max reported = 150 for women,
#   55 for men, out of ~11,000 partnered respondents) -- the same "heavily right-skewed by
#   a few extreme reporters" caution the notebook already gives for lifetime counts. That
#   heavy tail is exactly the real-world structure a power-law (rather than Gamma/Poisson)
#   propensity distribution is meant to capture, so it's used as-is rather than trimmed.
TARGETS = dict(
    mean_degree_annual=_POISSON_TARGETS['mean_degree_annual'],
    mean_degree_5yr=_POISSON_TARGETS['mean_degree_5yr'],
    p_single=_POISSON_TARGETS['p_single'],
    p_long=_POISSON_TARGETS['p_long'],
    p_short=_POISSON_TARGETS['p_short'],
    cv_degree_annual=1.831,
)

# Starting values = powerlaw.py's own COMMUNITY_PARS defaults for the knobs we're allowed
# to move.
INIT_KNOBS = dict(
    mean_partners_per_year=COMMUNITY_PARS['mean_partners_per_year'],
    frac_long=COMMUNITY_PARS['frac_long'],
    D_mean_short=COMMUNITY_PARS['D_mean_short'],
    D_mean_long=COMMUNITY_PARS['D_mean_long'],
    gamma_shape=COMMUNITY_PARS['gamma_shape'],
)


# ---------------------------------------------------------------------------
# Build pars / run / measure
# ---------------------------------------------------------------------------

def pooled_degree_stats_excl_singles(nh, act, t_end, steps_per_year, n_years, n):
    '''
    Pooled (both sexes) mean AND coefficient of variation of degree over an n_years
    window, excluding zero-degree people. The mean feeds mean_degree_annual/5yr; the CV
    (only meaningful/used for the 1-year window) feeds the power-law tail-index
    (gamma_shape/alpha) calibration below.
    '''
    edges = union_edges_window(nh, t_end, steps_per_year, n_years)
    deg = degree_from_edges(edges, n)
    f_union, m_union = active_union_window(act, t_end, steps_per_year, n_years)
    active = np.array(sorted(f_union | m_union), dtype=np.int64)
    if not active.size:
        return float('nan'), float('nan')
    active_deg = deg[active]
    partnered = active_deg[active_deg >= 1]
    if not partnered.size:
        return float('nan'), float('nan')
    mean = float(partnered.mean())
    cv = float(partnered.std() / mean) if mean > 0 else float('nan')
    return mean, cv


def build_sim(knobs, n_years):
    '''
    powerlaw.py's own COMMUNITY_PARS with only the five calibration knobs overridden --
    everything else (n_communities, mirrored acts/age_act_pars/condoms, mixing) stays at
    powerlaw.py's package defaults.
    '''
    community_pars = sc.mergedicts(COMMUNITY_PARS, dict(
        mean_partners_per_year=knobs['mean_partners_per_year'],
        frac_long=knobs['frac_long'],
        D_mean_short=knobs['D_mean_short'],
        D_mean_long=knobs['D_mean_long'],
        gamma_shape=knobs['gamma_shape'],
    ))
    return make_sim(
        n_agents=N_AGENTS, location=LOCATION, start=START, n_years=n_years, dt=DT,
        rand_seed=SEED, verbose=0, community_pars=community_pars,
        analyzers=[hpv.network_history(), _ActiveTracker()],
    )


def run_and_measure(knobs):
    n_years = BURN_IN_YEARS + max(MEASURE_YEARS, WINDOW_LONG_YEARS)
    sim = build_sim(knobs, n_years)
    sim.run()

    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    n = len(sim.people)
    t_end = sim.npts - 1
    steps_per_year = int(round(1 / DT))

    # --- Instantaneous snapshot at the end of the run: relationship-status proportions ---
    edges_now = nh.edges_at(t_end)
    active_f, active_m = act.active_female[t_end], act.active_male[t_end]
    deg_long = degree_from_edges(edges_now, n, lkey=LKEY_LONG)
    deg_short = degree_from_edges(edges_now, n, lkey=LKEY_SHORT)
    deg_total = deg_long + deg_short  # 's' and 'l' are disjoint edge sets, so this is exact

    long_pool = np.concatenate([deg_long[active_f] >= 1, deg_long[active_m] >= 1])
    short_pool = np.concatenate([deg_short[active_f] >= 1, deg_short[active_m] >= 1])
    single_pool = np.concatenate([deg_total[active_f] == 0, deg_total[active_m] == 0])
    p_long = float(long_pool.mean()) if long_pool.size else float('nan')
    p_short = float(short_pool.mean()) if short_pool.size else float('nan')
    p_single = float(single_pool.mean()) if single_pool.size else float('nan')

    # --- Windowed unions: pooled mean degree (+ CV, annual only) over the final 1-year and
    #     WINDOW_LONG_YEARS-year periods, excluding people who had zero partners in that window ---
    mean_degree_annual, cv_degree_annual = pooled_degree_stats_excl_singles(
        nh, act, t_end, steps_per_year, 1, n)
    mean_degree_5yr, _ = pooled_degree_stats_excl_singles(
        nh, act, t_end, steps_per_year, WINDOW_LONG_YEARS, n)

    return dict(
        mean_degree_annual=mean_degree_annual,
        mean_degree_5yr=mean_degree_5yr,
        cv_degree_annual=cv_degree_annual,
        p_single=p_single,
        p_long=p_long,
        p_short=p_short,
    )


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

def print_row(label, stats):
    cv = stats.get('cv_degree_annual', float('nan'))
    print(f"{label:>10} | "
          f"deg_annual={stats['mean_degree_annual']:.3f} deg_5yr={stats['mean_degree_5yr']:.3f} "
          f"cv_annual={cv:.3f} | "
          f"long={stats['p_long']:.3f} short={stats['p_short']:.3f} single={stats['p_single']:.3f}")


def calibrate():
    if TARGETS['mean_degree_5yr'] is None:
        raise ValueError(
            "TARGETS['mean_degree_5yr'] is not set. Fill in the real pooled Natsal figure "
            "(mean partners in the last 5 years, excluding singles) at the top of "
            "calibrate_default_poisson.py before running -- this script imports TARGETS "
            "from there so both networks are calibrated against the same numbers."
        )

    knobs = sc.dcp(INIT_KNOBS)

    print('=' * 100)
    print('COMMUNITY NETWORK (power-law) -- simple proportional calibration')
    print('=' * 100)
    print_row('target', TARGETS)

    for it in range(1, CALIBRATE_ITERS + 1):
        stats = run_and_measure(knobs)
        print_row(f'iter {it}', stats)
        print(f"{'':>10} | knobs: mean_partners_per_year={knobs['mean_partners_per_year']:.3f} "
              f"frac_long={knobs['frac_long']:.3f} "
              f"D_mean_short={knobs['D_mean_short']:.3f} "
              f"D_mean_long={knobs['D_mean_long']:.3f} "
              f"gamma_shape(alpha)={knobs['gamma_shape']:.3f}")

        # mean_partners_per_year drives the pooled annual distinct-partner count (analogous
        # to m_partners_c_par1/f_partners_c_par1 -- but a single knob, see module docstring).
        knobs['mean_partners_per_year'] *= clipped_ratio(
            TARGETS['mean_degree_annual'], stats['mean_degree_annual'])

        # frac_long is nudged on its odds scale toward the target standing long-partnership
        # fraction (analogous to m_scale, but a single split knob -- see module docstring;
        # p_short has no dedicated knob and is reported for comparison only, same as
        # calibrate_default_poisson's p_single).
        target_p_long = float(np.clip(TARGETS['p_long'], 1e-6, 1 - 1e-6))
        cur_frac_long = float(np.clip(knobs['frac_long'], 1e-6, 1 - 1e-6))
        meas_p_long = stats['p_long']
        if meas_p_long == meas_p_long and 0.0 < meas_p_long < 1.0:  # not NaN, not 0/1
            odds_ratio = clipped_ratio(
                target_p_long / (1.0 - target_p_long),
                meas_p_long / (1.0 - meas_p_long))
            new_odds = (cur_frac_long / (1.0 - cur_frac_long)) * odds_ratio
            knobs['frac_long'] = new_odds / (1.0 + new_odds)

        # Inverted: longer duration means LOWER 5-year distinct-partner count, so when
        # measured > target we need to *increase* duration, not decrease it. Split evenly
        # (sqrt) across the two duration knobs since both contribute turnover to
        # mean_degree_5yr -- same logic as calibrate_default_poisson's dur_pship knobs.
        dur_ratio = clipped_ratio(stats['mean_degree_5yr'], TARGETS['mean_degree_5yr']) ** 0.5
        knobs['D_mean_short'] *= dur_ratio
        knobs['D_mean_long'] *= dur_ratio

        # Inverted the same way as duration: a bigger alpha means a THINNER tail / lower CV,
        # so when measured CV > target we need to INCREASE alpha, not decrease it. Floored
        # at GAMMA_SHAPE_FLOOR since the Pareto sampler requires alpha > 2 for finite variance.
        meas_cv = stats['cv_degree_annual']
        if meas_cv == meas_cv and meas_cv > 0:  # not NaN
            cv_ratio = clipped_ratio(meas_cv, TARGETS['cv_degree_annual'])
            knobs['gamma_shape'] = max(knobs['gamma_shape'] * cv_ratio, GAMMA_SHAPE_FLOOR)

    print('-' * 100)
    print('Final calibrated knobs:')
    for k, v in knobs.items():
        print(f'  {k} = {v:.4f}')
    print('Re-run run_and_measure(knobs) with more agents/years to confirm before using these '
          'in a real sim.')
    return knobs


if __name__ == '__main__':
    calibrate()

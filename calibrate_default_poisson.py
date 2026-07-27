#!/usr/bin/env python3
"""
calibrate_default_poisson
==========================

Simple calibration harness for HPVsim's *default* network backend, see hpvsim_working/network.py), whose partner-count
distributions are Poisson / Poisson+1 for the casual ('c') layer

Targets (all non-gendered -- one pooled-population figure each):

    mean_degree_annual : mean number of *distinct* partners over the final 1-year window,
                          among people who had >=1 partner in that window (singles excluded)
    mean_degree_5yr     : same, but over the final 5-year window
    p_single            : proportion of the active population with zero current partners
                           (either layer) at a single point in time
    p_long              : proportion of the active population currently in a long-term
                           (marital-layer) partnership
    p_short             : proportion of the active population currently in a short-term
                           (casual-layer) partnership

p_long/p_short are not mutually exclusive and don't need to sum to 1 (someone can be in both,
or neither -- the "neither" share is p_single, roughly).

Six parameters are treated as directly responsible for these targets, and are the only things
this script adjusts (everything else -- age mixing, acts, condoms, age_act_pars, the
marital-layer m_partners/f_partners par1, both dur_pship shape parameters, all disease/genotype
pars, etc -- stays at its package default, see parameters.py):

    - layer_probs['m'] scale     -> proportion in a long-term (marital) relationship right now
    - layer_probs['c'] scale     -> proportion in a short-term/one-off (casual) relationship now
    - m_partners['c']['par1']    -> male contribution to pooled annual mean degree (poisson1)
    - f_partners['c']['par1']    -> female contribution to pooled annual mean degree (poisson)
    - dur_pship['c']['par1']     -> casual-partnership duration (lognormal)
    - dur_pship['m']['par1']     -> marital-partnership duration (neg_binomial)

dur_pship['c'] and dur_pship['m'] are what separate the annual window from the 5-year one --
both layers' partnerships can dissolve and reform within 5 years, so both contribute turnover
to the pooled mean_degree_5yr figure. Each round, the mean_degree_5yr correction is split evenly
between them (each knob gets the square root of the ratio, see clipped_ratio calls below) so the
two knobs' combined effect is roughly the size of a single-knob correction, rather than
double-applying it.

Note m_partners['c'] and f_partners['c'] stay as two separate knobs -- the 'default' network
structurally requires separate male/female casual-partner-count distributions (poisson1 vs
poisson) -- but both are now driven toward the *same* pooled mean_degree_annual target instead
of their own per-sex target, since the targets themselves are non-gendered.

p_single isn't given its own dedicated knob: with only two layer_probs scales available and
both already spoken for (p_long, p_short), there's no free layer-participation parameter left to
target it independently. It's reported each round for comparison, same as in the previous
(gendered) version of this script.

IMPORTANT: TARGETS['mean_degree_5yr'] has no package default -- it's a real epidemiological
figure (pooled mean number of partners in the past 5 years, excluding those reporting 0) that
needs to be filled in from Natsal before this script will run. See the TARGETS dict below.

This is a simple proportional (fixed-point) search, not a formal optimiser: each round, every
knob is rescaled by (target / measured) -- or (measured / target) for dur_pship, since duration
and degree move in opposite directions -- clipped to a modest per-round step so it can't
overshoot wildly, and the sim is re-run. Good enough given the mild coupling between knobs -- run
CALIBRATE_ITERS rounds and check the printed table converges; re-run with more iterations/agents
if it hasn't.
"""

import numpy as np
import sciris as sc

import hpvsim_working as hpv
from hpvsim_working import parameters as hppar

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_AGENTS = 20_000
LOCATION = 'united kingdom'
START = 1980
BURN_IN_YEARS = 20    # let the network reach equilibrium before measuring
MEASURE_YEARS = 1     # width of the annual-degree measurement window at the end of the run
WINDOW_LONG_YEARS = 5 # width of the long-window (5-year) measurement window at the end of the run
DT = 0.25
SEED = 1

CALIBRATE_ITERS = 6
MAX_STEP = 2.0        # clip each round's rescale factor to [1/MAX_STEP, MAX_STEP]

LKEY_CASUAL, LKEY_MARITAL = 'c', 'm'

# dur_pship['c']/['m'] are a lognormal and a neg_binomial distribution respectively; only their
# location parameter (par1) is calibrated. Their spread (par2) stays at the package default.
DUR_PSHIP_C_PAR2 = 2.0  # package default, see parameters.py layer_defaults['default']['dur_pship']['c']
DUR_PSHIP_M_PAR2 = 3.0  # package default, see parameters.py layer_defaults['default']['dur_pship']['m']

TARGETS = dict(
    # Pooled (unweighted mean of the female/male Natsal figures from the previous, gendered
    # version of this script: 1.04 and 1.27). Replace with the real pooled/weighted Natsal
    # figure if you have one -- this is a placeholder average, not a primary-source number.
    mean_degree_annual=1.4,
    # TODO(user): pooled Natsal "mean number of partners in the last 5 years", among people
    # reporting >=1 (i.e. excluding singles) -- no source figure has been supplied yet.
    mean_degree_5yr=2.5,
    # Pooled (unweighted mean of the previous female/male figures: 0.223 and 0.179).
    p_single=(0.223 + 0.179) / 2,
    p_long=0.618,
    p_short=0.278,
)

# Starting values = the package defaults for the knobs we're allowed to move (parameters.py,
# layer_defaults['default']).
INIT_KNOBS = dict(
    m_scale=1.0,
    c_scale=1.0,
    m_partners_c_par1=0.5,
    f_partners_c_par1=1.0,
    dur_pship_c_par1=1.0,
    dur_pship_m_par1=80.0,
)


# ---------------------------------------------------------------------------
# Active-population tracker (mirrors default_network_testing.py's _ActiveTracker: network_history
# only knows about literal HPVsim birth/death, not sexual debut, so "who's currently eligible" has
# to come from sim.people directly)
# ---------------------------------------------------------------------------

class _ActiveTracker(hpv.Analyzer):
    def __init__(self, label='active_tracker'):
        super().__init__(label=label)
        self.active_female = {}
        self.active_male = {}
        return

    def initialize(self, sim):
        super().initialize(sim)
        self._record(sim, t=0)
        return

    def apply(self, sim):
        self._record(sim, t=sim.t)
        return

    def _record(self, sim, t):
        ppl = sim.people
        act = ppl.is_active
        self.active_female[t] = np.where(act & ppl.is_female)[0].astype(np.int64)
        self.active_male[t] = np.where(act & ppl.is_male)[0].astype(np.int64)
        return


# ---------------------------------------------------------------------------
# Network reconstruction helpers
# ---------------------------------------------------------------------------

def degree_from_edges(edges, n, lkey=None):
    ''' Pooled degree array (length n) from an {eid: (f, m, lkey)} dict, optionally by layer '''
    deg = np.zeros(n, dtype=np.int64)
    for f, m, layer in edges.values():
        if lkey is not None and layer != lkey:
            continue
        deg[f] += 1
        deg[m] += 1
    return deg


def union_edges_window(nh, t_end, steps_per_year, n_years):
    '''
    Union of edges live at any point during the final ``n_years`` years ending at ``t_end`` --
    i.e. every distinct partnership held at some point in that window, matching the "mean
    degree over N years" target's definition (distinct partners in N years), not a snapshot at
    a single instant. n_years=1 gives the annual window, n_years=WINDOW_LONG_YEARS the long one.
    '''
    steps = n_years * steps_per_year
    t_start = t_end - steps + 1
    union = dict(nh.edges_at(t_start - 1)) if t_start > 0 else {}
    for t in range(max(t_start, 0), t_end + 1):
        delta = nh.deltas.get(t)
        if delta is None:
            continue
        for eid, f, m, layer in zip(delta.added_edges.eid, delta.added_edges.f,
                                     delta.added_edges.m, delta.added_edges.layer):
            union[int(eid)] = (int(f), int(m), nh.layer_map[layer])
    return union


def active_union_window(act, t_end, steps_per_year, n_years):
    ''' Union of active uids observed at any point during the final n_years years '''
    steps = n_years * steps_per_year
    t_start = max(t_end - steps + 1, 0)
    f_union, m_union = set(), set()
    for t in range(t_start, t_end + 1):
        f_union.update(act.active_female[t].tolist())
        m_union.update(act.active_male[t].tolist())
    return f_union, m_union


def pooled_mean_degree_excl_singles(nh, act, t_end, steps_per_year, n_years, n):
    ''' Pooled (both sexes) mean degree over an n_years window, excluding zero-degree people '''
    edges = union_edges_window(nh, t_end, steps_per_year, n_years)
    deg = degree_from_edges(edges, n)
    f_union, m_union = active_union_window(act, t_end, steps_per_year, n_years)
    active = np.array(sorted(f_union | m_union), dtype=np.int64)
    if not active.size:
        return float('nan')
    active_deg = deg[active]
    partnered = active_deg[active_deg >= 1]
    return float(partnered.mean()) if partnered.size else float('nan')


# ---------------------------------------------------------------------------
# Build pars / run / measure
# ---------------------------------------------------------------------------

def build_pars(knobs, n_years):
    '''
    Package-default 'default' layer_probs, m_partners['c'], f_partners['c'], dur_pship['c']/['m'],
    with only the six calibration knobs overridden. reset_layer_pars() merges dict-of-layer-keys
    parameters key-by-key (see parameters.py), so passing only the 'c' entry for
    m_partners/f_partners leaves the marital-layer 'm' entry at its package default untouched.
    '''
    default_mixing, default_layer_probs = hppar.get_mixing('default')

    lp_m = sc.dcp(default_layer_probs['m'])
    lp_c = sc.dcp(default_layer_probs['c'])
    lp_m[1:, :] = np.clip(lp_m[1:, :] * knobs['m_scale'], 0.0, 1.0)
    lp_c[1:, :] = np.clip(lp_c[1:, :] * knobs['c_scale'], 0.0, 1.0)

    pars = dict(
        network='default',
        n_agents=N_AGENTS,
        location=LOCATION,
        start=START,
        n_years=n_years,
        dt=DT,
        rand_seed=SEED,
        verbose=0,
        layer_probs=dict(m=lp_m, c=lp_c),
        m_partners=dict(c=dict(dist='poisson1', par1=knobs['m_partners_c_par1'])),
        f_partners=dict(c=dict(dist='poisson', par1=knobs['f_partners_c_par1'])),
        dur_pship=dict(
            m=dict(dist='neg_binomial', par1=knobs['dur_pship_m_par1'], par2=DUR_PSHIP_M_PAR2),
            c=dict(dist='lognormal', par1=knobs['dur_pship_c_par1'], par2=DUR_PSHIP_C_PAR2),
        ),
        analyzers=[hpv.network_history(), _ActiveTracker()],
    )
    return pars


def run_and_measure(knobs):
    n_years = BURN_IN_YEARS + max(MEASURE_YEARS, WINDOW_LONG_YEARS)
    sim = hpv.Sim(build_pars(knobs, n_years))
    sim.run()

    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    n = len(sim.people)
    t_end = sim.npts - 1
    steps_per_year = int(round(1 / DT))

    # --- Instantaneous snapshot at the end of the run: relationship-status proportions ---
    edges_now = nh.edges_at(t_end)
    active_f, active_m = act.active_female[t_end], act.active_male[t_end]
    deg_m_layer = degree_from_edges(edges_now, n, lkey=LKEY_MARITAL)
    deg_c_layer = degree_from_edges(edges_now, n, lkey=LKEY_CASUAL)
    deg_total = deg_m_layer + deg_c_layer  # 'm' and 'c' are disjoint edge sets, so this is exact

    long_pool = np.concatenate([deg_m_layer[active_f] >= 1, deg_m_layer[active_m] >= 1])
    short_pool = np.concatenate([deg_c_layer[active_f] >= 1, deg_c_layer[active_m] >= 1])
    single_pool = np.concatenate([deg_total[active_f] == 0, deg_total[active_m] == 0])
    p_long = float(long_pool.mean()) if long_pool.size else float('nan')
    p_short = float(short_pool.mean()) if short_pool.size else float('nan')
    p_single = float(single_pool.mean()) if single_pool.size else float('nan')

    # --- Windowed unions: pooled mean degree over the final 1-year and WINDOW_LONG_YEARS-year
    #     periods, excluding people who had zero partners in that window ---
    mean_degree_annual = pooled_mean_degree_excl_singles(nh, act, t_end, steps_per_year, 1, n)
    mean_degree_5yr = pooled_mean_degree_excl_singles(nh, act, t_end, steps_per_year, WINDOW_LONG_YEARS, n)

    return dict(
        mean_degree_annual=mean_degree_annual,
        mean_degree_5yr=mean_degree_5yr,
        p_single=p_single,
        p_long=p_long,
        p_short=p_short,
    )


# ---------------------------------------------------------------------------
# Calibration loop
# ---------------------------------------------------------------------------

def clipped_ratio(target, measured):
    if not measured or measured != measured:  # zero or NaN
        return 1.0
    ratio = target / measured
    return float(np.clip(ratio, 1.0 / MAX_STEP, MAX_STEP))


def print_row(label, stats):
    print(f"{label:>10} | "
          f"deg_annual={stats['mean_degree_annual']:.3f} deg_5yr={stats['mean_degree_5yr']:.3f} | "
          f"long={stats['p_long']:.3f} short={stats['p_short']:.3f} single={stats['p_single']:.3f}")


def calibrate():
    if TARGETS['mean_degree_5yr'] is None:
        raise ValueError(
            "TARGETS['mean_degree_5yr'] is not set. This script now calibrates a 5-year mean "
            "degree target (via dur_pship['c']) in addition to the annual one -- fill in the "
            "real pooled Natsal figure (mean partners in the last 5 years, excluding singles) "
            "at the top of calibrate_default_poisson.py before running."
        )

    knobs = sc.dcp(INIT_KNOBS)

    print('=' * 100)
    print('DEFAULT NETWORK (poisson) -- simple proportional calibration')
    print('=' * 100)
    print_row('target', TARGETS)

    for it in range(1, CALIBRATE_ITERS + 1):
        stats = run_and_measure(knobs)
        print_row(f'iter {it}', stats)
        print(f"{'':>10} | knobs: m_scale={knobs['m_scale']:.3f} c_scale={knobs['c_scale']:.3f} "
              f"m_partners_c_par1={knobs['m_partners_c_par1']:.3f} "
              f"f_partners_c_par1={knobs['f_partners_c_par1']:.3f} "
              f"dur_pship_c_par1={knobs['dur_pship_c_par1']:.3f} "
              f"dur_pship_m_par1={knobs['dur_pship_m_par1']:.3f}")

        knobs['m_scale'] *= clipped_ratio(TARGETS['p_long'], stats['p_long'])
        knobs['c_scale'] *= clipped_ratio(TARGETS['p_short'], stats['p_short'])
        knobs['m_partners_c_par1'] *= clipped_ratio(TARGETS['mean_degree_annual'], stats['mean_degree_annual'])
        knobs['f_partners_c_par1'] *= clipped_ratio(TARGETS['mean_degree_annual'], stats['mean_degree_annual'])
        # Inverted: longer duration (higher par1) means LOWER 5-year distinct-partner count, so
        # when measured > target we need to *increase* par1, not decrease it. Split evenly (sqrt)
        # across the two duration knobs since both contribute turnover to mean_degree_5yr.
        dur_ratio = clipped_ratio(stats['mean_degree_5yr'], TARGETS['mean_degree_5yr']) ** 0.5
        knobs['dur_pship_c_par1'] *= dur_ratio
        knobs['dur_pship_m_par1'] *= dur_ratio

    print('-' * 100)
    print('Final calibrated knobs:')
    for k, v in knobs.items():
        print(f'  {k} = {v:.4f}')
    print('Re-run run_and_measure(knobs) with more agents/years to confirm before using these '
          'in a real sim.')
    return knobs


if __name__ == '__main__':
    calibrate()

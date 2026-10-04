"""
run_r0_corrected_1900.py
========================

"R0-corrected" runs: a one-community network at MEAN DEGREE 1.4 with the per-act transmissibility
(and acts per relationship) set so that every simulated genotype has R0 between 1 and 1.5, and only
the VACCINE genotypes simulated. 100,000 agents, 1900-2050, 50 seeds, the usual NHS screening +
non-targeted NHS_Vacc programme, per-genotype export. Three networks, chosen by the NETWORK
environment variable (environment rather than argv, so spawned workers inherit it):

    NETWORK=gamma2        Gamma propensity, shape 2 (theta CV 0.71) -- basePars_community's
                          network with one community; relationship lengths 12.3 / 172.9 months,
                          standing long fraction 0.618. The default.
    NETWORK=powerlaw3p5   Pareto propensity, alpha 3.5 (theta CV 0.44, heavy tail), singleness gate
                          OFF -- basePars_community_powerlaw's network exactly as
                          run_powerlaw_nogate.py ran it: relationship lengths 12.3 / 239 months,
                          standing long fraction 0.8662, its blended age-mixing kernel.
    NETWORK=default       HPVsim's built-in 'default' network exactly as basePars.py configures it
                          (run_default_meandeg1p4.py's network): marital 'm' / casual 'c' layers,
                          the six calibrate_default_poisson knobs, basePars' age-mixing matrices and
                          cross-layer concurrency (f 0.044, m 0.50).

All three are at mean degree 1.4 EXCLUDING singles (the Natsal convention): the two community
networks fit mean_partners_per_year = 1.4 that way (run_powerlaw_nogate.py's calibrate_kwargs
exclude_singles), and basePars.py's knobs are calibrate_default_poisson's fit to the same target.
EVERYTHING ELSE is shared, so the runs differ only in the network: basePars_community's condoms
(short/casual 0.50, long/marital 0.17 -- the same numbers basePars.py has), package genotype
parameters (rel_beta hpv18 0.75, hi5 0.9), the same initial prevalence, the package acts (50 short
/ 80 long a year at peak; the default network's 'c' / 'm' defaults are the same distributions) and a
per-network BETA. (basePars_community_powerlaw's own condoms are the old swapped s=0.17 / l=0.50 and its
genotype parameters carry a calibration with hi5 rel_beta 0.064, under which hi5 cannot share an
R0 band with the other two; neither is used here. Its mixing dict is also the old swapped one, but
the power-law network takes age mixing from community_pars['age_mixing'], so that never reaches it.
Likewise basePars.py's calibrated beta 0.33 and hi5 rel_beta 0.064 are not used by NETWORK=default:
only its network is.)

GENOTYPES
---------
OHR is switched off: it is not in pars['genotypes'], so it is never seeded, never transmitted and
has no result columns. Every tracked genotype keeps its OWN initial prevalence: init_hpv_prev is
the probability of being infected at t=0 and init_hpv_dist then splits infections by genotype, so
dropping OHR from init_hpv_dist alone would hand its share to the other three. rel_init_prev is
therefore set to the tracked genotypes' share of the init_hpv_dist weights (2.3+0.9+2.2)/7.5 = 0.72,
which leaves hpv16/18/hi5 at exactly the initial prevalence they had before and OHR at 0.

Dropping OHR also removes the cross-immunity OHR infections used to give the vaccine types
(cross_imm_sus_med 0.3, cross_imm_sev_med 0.5) -- that is part of what "turning it off" means.

TRANSMISSION
------------
Target: R0 between 1 and 1.5 for every genotype. r0_corrected_acts.py works it out from the
partnership timelines recorded by --network (analytic) and from small-seed outbreaks (--r0-truth,
simulated ground truth):

  * Acts and per-act probability p enter a partnership only through the hazard a * (-ln(1-p)).
  * Analytic bounds: the distinct-person R0 (r0_hpv) and the reinfection-counting one (reff_hpv)
    bracket the truth; on the Gamma network the first cannot reach 1 at ANY number of acts (~0.8
    at saturation) and the second over-counts (~1.9) by treating back-and-forth reinfection in a
    couple as new chains.
  * Simulated R0 (realised offspring, reinfections included) is what the choice is made on.

Per network (csvs/<TAG>_acts_analysis.txt has the full working):

    gamma2        beta 0.25 -> 0.09, acts kept at 50 / 80 a year at peak: simulated R0 hpv16 1.30,
                  hpv18 1.20, hi5 1.21 (saturates at ~1.3-1.4). Acts alone: ~15.7 / 25.1 a year.
    powerlaw3p5   beta 0.25 -> 0.05, acts kept at 50 / 80: simulated R0 hpv16 1.29, hpv18 1.20,
                  hi5 1.23 (saturates at ~1.34-1.40). Acts alone: ~8.6 / 13.7 a year. The analytic
                  bounds are wider here (distinct <= 0.68, reinfection-counting ~2.2).
    default       beta 0.25 -> 0.027, acts kept at 50 / 80: simulated R0 hpv16 1.37, hpv18 1.11, hi5
                  1.18. No saturation here: realised R0 keeps rising with beta (hpv16 0.85 at 0.012,
                  1.48 at 0.035), and above ~0.05 the ground-truth outbreak saturates inside its case
                  window, so the fit uses beta <= 0.035 only (GT_FIT_MAX_BETA=0.035 for --acts).

The values are ACTS_S / ACTS_L / BETA (env-overridable; BETA defaults per network). ACTS_S is the
short layer ('s', or the default network's casual 'c'), ACTS_L the long one ('l' / marital 'm').

MODES
-----
    python run_r0_corrected_1900.py --network    network-only sim + the usual network figures,
                                                 diagnostic text, and the recorded partnership
                                                 timelines --acts needs  (~30 min)
    python run_r0_corrected_1900.py --draw       whole-run network drawing, n=1000 (network.py style)
    python run_r0_corrected_1900.py --r0-truth   simulated ground-truth R0 over a beta grid (~30 min)
    python run_r0_corrected_1900.py --acts       analytic bounds + ground truth -> acts / beta choice
    python run_r0_corrected_1900.py --selftest   pipeline + column checks, small sim
    python run_r0_corrected_1900.py              the 50 production runs

each prefixed with NETWORK=powerlaw3p5 (or NETWORK=default) for the other runs. Figures go to
figs/R0 corrected/<network>/ (plot_r0_corrected.py draws the outcome figures). The default network
has no CommunityNetworkBackend, so its --network skips the age-mixing figure (its mixing is the
input matrices) and the calibrated-backend lines, and its duration targets are dur_pship's means.

A 1900 start needs the two adjustments run_gamma2_nocomm_1900.py documents (1950 age structure,
migration off); make_pars() applies them the same way.

If stdout is redirected to a file, set PYTHONIOENCODING=utf-8 (Sim.brief() prints a glyph cp1252
cannot encode). The Pareto sampler is installed process-wide by importing powerlaw (which
basePars_community_powerlaw does), so that import happens here only when NETWORK=powerlaw3p5;
the Gamma run must never see it. --draw imports network.py (which imports powerlaw) and sets the
right sampler around its own sim.
"""
import os
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc

import basePars_community as bpc   # shared non-network pars (and the Gamma network)
import hpvsim_working as hpv
from hpvsim_working import age_community_bipartite_network_model as acbnm
import NHS_2025_lambdamu
import NHS_Vacc
import r0_hpv
import reff_hpv
from community_testing import (
    _ActiveTracker, _MixingTracker, degree_from_edges, added_edges_dict, plot_pmf,
    safe_mean, band_labels, WINDOW_STYLE, WINDOW_COLOR,
)


# -------------------------------------------------------------------
# which network
# -------------------------------------------------------------------

ROOT = pathlib.Path(__file__).parent

# community=True: CommunityNetworkBackend, layers 's' (short) / 'l' (long). The default network's
# layers are 'c' (casual, the short analogue) and 'm' (marital, the long one).
BETA_DEFAULT = 0.027  # --acts choice for NETWORK=default (the --network context was recorded at 0.25)
NETWORKS = {
    'gamma2': dict(
        label='Gamma shape 2', community=True, pareto=False, shape=2.0, fig='gamma2_meandeg1p4',
        tag='gamma2_meandeg1p4_vaxgeno_100k_1900_2050_50runs',
        beta=0.09, lkeys=('s', 'l'),
    ),
    'powerlaw3p5': dict(
        label='Power law alpha 3.5, gate off', community=True, pareto=True, shape=3.5,
        fig='powerlaw_alpha3p5_nogate',
        tag='powerlaw3p5_nogate_meandeg1p4_vaxgeno_100k_1900_2050_50runs',
        beta=0.05, lkeys=('s', 'l'),
    ),
    'default': dict(
        label='Default network', community=False, pareto=False, shape=None, fig='default_meandeg1p4',
        tag='default_meandeg1p4_vaxgeno_100k_1900_2050_50runs',
        beta=BETA_DEFAULT, lkeys=('c', 'm'),
    ),
}
NETWORK = os.environ.get('NETWORK', 'gamma2')
if NETWORK not in NETWORKS:
    raise ValueError(f'NETWORK must be one of {list(NETWORKS)}, not {NETWORK!r}')
NET = NETWORKS[NETWORK]
NET_LABEL = NET['label']

LKEY_SHORT, LKEY_LONG = NET['lkeys']
TYPE_LABEL = ({LKEY_SHORT: 'short', LKEY_LONG: 'long'} if NET['community']
              else {LKEY_SHORT: 'casual', LKEY_LONG: 'marital'})
TYPE_COLOR = {LKEY_SHORT: 'C0', LKEY_LONG: 'C3'}

if NET['pareto']:
    import basePars_community_powerlaw as bpw  # imports powerlaw -> installs the Pareto sampler
    NET_COMMUNITY_PARS = bpw.community_pars
elif NET['community']:
    NET_COMMUNITY_PARS = bpc.community_pars
else:
    import basePars as bpd  # the default network's calibrated configuration
    NET_COMMUNITY_PARS = None
# completed-duration bookkeeping keyed by each backend's own layer keys
if NET['community']:
    from community_testing import compute_duration_stats
else:
    from default_network_testing import compute_duration_stats


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

SHAPE = float(os.environ.get('SHAPE', NET['shape'])) if NET['community'] else None  # Gamma shape, or Pareto alpha
MEAN_DEGREE = float(os.environ.get('MEAN_DEGREE', 1.4))
# Natsal convention: annual distinct partners among people with at least one (run_powerlaw_nogate.py)
EXCLUDE_SINGLES = os.environ.get('EXCLUDE_SINGLES', '1').lower() not in ('0', 'false', 'no')
CALIBRATE_MAX_ITERS = int(os.environ.get('CALIBRATE_MAX_ITERS', 8))

VAX_GENOTYPES = ['hpv16', 'hpv18', 'hi5']  # the genotypes NHS_Vacc's nonavalent covers at rel_imm 1

# Transmission. neg_binomial acts per year at peak (age_act_pars then scale them by age); par2 is the
# package's dispersion and is not moved. BETA is the per-act transmission probability for hpv16
# (rel_beta 1), female->male; male->female is transm2f=3.69 times that. The per-network default is
# the --acts choice (see the module docstring); ACTS_S=15.7 ACTS_L=25.1 BETA=0.25 runs the Gamma
# network's acts-only equivalent instead.
ACTS_S = float(os.environ.get('ACTS_S', 50.0))
ACTS_L = float(os.environ.get('ACTS_L', 80.0))
BETA = float(os.environ.get('BETA', NET['beta']))
ACTS_S_PAR2, ACTS_L_PAR2 = 5.0, 40.0

N_AGENTS = int(os.environ.get('N_AGENTS', 100_000))
START = int(os.environ.get('START', 1900))
END = int(os.environ.get('END', 2050))

TAG = os.environ.get('TAG', NET['tag'])
CSV_DIR = ROOT / 'csvs'
FIG_DIR = ROOT / 'figs' / 'R0 corrected' / NET['fig']
NET_DIR = FIG_DIR / 'network'

DATA_START_YEAR = 1950
AGE_DATAFILE = ROOT / 'csvs' / 'equilibrium_1900' / 'uk_age_distribution_1950.csv'

# 5 workers ran the Gamma production at ~0.7-1.0 GB each with 2+ GB to spare (3 genotypes);
# run_gamma2_nocomm_1900.py has the 4-genotype measurements behind its more cautious 3
N_RUNS = int(os.environ.get('N_RUNS', 5))
N_CPUS = int(os.environ.get('N_CPUS', 5))
SEEDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
SEEDS = SEEDS[:int(os.environ.get('N_BATCHES', len(SEEDS)))]
RESUME = True

# --network: network-only, so no epidemic or interventions. EARLY/LATE_YEAR are the two windows of
# the distributions figure (run_gamma_sweep_network_plots.py uses 3 and 50). R0_T0 is when the R0
# recorder starts; NET_END leaves it 30 years of partnership timelines, enough for the long tail of
# women's infections.
NET_SEED = int(os.environ.get('NET_SEED', 0))
NET_AGENTS = int(os.environ.get('NET_AGENTS', N_AGENTS))
NET_END = int(os.environ.get('NET_END', 1960))
R0_T0 = int(os.environ.get('R0_T0', 1930))
EARLY_YEAR, LATE_YEAR = 3, 50
DRAW_AGENTS = int(os.environ.get('DRAW_AGENTS', 1000))  # network.py's chosen drawing size

SELFTEST_AGENTS = int(os.environ.get('SELFTEST_AGENTS', 20_000))
SELFTEST_END = int(os.environ.get('SELFTEST_END', 2030))
SELFTEST_RUNS = int(os.environ.get('SELFTEST_RUNS', 2))

N_GENOTYPE_RESULTS = 21  # '*_by_genotype' results HPVsim stores; checked by --selftest


def ctx_path():
    return CSV_DIR / f'{TAG}_r0_context.obj'


def pareto_sampler_installed():
    """ Whether the process-wide propensity sampler is powerlaw.py's Pareto one """
    return acbnm._sample_side_theta.__name__ == '_sample_side_theta_powerlaw'


def theta_cv(shape):
    """ Propensity CV: Gamma 1/sqrt(shape); Pareto(alpha) 1/sqrt(alpha (alpha-2)) (powerlaw.py) """
    return 1 / np.sqrt(shape * (shape - 2)) if NET['pareto'] else 1 / np.sqrt(shape)


# -------------------------------------------------------------------
# parameters
# -------------------------------------------------------------------

def community_pars():
    """
    The network's own community_pars (durations, frac_long, and for the power law its age-mixing
    kernel) with one community, the shape, gate off, and mean degree 1.4 excluding singles.
    """
    return sc.dcp(sc.mergedicts(NET_COMMUNITY_PARS, dict(
        gamma_shape=SHAPE,
        n_communities=1,
        community_probs=np.array([1.0]),
        community_mixing=np.array([[1.0]]),
        community_labels=['All'],
        p_single_annual=0.0,
        mean_partners_per_year=MEAN_DEGREE,
        # CommunityNetworkBackend pops this and forwards it to acbnm.calibrate()
        calibrate_kwargs=dict(exclude_singles=EXCLUDE_SINGLES, max_iters=CALIBRATE_MAX_ITERS),
    )))


def init_prev_keep_fraction():
    """ Tracked genotypes' share of basePars_community's init_hpv_dist (see module docstring) """
    w = bpc.base_pars['init_hpv_dist']
    return sum(w[g] for g in VAX_GENOTYPES) / sum(w.values())


def acts_pars(acts_s=None, acts_l=None):
    return {
        LKEY_SHORT: dict(dist='neg_binomial', par1=ACTS_S if acts_s is None else acts_s, par2=ACTS_S_PAR2),
        LKEY_LONG: dict(dist='neg_binomial', par1=ACTS_L if acts_l is None else acts_l, par2=ACTS_L_PAR2),
    }


DEFAULT_NETWORK_KEYS = ('mixing', 'layer_probs', 'm_partners', 'f_partners', 'dur_pship',
                        'f_cross_layer', 'm_cross_layer')


def default_network_pars():
    """
    basePars.py's default network (the parameters that shape it, nothing else) and
    basePars_community's condoms carried over to the 'c' / 'm' layer keys. Every layer-keyed
    parameter is given with the default network's keys only: reset_layer_pars() adds any extra key
    it finds as a new layer, so a stray 's' / 'l' would silently create two more.
    """
    out = {k: sc.dcp(bpd.base_pars_geno[k]) for k in DEFAULT_NETWORK_KEYS}
    out['network'] = 'default'
    out['condoms'] = {LKEY_SHORT: bpc.base_pars['condoms']['s'], LKEY_LONG: bpc.base_pars['condoms']['l']}
    return out


def make_pars(seed, **overrides):
    """
    basePars_community + this network (community_pars, or basePars.py's default network) + the
    changes in the module docstring. ``overrides`` win over everything and are how --network /
    --selftest shrink the run without touching the configuration.
    """
    pars = sc.dcp(bpc.base_pars_geno)
    if NET['community']:
        pars['community_pars'] = community_pars()
    else:
        pars.pop('community_pars')
        pars.update(default_network_pars())

    w = bpc.base_pars['init_hpv_dist']
    pars.update(
        genotypes=list(VAX_GENOTYPES),
        genotype_pars=sc.objdict({g: sc.dcp(bpc.base_pars['genotype_pars'][g]) for g in VAX_GENOTYPES}),
        init_hpv_dist={g: w[g] for g in VAX_GENOTYPES},
        rel_init_prev=init_prev_keep_fraction(),
        acts=acts_pars(),
        beta=BETA,
        n_agents=N_AGENTS,
        start=START,
        end=END,
        rand_seed=seed,
        interventions=NHS_2025_lambdamu.get_interventions(l=1, m=1) + sc.dcp(NHS_Vacc.vaccinations),
        analyzers=[],
        verbose=-1,
    )
    pars.update(overrides)

    if pars['start'] < DATA_START_YEAR:
        if not AGE_DATAFILE.exists():
            raise FileNotFoundError(f'{AGE_DATAFILE} is needed for a start before {DATA_START_YEAR}')
        pars.setdefault('age_datafile', str(AGE_DATAFILE))
        pars.setdefault('use_migration', False)
    return pars


def describe(pars):
    print(f'Configuration ({NETWORK}):')
    if NET['community']:
        cp = pars['community_pars']
        kind = 'Pareto alpha' if NET['pareto'] else 'Gamma shape'
        print(f"  network            community, n_communities={cp['n_communities']}, "
              f"{kind} {cp['gamma_shape']} (theta CV {theta_cv(cp['gamma_shape']):.3f})")
        print(f"  mean degree        {cp['mean_partners_per_year']} "
              f"({'EXCLUDING' if cp['calibrate_kwargs']['exclude_singles'] else 'including'} singles), "
              f"gate {'on' if cp.get('p_single_annual') else 'OFF'}")
        print(f"  durations          short {cp['D_mean_short']} mo, long {cp['D_mean_long']} mo, "
              f"frac_long {cp['frac_long']}   age mixing "
              f"{'community_pars kernel' if cp.get('age_mixing') is not None else 'from mixing[s]'}")
    else:
        dp = pars['dur_pship']
        print(f"  network            {pars['network']} (HPVsim built-in, basePars.py's knobs: "
              f"m_scale {bpd.m_scale}, c_scale {bpd.c_scale}, casual partners m poisson1 "
              f"{bpd.m_partners_c_par1} / f poisson {bpd.f_partners_c_par1})")
        print(f"  mean degree        {MEAN_DEGREE} (EXCLUDING singles; calibrate_default_poisson's target)")
        print(f"  durations          casual lognormal mean {dp['c']['par1']} yr, marital neg_binomial "
              f"mean {dp['m']['par1']} yr   cross-layer f {pars['f_cross_layer']:.3f}, "
              f"m {pars['m_cross_layer']:.3f}")
    print(f"  acts/yr at peak    {LKEY_SHORT} {pars['acts'][LKEY_SHORT]['par1']:g}, "
          f"{LKEY_LONG} {pars['acts'][LKEY_LONG]['par1']:g}   beta {pars['beta']:g}   condoms "
          f"{LKEY_SHORT}={pars['condoms'][LKEY_SHORT]}, {LKEY_LONG}={pars['condoms'][LKEY_LONG]}")
    print(f"  genotypes          {list(pars['genotypes'])}   rel_init_prev {pars['rel_init_prev']:.4f}")
    print(f"  years              {pars['start']}-{pars['end']}   n_agents={pars['n_agents']:,}   "
          f"migration {'on' if pars.get('use_migration', True) else 'OFF'}")
    print(f"  interventions      {len(pars['interventions'])}", flush=True)
    return


# -------------------------------------------------------------------
# --network: network-only sim, usual figures, R0 timelines
# -------------------------------------------------------------------

def network_sim(n_agents, end, seed, extra_analyzers=()):
    pars = make_pars(seed, n_agents=n_agents, end=end, interventions=[], rel_init_prev=0.0,
                     analyzers=[hpv.network_history(), _ActiveTracker(), *extra_analyzers],
                     verbose=0.1)
    return hpv.Sim(pars, label=f'{TAG} network-only')


def year_of(sim, t):
    return int(sim.yearvec[t] - sim['start']) + 1


def window_of(year):
    return {EARLY_YEAR: 'early', LATE_YEAR: 'late'}.get(year)


def network_series(sim):
    """
    One pass over the network history: per-step instantaneous / quarterly-union degree, standing
    long fraction, per-year annual degree (all and excluding singles), and the early/late samples
    for the distributions figure. Same definitions as run_gamma_sweep_network_plots.run_for_shape().
    """
    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    lm = nh.layer_map
    n = len(sim.people)

    inst = {'early': [], 'late': []}
    inst_type = {lk: {'early': [], 'late': []} for lk in (LKEY_SHORT, LKEY_LONG)}
    quart = {'early': [], 'late': []}
    yearly = {'early': None, 'late': None}
    yearly_type = {lk: {'early': None, 'late': None} for lk in (LKEY_SHORT, LKEY_LONG)}
    ts = dict(inst_mean=[], quart_mean=[], long_frac=[])
    ann = dict(year=[], mean_all=[], mean_excl=[], frac_zero=[], cv_excl=[], e2e_excl=[])

    running = None
    base, added, obs_f, obs_m = {}, {}, set(), set()

    def finalize(year):
        union = {**base, **added}
        deg = degree_from_edges(union, n)
        counts = np.concatenate([deg[sorted(obs_f)], deg[sorted(obs_m)]])
        part = counts[counts >= 1]
        ann['year'].append(sim['start'] + year - 1)
        ann['mean_all'].append(counts.mean() if counts.size else np.nan)
        ann['mean_excl'].append(part.mean() if part.size else np.nan)
        ann['frac_zero'].append((counts == 0).mean() if counts.size else np.nan)
        ann['cv_excl'].append(part.std() / part.mean() if part.size else np.nan)
        ann['e2e_excl'].append((part ** 2).mean() / part.mean() if part.size else np.nan)
        w = window_of(year)
        if w:
            yearly[w] = counts
            for lk in (LKEY_SHORT, LKEY_LONG):
                d = degree_from_edges(union, n, lkey=lk)
                yearly_type[lk][w] = np.concatenate([d[sorted(obs_f)], d[sorted(obs_m)]])

    for t in range(sim.npts):
        year = year_of(sim, t)
        w = window_of(year)
        if year != running:
            if running is not None:
                finalize(running)
            running = year
            base = nh.edges_at(t - 1) if t > 0 else {}
            added, obs_f, obs_m = {}, set(), set()

        now = nh.edges_at(t)
        af, am = act.active_female[t], act.active_male[t]
        deg = degree_from_edges(now, n)
        pooled = np.concatenate([deg[af], deg[am]])
        ts['inst_mean'].append(pooled.mean() if pooled.size else 0.0)
        if w:
            inst[w].append(pooled)
            for lk in (LKEY_SHORT, LKEY_LONG):
                d = degree_from_edges(now, n, lkey=lk)
                inst_type[lk][w].append(np.concatenate([d[af], d[am]]))
        n_now = len(now)
        ts['long_frac'].append(sum(1 for *_, lk in now.values() if lk == LKEY_LONG) / n_now if n_now else 0.0)

        add_t = added_edges_dict(nh.deltas.get(t), lm)
        qu = {**(nh.edges_at(t - 1) if t > 0 else {}), **add_t}
        dq = degree_from_edges(qu, n)
        pq = np.concatenate([dq[af], dq[am]])
        ts['quart_mean'].append(pq.mean() if pq.size else 0.0)
        if w:
            quart[w].append(pq)
        added.update(add_t)
        obs_f.update(af.tolist())
        obs_m.update(am.tolist())
    finalize(running)

    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return dict(inst=cat(inst), inst_type={lk: cat(v) for lk, v in inst_type.items()},
                quart=cat(quart), yearly=yearly, yearly_type=yearly_type, ts=ts,
                annual=pd.DataFrame(ann), durations=compute_duration_stats(nh))


def net_targets(sim):
    """
    The reference values the network figures and diagnostic draw against: the community backend's
    calibrated standing long fraction / formation probability and its duration means; the default
    network has neither scalar (default_network_testing.py point 2), so dur_pship's means (par1 is
    the mean of both its distributions), in months, are its only targets.
    """
    if NET['community']:
        cp, p = sim['community_pars'], sim.network_backend._params
        return dict(frac_long=p['frac_long_target'], p_form_long=p['p_form_long'],
                    dur={LKEY_SHORT: cp['D_mean_short'], LKEY_LONG: cp['D_mean_long']},
                    title=f"shape={cp['gamma_shape']}, 1 community, mean degree {MEAN_DEGREE} excl. singles, "
                          f"durations {cp['D_mean_short']:.1f}/{cp['D_mean_long']:.1f} months, "
                          f"standing long fraction {p['frac_long_target']}")
    dur = {lk: sim['dur_pship'][lk]['par1'] * 12 for lk in (LKEY_SHORT, LKEY_LONG)}
    return dict(frac_long=None, p_form_long=None, dur=dur,
                title=f"basePars.py knobs, mean degree {MEAN_DEGREE} excl. singles, casual/marital "
                      f"mean durations {dur[LKEY_SHORT]:.0f}/{dur[LKEY_LONG]:.0f} months, cross-layer "
                      f"f {sim['f_cross_layer']:.3f} / m {sim['m_cross_layer']:.3f}")


def plot_distributions(sim, s, out_png):
    """ The 8-panel figure of run_gamma_sweep_network_plots.py / community_testing.py """
    tg = net_targets(sim)
    lab = {'early': f'year {EARLY_YEAR}', 'late': f'year {LATE_YEAR}'}
    short_d = np.asarray(s['durations'][LKEY_SHORT], float)
    long_d = np.asarray(s['durations'][LKEY_LONG], float)

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    plot_pmf(axes[0, 0], [s['inst']['early'], s['inst']['late']], [lab['early'], lab['late']],
             [WINDOW_COLOR['early'], WINDOW_COLOR['late']], ['-', '-'], ['o', 'o'],
             'current partners', '1. Instantaneous degree\n(HPVsim quarterly steps pooled within year)')
    plot_pmf(axes[0, 1], [s['quart']['early'], s['quart']['late']], [lab['early'], lab['late']],
             [WINDOW_COLOR['early'], WINDOW_COLOR['late']], ['-', '-'], ['s', 's'],
             'partners in the quarter', '2. Quarterly degree\n(union over one HPVsim step)')

    ax = axes[0, 2]
    mx = int(max(s['yearly']['early'].max(), s['yearly']['late'].max()))
    bins = np.arange(mx + 2) - 0.5
    for w in ('early', 'late'):
        ax.hist(s['yearly'][w], bins=bins, density=True, histtype='step', linewidth=2,
                color=WINDOW_COLOR[w], label=lab[w])
    ax.axvline(MEAN_DEGREE, linestyle='--', color='gray', label=f'target mean {MEAN_DEGREE} (excl. singles)')
    ax.set_title('3. Annual degree\n(distinct partners in 12 months)')
    ax.set_xlabel('distinct partners'); ax.set_ylabel('fraction of people'); ax.set_yscale('log')
    ax.legend(fontsize=8)

    samples, labels, colors, styles, markers = [], [], [], [], []
    for lk in (LKEY_SHORT, LKEY_LONG):
        for w in ('early', 'late'):
            samples.append(s['inst_type'][lk][w]); labels.append(f'{TYPE_LABEL[lk]}, {lab[w]}')
            colors.append(TYPE_COLOR[lk]); styles.append(WINDOW_STYLE[w])
            markers.append('o' if lk == LKEY_SHORT else 's')
    plot_pmf(axes[0, 3], samples, labels, colors, styles, markers,
             'current partners of the selected type', '4. Instantaneous degree by type')

    ax = axes[1, 0]
    mx = int(max(short_d.max() if short_d.size else 0, long_d.max() if long_d.size else 0))
    bins = np.arange(mx + 2) - 0.5
    if short_d.size:
        ax.hist(short_d, bins=bins, density=True, histtype='step', linewidth=2,
                color=TYPE_COLOR[LKEY_SHORT], label=f'{TYPE_LABEL[LKEY_SHORT]} (mean {short_d.mean():.1f} mo)')
    if long_d.size:
        ax.hist(long_d, bins=bins, density=True, histtype='step', linewidth=2,
                color=TYPE_COLOR[LKEY_LONG], label=f'{TYPE_LABEL[LKEY_LONG]} (mean {long_d.mean():.1f} mo)')
    ax.axvline(tg['dur'][LKEY_SHORT], linestyle='--', color=TYPE_COLOR[LKEY_SHORT], alpha=0.6)
    ax.axvline(tg['dur'][LKEY_LONG], linestyle='--', color=TYPE_COLOR[LKEY_LONG], alpha=0.6)
    ax.set_title('5. Completed partnership durations\n(HPVsim-step resolution)')
    ax.set_xlabel('duration in months'); ax.set_ylabel('fraction of completed partnerships')
    ax.set_yscale('log'); ax.legend(fontsize=8)

    ax = axes[1, 1]
    a = s['annual']
    ax.axhline(MEAN_DEGREE, linestyle='--', color='gray', label='target (excl. singles)')
    ax.plot(a['year'], a['mean_excl'], 'o-', ms=3, color='C2', label='realised, excluding singles')
    ax.plot(a['year'], a['mean_all'], 's-', ms=3, color='C7', label='realised, including singles')
    ax.set_ylim(0, max(np.nanmax(a['mean_excl']), MEAN_DEGREE) * 1.3)
    ax.set_title('6. Mean annual degree'); ax.set_xlabel('year'); ax.set_ylabel('mean distinct partners')
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(np.asarray(sim.yearvec), s['ts']['long_frac'], color=TYPE_COLOR[LKEY_LONG], lw=1.2,
            label='standing fraction')
    if tg['frac_long'] is not None:
        ax.axhline(tg['frac_long'], linestyle='--', color='gray', label='target')
        ax.axhline(tg['p_form_long'], linestyle=':', color=TYPE_COLOR[LKEY_SHORT],
                   label=f"formation probability = {tg['p_form_long']:.2f}")
    ax.set_ylim(0, 1); ax.set_title(f'7. Standing {TYPE_LABEL[LKEY_LONG]}-partnership fraction')
    ax.set_xlabel('year'); ax.set_ylabel('fraction of active edges'); ax.legend(loc='center right', fontsize=8)

    samples, labels, colors, styles, markers = [], [], [], [], []
    for lk in (LKEY_SHORT, LKEY_LONG):
        for w in ('early', 'late'):
            samples.append(s['yearly_type'][lk][w]); labels.append(f'{TYPE_LABEL[lk]}, {lab[w]}')
            colors.append(TYPE_COLOR[lk]); styles.append(WINDOW_STYLE[w])
            markers.append('o' if lk == LKEY_SHORT else 's')
    plot_pmf(axes[1, 3], samples, labels, colors, styles, markers,
             'distinct partners of the selected type', '8. Annual degree by type\n(12-month union)')

    kind = 'community network' if NET['community'] else '(HPVsim built-in)'
    fig.suptitle(f"{NET_LABEL} {kind}, network-only -- n_agents={sim['n_agents']:,}, "
                 f"{sim['start']}-{sim['end']}, {tg['title']}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'saved {out_png}')


def plot_mixing(sim, s, out_png):
    """
    community_testing.py's mixing figure, minus the community panels (one community here), plus
    annual degree and singleness by age band from the final year.
    """
    be = sim.network_backend
    mix = sim.get_analyzer('mixing_tracker')
    blab = band_labels(be._params)
    nb = len(blab)
    A = np.asarray(be._params['A_age'], float)
    M = mix.Mage.astype(float)
    Mc = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    Ac = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
    ok = (M.sum(axis=1) > 0) & (A.sum(axis=1) > 0)
    corr = float(np.corrcoef(Mc[ok].ravel(), Ac[ok].ravel())[0, 1]) if ok.any() else np.nan

    # degree by age band over the last year of the run
    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    t1 = sim.npts - 1
    spy = int(round(1 / sim['dt']))
    union = dict(nh.edges_at(t1 - spy))
    for t in range(t1 - spy + 1, t1 + 1):
        union.update(added_edges_dict(nh.deltas.get(t), nh.layer_map))
    deg = degree_from_edges(union, len(sim.people))
    active = np.unique(np.concatenate([np.concatenate([act.active_female[t], act.active_male[t]])
                                       for t in range(t1 - spy + 1, t1 + 1)]))
    edges_b = np.asarray(be._params['age_band_edges'], float)
    band = np.searchsorted(edges_b, sim.people.age[active], side='right')
    d = deg[active]
    mean_excl = np.array([d[(band == b) & (d >= 1)].mean() if ((band == b) & (d >= 1)).any() else np.nan
                          for b in range(nb)])
    frac_zero = np.array([(d[band == b] == 0).mean() if (band == b).any() else np.nan for b in range(nb)])

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    src = ("community_pars['age_mixing']" if sim['community_pars'].get('age_mixing') is not None
           else "sim['mixing']['s']")
    for ax, mat, title in ((axes[0, 0], A, f"1. Input age-mixing kernel A\n(sourced from {src})"),
                           (axes[0, 1], Mc, f'2. Realised age mixing (row-normalised), final snapshot\n'
                                            f'P(V band | U band); corr with row-normalised A = {corr:.2f}')):
        im = ax.imshow(mat, origin='lower', cmap='viridis', aspect='auto', vmin=0)
        ax.set_title(title); ax.set_xlabel('V-side (male) age band'); ax.set_ylabel('U-side (female) age band')
        ax.set_xticks(range(nb)); ax.set_xticklabels(blab, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(nb)); ax.set_yticklabels(blab, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    ax.bar(range(nb), mean_excl, color='C2')
    ax.axhline(MEAN_DEGREE, ls='--', color='gray', label=f'overall target {MEAN_DEGREE}')
    ax.set_xticks(range(nb)); ax.set_xticklabels(blab, rotation=45, ha='right', fontsize=7)
    ax.set_title('3. Annual distinct partners by age band\n(final year, excluding singles)')
    ax.set_ylabel('mean distinct partners'); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(range(nb), frac_zero, color='C7')
    ax.set_ylim(0, 1)
    ax.set_xticks(range(nb)); ax.set_xticklabels(blab, rotation=45, ha='right', fontsize=7)
    ax.set_title('4. Fraction of sexually active people with no partner in the final year')
    ax.set_ylabel('fraction')

    fig.suptitle(f"{NET_LABEL} community network -- age mixing and degree by age\n"
                 f"n_agents={sim['n_agents']:,}, {nb} age bands, final snapshot of {int(M.sum()):,} "
                 f"standing edges", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'saved {out_png}')


def network_analysis():
    """
    Network-only sim of the production configuration (100k agents, 1900 start, migration off),
    the usual figures, a diagnostic text, and the R0 recorder / parameter context --acts uses.
    """
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    NET_DIR.mkdir(parents=True, exist_ok=True)
    rec = r0_hpv.R0Recorder(t0_year=R0_T0)
    # StateSnapshots supplies ages (women's dur_cin doubles from age_risk['age']) and the end state
    # reff_hpv's estimator reads; with no infection or vaccination its immunity fields are all zero
    snaps = reff_hpv.StateSnapshots(years=[R0_T0])
    # _MixingTracker reads CommunityNetworkBackend's internal state, which the default network lacks
    mixing = [_MixingTracker()] if NET['community'] else []
    sim = network_sim(NET_AGENTS, NET_END, NET_SEED, extra_analyzers=[*mixing, rec, snaps])
    describe(sim.pars)
    print(f'\nNetwork-only run: seed {NET_SEED}, {NET_AGENTS:,} agents, {START}-{NET_END}; '
          f'R0 recorder from {R0_T0}', flush=True)
    sim.run()

    s = network_series(sim)
    plot_distributions(sim, s, NET_DIR / f'{TAG}_network_distributions.png')
    if NET['community']:
        plot_mixing(sim, s, NET_DIR / f'{TAG}_network_mixing.png')
    s['annual'].to_csv(CSV_DIR / f'{TAG}_network_annual_degree.csv', index=False)

    # R0 context: recorder + parameters, detached from the Sim (reff_hpv.make_context)
    rec = sim.get_analyzer('R0Recorder')
    ctx = reff_hpv.make_context(sim, rec, sim.get_analyzer('StateSnapshots'))
    ctx['pars']['rel_beta'] = {g: sim['genotype_pars'][g]['rel_beta'] for g in VAX_GENOTYPES}
    ctx['net'] = dict(annual=s['annual'], long_frac=np.asarray(s['ts']['long_frac']),
                      inst_mean=np.asarray(s['ts']['inst_mean']), yearvec=np.asarray(sim.yearvec),
                      dur_short=np.asarray(s['durations'][LKEY_SHORT]),
                      dur_long=np.asarray(s['durations'][LKEY_LONG]),
                      calibrated=dict(getattr(sim.network_backend, '_params', None) or {}))
    sc.save(ctx_path(), ctx)
    print(f'saved {ctx_path()}')

    a = s['annual']
    last = a.iloc[-5:]
    dur_s, dur_l = s['durations'][LKEY_SHORT], s['durations'][LKEY_LONG]
    tg = net_targets(sim)
    if NET['community']:
        p = sim.network_backend._params
        head = [f'  {NET_LABEL} (shape {SHAPE}), 1 community, mean_partners_per_year={MEAN_DEGREE} '
                f'({"EXCLUDING" if EXCLUDE_SINGLES else "including"} singles), gate off',
                f"  calibrated: rho={p['rho']:.4e}  k_snap={p.get('k_snap', float('nan')):.4f}  "
                f"p_form_long={p['p_form_long']:.3f}"]
    else:
        head = [f'  {NET_LABEL}: {tg["title"]}',
                f'  layers {LKEY_SHORT} ({TYPE_LABEL[LKEY_SHORT]}) / {LKEY_LONG} ({TYPE_LABEL[LKEY_LONG]}); '
                f'degree counts edges, so a casual pair holding two parallel partnerships counts twice '
                f'(as calibrate_default_poisson measured it)']
    lines = [
        f'{TAG}  --  network diagnostic (network-only, seed {NET_SEED}, {NET_AGENTS:,} agents, '
        f'{START}-{NET_END}, migration {"on" if sim["use_migration"] else "off"})',
        '',
        *head,
        '',
        '  Annual distinct partners, mean of the last 5 years (and the whole-run range)',
        f"    excluding singles        {last['mean_excl'].mean():.3f}   target {MEAN_DEGREE}"
        f"   (range {a['mean_excl'].min():.3f}-{a['mean_excl'].max():.3f})",
        f"    including singles        {last['mean_all'].mean():.3f}",
        f"    fraction with none       {last['frac_zero'].mean():.3f}",
        f"    CV among partnered       {last['cv_excl'].mean():.3f}   (Natsal 1.831)",
        f"    E[k^2]/E[k] partnered    {last['e2e_excl'].mean():.3f}   (Natsal het1yr ~5-7.7)",
        '',
        f"  Instantaneous mean degree (active people), last 5 years  {np.mean(s['ts']['inst_mean'][-20:]):.3f}",
        f"  Standing {TYPE_LABEL[LKEY_LONG]} fraction, last 5 years".ljust(60)
        + f"{np.mean(s['ts']['long_frac'][-20:]):.3f}"
        + (f"   target {tg['frac_long']}" if tg['frac_long'] is not None else ''),
        f"  Completed durations: {TYPE_LABEL[LKEY_SHORT]} {safe_mean(dur_s):.1f} mo (target {tg['dur'][LKEY_SHORT]:.1f}), "
        f"{TYPE_LABEL[LKEY_LONG]} {safe_mean(dur_l):.1f} mo (target {tg['dur'][LKEY_LONG]:.1f}, right-censored run)",
    ]
    text = '\n'.join(lines) + '\n'
    out = CSV_DIR / f'{TAG}_network_diagnostic.txt'
    out.write_text(text, encoding='utf-8')
    print('\n' + text)
    print(f'wrote {out}')
    return


# -------------------------------------------------------------------
# --draw: whole-run network picture (network.py style)
# -------------------------------------------------------------------

def draw():
    """
    network.py's cumulative whole-run partnership drawing for THIS configuration, at n=1000 (the
    size network.py settled on). network.py imports powerlaw, which installs the Pareto sampler;
    its _theta_sampler context sets this run's sampler (Gamma or Pareto) for our sim.
    """
    import network as netdraw  # noqa: E402 -- deliberately late, see docstring
    NET_DIR.mkdir(parents=True, exist_ok=True)
    with netdraw._theta_sampler(pareto=NET['pareto']):
        assert pareto_sampler_installed() == NET['pareto']
        sim = network_sim(DRAW_AGENTS, NET_END, NET_SEED)
        sim['verbose'] = 0
        sim.run()
    nh = sim.get_analyzer('network_history')
    edges = netdraw.collect_edges(nh, 0, sim.npts - 1)
    ppl = sim.people
    uids = np.unique(np.concatenate([edges['f'].values, edges['m'].values]))
    nodes = pd.DataFrame(dict(uid=uids, sex=np.where(ppl.sex[uids] == 0, 'f', 'm'),
                              community=ppl.community[uids].astype(int)))
    edges['start_year'] = netdraw.step_year(sim, edges['t_start'])
    edges['end_year'] = netdraw.step_year(sim, edges['t_end'])
    G = netdraw.build_graph(edges, nodes)
    stats = netdraw.graph_stats(G)
    pos = netdraw.packed_layout(G, seed=NET_SEED)
    fig, ax = plt.subplots(figsize=(10, 10.6), facecolor=netdraw.SURFACE)
    netdraw.draw_network(ax, G, pos, f'{NET_LABEL}, mean degree {MEAN_DEGREE} (excl. singles) '
                                     f'-- n_agents = {DRAW_AGENTS:,}, seed {NET_SEED}, {START}-{NET_END}', stats)
    ax.legend(handles=netdraw.legend_handles(), loc='upper right', frameon=False, fontsize=9)
    fig.tight_layout()
    out = NET_DIR / f'{TAG}_network_graph_n{DRAW_AGENTS}.png'
    fig.savefig(out, dpi=130, facecolor=netdraw.SURFACE)
    plt.close(fig)
    print(f'saved {out}')
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()})
    return


# -------------------------------------------------------------------
# export and production runs (as run_gamma2_nocomm_1900.py)
# -------------------------------------------------------------------

def export_df(run_sim):
    """ to_df() widened with every by-genotype result (one column per genotype), infections_star, prevalence_star """
    df = run_sim.to_df(date_index=True)
    res = run_sim.results
    genotypes = list(run_sim['genotype_map'].values())
    newcols = {}
    suffix = '_by_genotype'
    for key in run_sim.result_keys('genotype'):
        stem = key[:-len(suffix)] if key.endswith(suffix) else key
        vals = np.asarray(res[key][:], dtype=float)
        for gi, g in enumerate(genotypes):
            newcols[f'{stem}_{g}'] = vals[gi, :]
    n_infected = np.asarray(res['n_infected'][:], dtype=float)
    newcols['infections_star'] = n_infected
    newcols['prevalence_star'] = sc.safedivide(n_infected, np.asarray(res['n_alive'][:], dtype=float))
    clashes = sorted(set(newcols) & set(df.columns))
    if clashes:
        raise ValueError(f'new columns collide with existing to_df() columns: {sc.strjoin(clashes)}')
    return df.join(pd.DataFrame(newcols, index=df.index))


def run_batches(csv_path, seed_list, n_runs, n_cpus, resume=True, label=None, **par_overrides):
    """ MultiSim in batches of n_runs, appending each finished run to one CSV. """
    done_seeds = set()
    header_written = False
    if csv_path.exists():
        if not resume:
            raise FileExistsError(f'{csv_path} already exists -- move or delete it first.')
        prev = pd.read_csv(csv_path, index_col=0)
        done_seeds = {int(x) for x in prev['Seed'].unique()}
        header_written = True
        print(f'RESUME: {len(done_seeds)} run(s) already present; those seeds are skipped.', flush=True)

    for seed in seed_list:
        batch = [seed + i for i in range(n_runs)]
        if all(s in done_seeds for s in batch):
            print(f'Skipping seeds {batch[0]}-{batch[-1]} (already in the CSV).', flush=True)
            continue
        sim = hpv.Sim(make_pars(seed, **par_overrides), label=label or TAG)
        print(f'Running MultiSim with n_runs = {n_runs}  (seeds {seed}-{seed + n_runs - 1}) ...', flush=True)
        msim = hpv.MultiSim(sim)
        msim.run(n_runs=n_runs, n_cpus=n_cpus)
        print('MultiSim run complete.', flush=True)
        for i, run_sim in enumerate(msim.sims):
            this_seed = seed + i
            if this_seed in done_seeds:
                continue
            try:
                temp_df = export_df(run_sim)
            except Exception as e:
                print(f'Could not save run results to df: {e}', flush=True)
                continue
            temp_df['Seed'] = this_seed
            temp_df.to_csv(csv_path, mode='a', index=True, header=not header_written)
            header_written = True
            prev16 = float(run_sim.results['hpv_prevalence_by_genotype'][0, -1])
            print(f'Seed:{this_seed} is done  --  {len(temp_df.columns)} columns, '
                  f'final hpv16 prevalence {prev16:.4f}', flush=True)
        del msim
    return


def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f'{TAG}.csv'
    describe(make_pars(SEEDS[0]))
    print(f'Seed batches: {SEEDS}  -> up to {len(SEEDS) * N_RUNS} runs')
    print(f'Outputs will be saved to: {csv_path}', flush=True)
    run_batches(csv_path, SEEDS, N_RUNS, N_CPUS, resume=RESUME)
    print(f'\nDone. Wrote {csv_path}')
    return


# -------------------------------------------------------------------
# --selftest
# -------------------------------------------------------------------

def selftest():
    """ The real code path (make_pars -> MultiSim -> export_df -> CSV) small, then invariants on the CSV """
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f'{TAG}_selftest.csv'
    if csv_path.exists():
        csv_path.unlink()

    # what actually reaches an initialised Sim (layer pars go through reset_layer_pars)
    probe = hpv.Sim(make_pars(0, n_agents=2000, end=START + 2, interventions=[]))
    probe.initialize()
    pars = make_pars(0, n_agents=SELFTEST_AGENTS, end=SELFTEST_END)
    describe(pars)
    print(f'\nSelf-test: {SELFTEST_RUNS} runs, {SELFTEST_AGENTS:,} agents, {START}-{SELFTEST_END}', flush=True)
    run_batches(csv_path, [0], SELFTEST_RUNS, min(SELFTEST_RUNS, N_CPUS), resume=False,
                label=f'{TAG} self-test', n_agents=SELFTEST_AGENTS, end=SELFTEST_END)

    df = pd.read_csv(csv_path, index_col=0)
    gt = list(VAX_GENOTYPES)
    gen_cols = [c for c in df.columns if any(c.endswith(f'_{g}') for g in gt)]
    w = bpc.base_pars['init_hpv_dist']
    tol = 1e-9
    n_alive = df['n_alive'].to_numpy()
    if NET['community']:
        cp = probe['community_pars']
        net_checks = [
            (f"community network, {'Pareto' if NET['pareto'] else 'Gamma'} propensity sampler installed",
             probe['network'] == 'community' and pareto_sampler_installed() == NET['pareto']),
            (f'shape {SHAPE}, one community',
             float(cp['gamma_shape']) == SHAPE and int(cp['n_communities']) == 1),
            (f'mean degree {MEAN_DEGREE} excluding singles, gate off',
             float(cp['mean_partners_per_year']) == MEAN_DEGREE
             and cp['calibrate_kwargs']['exclude_singles'] is EXCLUDE_SINGLES and not cp.get('p_single_annual')),
            (f"the network's usual relationship lengths ({cp['D_mean_short']} / {cp['D_mean_long']} mo, "
             f"long fraction {cp['frac_long']})",
             all(cp[k] == NET_COMMUNITY_PARS[k] for k in ('D_mean_short', 'D_mean_long', 'frac_long'))),
        ]
    else:
        bp = bpd.base_pars_geno
        same = lambda a, b: bool(np.array_equal(np.asarray(a), np.asarray(b)))
        net_checks = [
            ('default network, Gamma propensity sampler untouched',
             probe['network'] == 'default' and not pareto_sampler_installed()),
            (f'exactly the default layers {LKEY_SHORT} / {LKEY_LONG} (no stray community s / l)',
             all(set(probe[k]) == {LKEY_SHORT, LKEY_LONG}
                 for k in ('acts', 'condoms', 'mixing', 'layer_probs', 'dur_pship', 'age_act_pars'))),
            ("basePars.py's network: mixing, layer_probs, durations, casual partners, cross-layer",
             all(same(probe['mixing'][lk], bp['mixing'][lk]) and same(probe['layer_probs'][lk], bp['layer_probs'][lk])
                 and probe['dur_pship'][lk] == bp['dur_pship'][lk] for lk in (LKEY_SHORT, LKEY_LONG))
             and probe['m_partners'][LKEY_SHORT] == bp['m_partners'][LKEY_SHORT]
             and probe['f_partners'][LKEY_SHORT] == bp['f_partners'][LKEY_SHORT]
             and probe['f_cross_layer'] == bp['f_cross_layer'] and probe['m_cross_layer'] == bp['m_cross_layer']),
        ]
    bc = bpc.base_pars['condoms']
    checks = net_checks + [
        ("non-network pars are basePars_community's (condoms short/long, genotype pars)",
         probe['condoms'][LKEY_SHORT] == bc['s'] and probe['condoms'][LKEY_LONG] == bc['l']
         and all(probe['genotype_pars'][g]['rel_beta'] == bpc.base_pars['genotype_pars'][g]['rel_beta']
                 for g in VAX_GENOTYPES)),
        (f'genotypes are {gt} only (OHR off)',
         list(probe['genotype_map'].values()) == gt and not any('ohr' in c for c in df.columns)),
        ('tracked genotypes keep their initial prevalence (rel_init_prev = their share)',
         abs(probe['rel_init_prev'] - sum(w[g] for g in gt) / sum(w.values())) < tol
         and set(probe['init_hpv_dist']) == set(gt)),
        (f"acts reached the sim: {LKEY_SHORT} {ACTS_S:g}, {LKEY_LONG} {ACTS_L:g}",
         probe['acts'][LKEY_SHORT]['par1'] == ACTS_S and probe['acts'][LKEY_LONG]['par1'] == ACTS_L),
        (f'beta reached the sim: {BETA:g}', probe['beta'] == BETA),
        (f'{N_GENOTYPE_RESULTS * len(gt)} by-genotype columns', len(gen_cols) == N_GENOTYPE_RESULTS * len(gt)),
        (f'{SELFTEST_RUNS} seeds present', df['Seed'].nunique() == SELFTEST_RUNS),
        ('hpv_prevalence == n_infectious / n_alive',
         bool(np.allclose(df['hpv_prevalence'], df['n_infectious'] / n_alive, atol=tol))),
        ('infections == sum over genotypes',
         bool(np.allclose(df['infections'], sum(df[f'infections_{g}'] for g in gt), rtol=1e-6))),
        ('screening live after 1980', bool(float(df.loc[df.index >= 1981, 'cum_screens'].max()) > 0)),
        ('vaccination live after 2008', bool(float(df.loc[df.index >= 2009, 'cum_doses'].max()) > 0)),
    ]
    print()
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    n_fail = sum(1 for _, ok in checks if not ok)
    print(f'\n{len(checks) - n_fail}/{len(checks)} checks passed.')
    for g in gt:
        s = df.groupby(df.index)[f'hpv_prevalence_{g}'].mean()
        print(f'  {g} prevalence (mean of runs): ' + ', '.join(f'{int(y)} {s.loc[y]:.4f}'
              for y in s.index if int(y) in (START, START + 1, 1920, 1950, 1980, 2008, SELFTEST_END)))
    print(f'Wrote {csv_path}  ({len(df.columns)} columns, {len(df)} rows).')
    if n_fail:
        sys.exit(1)
    return


if __name__ == '__main__':
    args = set(sys.argv[1:])
    if '--network' in args:
        network_analysis()
    elif '--draw' in args:
        draw()
    elif '--acts' in args:
        import r0_corrected_acts
        r0_corrected_acts.main()
    elif '--r0-truth' in args:
        import r0_corrected_acts
        r0_corrected_acts.ground_truth(ncpus=int(os.environ.get('GT_CPUS', 6)))
    elif '--selftest' in args:
        selftest()
    else:
        main()

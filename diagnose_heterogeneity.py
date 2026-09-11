"""
diagnose_heterogeneity.py
=========================

Why the power-law community network cannot reproduce Natsal's partner-count heterogeneity, and
which knob (if any) can fix it without breaking the other calibration targets.

The problem
-----------
calibrate_community_powerlaw.py targets cv_degree_annual = 1.831 -- the pooled coefficient of
variation of het1yr among partnered Natsal-3 respondents. It achieved 0.988, with the Pareto tail
index alpha pinned at GAMMA_SHAPE_FLOOR (2.05), and concluded "the Pareto tail cannot get heavier
without crossing the finite-variance bound."

That conclusion is right but the diagnosis is incomplete. At alpha = 2.05 the PROPENSITY theta has
CV = 1/sqrt(alpha*(alpha-2)) = 3.12 -- already 1.7x ABOVE the 1.831 target -- and the realised
annual degree still only reaches CV 0.988. The heterogeneity is being destroyed somewhere between
theta and the realised partner count, so making theta heavier cannot fix it. Confirmed by direct
measurement (20k agents, attribution analyzer): driving alpha from 3 to its 2.05 floor moves the
acquisition Gini only 0.100 -> 0.142, against a literature value of 0.33-0.38 (Gsteiger et al.
2020, cited in hpvsim_working/analysis.py).

FINDINGS (20k agents, single seed -- see the caveat at the end)
--------------------------------------------------------------
Two hypotheses were tested and BOTH were wrong. Recorded here so they are not re-tried.

1. "Partnership duration caps the annual partner count." REFUTED by stage 1. Shortening
   D_mean_short 8x (12.3 -> 1.5 months) moves cv_degree_annual only 1.296 -> 1.433 -> 1.316, i.e.
   ~10% and non-monotonically, while p_single blows out from 0.288 to 0.517. Duration is not the
   binding constraint, and spending it would break the Natsal p_single target to buy almost nothing.

2. "The network saturates -- high-theta nodes cannot realise their propensity." REFUTED by
   saturation(). The realised degree CV is 1.564 against a Poisson-mixture prediction of 1.574,
   i.e. 99% of what proportionality predicts. Mean degree by theta decile tracks the proportional
   prediction within 86-121% across the whole range. The network transmits propensity heterogeneity
   into partner-count heterogeneity essentially faithfully. Realised max annual degree is 250, not
   the ~20 suggested by network_stats_community_powerlaw.npz (whose histogram simply has its
   overflow bin at 20).

What is actually going on: the realised theta CV is 1.395, NOT the analytic 3.12. At alpha = 2.05
the analytic CV is carried by draws so rare they never appear in a population of this size -- the
sample CV of a near-infinite-variance Pareto converges extremely slowly. So lowering alpha buys far
less heterogeneity than 1/sqrt(alpha*(alpha-2)) advertises, which is exactly what the direct
attribution sweep showed (Gini 0.100 -> 0.142 for alpha 3 -> 2.05).

And the concentration the network does produce already matches the data: the top 10% by realised
annual degree hold 35.3% of all partnerships, against 35.0% (female) / 39.5% (male) computed from
raw Natsal-3. So the acquisition-Gini shortfall (0.10-0.14 modelled vs 0.33-0.38 in Gsteiger et al.
2020) is probably NOT a network-heterogeneity failure -- it more likely lives in the infection
dynamics, where near-universal HPV acquisition compresses infection counts relative to the partner
counts that drive them.

CAVEAT: all of the above is one seed at 20k agents, and this size does not reproduce the 200k
calibration (mean_degree_annual 2.4 here vs 1.447 at 200k; cv_degree_annual 1.296 vs 0.988). The
qualitative conclusions should hold, but confirm the top-decile share at full size before relying
on it.

What is fixed and what is swept
-------------------------------
p_single_annual is held at 0.20 in every cell. The 20%-single / 80%-partnered split is itself a
Natsal figure (p_single = (0.223 + 0.179)/2 = 0.201, calibrate_default_poisson.py), so it is a
target to satisfy, not a knob to spend. Buying CV by making more people single would be fitting one
Natsal number by breaking another.

Everything is measured through calibrate_community_powerlaw.run_and_measure(), which is the
calibration's own code path -- so the CV here is defined exactly as the 1.831 target is, and the
other five targets come along for free as guardrails.

Usage
-----
    python diagnose_heterogeneity.py            # the full grid at N_AGENTS_SWEEP
    python diagnose_heterogeneity.py --quick    # stage 1 only
    N_AGENTS=200000 python diagnose_heterogeneity.py --confirm 3.0,2.05
        # re-measure one (D_mean_short, alpha) cell at full population size

Writes csvs/heterogeneity_diagnosis.csv and prints the table. Runs nothing epidemiological -- these
are network-only sims with interventions stripped (build_sim already does that).
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sciris as sc

import powerlaw  # noqa: F401 -- installs the Pareto theta sampler at import, must precede the rest
import calibrate_community_powerlaw as cal
from calibrate_default_poisson import (
    N_AGENTS as N_AGENTS_DEFAULT, degree_from_edges, union_edges_window, active_union_window,
)


OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'
OUT_CSV = 'heterogeneity_diagnosis.csv'

# CV is a distributional statistic and converges fast, so the sweep runs smaller than the
# calibration harness's own 200k default; the winning cell is then re-measured at full size.
N_AGENTS_SWEEP = int(os.environ.get('N_AGENTS', 50_000))

# Natsal-derived, held fixed in every cell -- see the module docstring.
P_SINGLE_ANNUAL = 0.20

TARGET_CV = cal.TARGETS['cv_degree_annual']


def base_knobs():
    """The calibrated knob set, which is what run_and_measure expects."""
    return dict(
        mean_partners_per_year=1.5,
        frac_long=0.8662,
        D_mean_short=12.3,   # Natsal-3
        D_mean_long=239,     # Natsal-3, mortality-adjusted
        gamma_shape=2.05,    # GAMMA_SHAPE_FLOOR
    )


def measure(label, **overrides):
    """One grid cell. Returns the six calibration statistics plus the knobs that produced them."""
    knobs = base_knobs()
    knobs.update(overrides)
    cal.N_AGENTS = N_AGENTS_SWEEP
    cal.COMMUNITY_PARS = sc.mergedicts(powerlaw.COMMUNITY_PARS,
                                       dict(p_single_annual=P_SINGLE_ANNUAL))
    theta_cv = (1 / np.sqrt(knobs['gamma_shape'] * (knobs['gamma_shape'] - 2))
                if knobs['gamma_shape'] > 2 else np.inf)
    print(f'  [{label}] {knobs}  (theta CV {theta_cv:.2f}) ...', flush=True)
    try:
        stats = cal.run_and_measure(knobs)
    except Exception as e:
        print(f'    FAILED: {e}', flush=True)
        stats = {k: float('nan') for k in
                 ('mean_degree_annual', 'mean_degree_5yr', 'cv_degree_annual',
                  'p_single', 'p_long', 'p_short')}
    row = dict(label=label, theta_cv=theta_cv, **knobs, **stats)
    print(f"    cv_degree_annual {stats['cv_degree_annual']:.3f} (target {TARGET_CV})   "
          f"mean_degree_annual {stats['mean_degree_annual']:.3f} (target {cal.TARGETS['mean_degree_annual']})   "
          f"p_single {stats['p_single']:.3f} (target {cal.TARGETS['p_single']:.3f})", flush=True)
    return row


def report(rows):
    df = pd.DataFrame(rows)
    show = ['label', 'D_mean_short', 'gamma_shape', 'mean_partners_per_year', 'theta_cv',
            'cv_degree_annual', 'mean_degree_annual', 'p_single', 'p_long', 'p_short']
    print('\n' + '=' * 100)
    print(f'TARGETS: cv_degree_annual {TARGET_CV}   mean_degree_annual '
          f"{cal.TARGETS['mean_degree_annual']}   p_single {cal.TARGETS['p_single']:.3f}   "
          f"p_long {cal.TARGETS['p_long']}   p_short {cal.TARGETS['p_short']}")
    print('=' * 100)
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        print(df[show].to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    best = df.loc[df['cv_degree_annual'].idxmax()] if df['cv_degree_annual'].notna().any() else None
    if best is not None:
        print(f"\nHighest CV reached: {best['cv_degree_annual']:.3f} at {best['label']} "
              f"({best['cv_degree_annual'] / TARGET_CV:.0%} of target)")
        if best['cv_degree_annual'] < TARGET_CV:
            print('  -> No cell in this grid reaches the Natsal CV. If the best cell is also the')
            print('     shortest D_mean_short, duration is the binding constraint and the fix is')
            print('     structural (partnership timing or propensity form), not a knob setting.')
    outdir = pathlib.Path(OUTPUT_DIR); outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / OUT_CSV
    df.to_csv(path, index=False)
    print(f'\nWrote {path}')
    return df


def saturation(**overrides):
    """
    Where the heterogeneity is actually lost: the realised map from propensity to partner count.

    If annual degree were proportional to theta (a Poisson mixture with rate ~ theta), then a
    decile whose mean theta is k times the population mean would show k times the mean degree, and
    the degree CV would satisfy CV_D^2 = 1/E[D] + CV_theta^2. Any large shortfall against those two
    predictions is the network SATURATING -- a very high-theta node keeps being drawn as an
    endpoint but its extra draws land on partners it already has, or on an age/community block with
    too few available partners, so its distinct-partner count stops growing.

    Prints mean degree by theta decile against the proportional prediction, and the top-decile
    share of all partnerships against Natsal (0.350 female / 0.395 male, computed from the raw
    Natsal-3 file using natsal_analysis_working.ipynb's own cleaning rules).
    """
    knobs = base_knobs(); knobs.update(overrides)
    cal.N_AGENTS = N_AGENTS_SWEEP
    cal.COMMUNITY_PARS = sc.mergedicts(powerlaw.COMMUNITY_PARS,
                                       dict(p_single_annual=P_SINGLE_ANNUAL))
    n_years = cal.BURN_IN_YEARS + max(cal.MEASURE_YEARS, cal.WINDOW_LONG_YEARS)
    sim = cal.build_sim(knobs, n_years)
    sim.run()

    nh, act = sim.get_analyzer('network_history'), sim.get_analyzer('active_tracker')
    n, t_end = len(sim.people), sim.npts - 1
    spy = int(round(1 / cal.DT))
    deg = degree_from_edges(union_edges_window(nh, t_end, spy, 1), n)
    f_union, m_union = active_union_window(act, t_end, spy, 1)
    active = np.array(sorted(f_union | m_union), dtype=np.int64)

    # theta straight off the backend, for whoever is still in the network
    tu, tv = sim.network_backend._theta_true_u, sim.network_backend._theta_true_v
    theta = np.array([tu.get(i, tv.get(i, np.nan)) for i in active.tolist()])
    ok = np.isfinite(theta)
    active, theta, d = active[ok], theta[ok], deg[active][ok]

    print()
    print(f'SATURATION CHECK  {knobs}')
    print(f'  n={len(d):,}  mean degree={d.mean():.3f}  CV={d.std()/d.mean():.3f}  max={d.max()}')
    cv_pred = np.sqrt(1 / d.mean() + (theta.std() / theta.mean()) ** 2)
    print(f'  realised theta CV={theta.std()/theta.mean():.3f}  ->  Poisson-mixture predicted '
          f'degree CV={cv_pred:.3f}   actual={d.std()/d.mean():.3f}  '
          f'({d.std()/d.mean()/cv_pred:.0%} of prediction)')
    print()
    print(f"  {'theta decile':>13}{'mean theta':>12}{'x pop mean':>12}"
          f"{'mean degree':>13}{'proportional':>14}{'realised/pred':>15}")
    q = np.quantile(theta, np.linspace(0, 1, 11))
    for k in range(10):
        m = (theta >= q[k]) & (theta <= q[k + 1] if k == 9 else theta < q[k + 1])
        if not m.any():
            continue
        ratio = theta[m].mean() / theta.mean()
        pred = ratio * d.mean()
        print(f'  {k+1:>13}{theta[m].mean():>12.3f}{ratio:>12.2f}'
              f'{d[m].mean():>13.3f}{pred:>14.3f}{d[m].mean()/pred:>14.0%}')

    order = np.argsort(-d)
    top10 = order[:max(1, len(d) // 10)]
    print()
    print(f'  top 10% by realised degree hold {d[top10].sum()/d.sum():.1%} of all partnerships'
          f'   (Natsal: 35.0% female / 39.5% male)')
    return


def main():
    quick = '--quick' in sys.argv
    if '--saturation' in sys.argv:
        saturation()
        return
    print(f'Heterogeneity diagnosis: N_AGENTS={N_AGENTS_SWEEP}, p_single_annual={P_SINGLE_ANNUAL} '
          f'(fixed), target cv_degree_annual={TARGET_CV}\n')
    rows = []

    # Stage 1 -- duration. The suspect: a ~12-month casual tie caps annual partner turnover
    # regardless of theta. If CV climbs steeply here, duration is the binding constraint.
    print('Stage 1: D_mean_short (alpha pinned at its 2.05 floor)')
    for d in (12.3, 6.0, 3.0, 1.5):
        rows.append(measure(f'D={d}', D_mean_short=d))

    if not quick:
        # Stage 2 -- alpha, crossed with the shortest duration. If alpha still barely moves the CV
        # once duration is free, propensity heterogeneity is confirmed as second-order.
        print('\nStage 2: gamma_shape at the shortest duration')
        for a in (2.5, 3.0):
            rows.append(measure(f'D=1.5,alpha={a}', D_mean_short=1.5, gamma_shape=a))

        # Stage 3 -- is the CV simply mean-driven?
        print('\nStage 3: mean_partners_per_year')
        for m in (3.0,):
            rows.append(measure(f'mppy={m}', mean_partners_per_year=m))
            rows.append(measure(f'D=1.5,mppy={m}', D_mean_short=1.5, mean_partners_per_year=m))

    report(rows)


if __name__ == '__main__':
    main()

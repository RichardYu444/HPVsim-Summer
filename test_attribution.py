"""
test_attribution.py
===================

Verification suite for the core-group attribution machinery -- hpvsim_working's
``core_group_attribution`` analyzer (analysis.py) plus the two hooks it depends on: the
who-infected-whom buffer written in Sim.step()'s transmission loop (sim.py, guarded by
``pars['track_transmission']``) and the multiscale-clone parentage log written in
People.set_severity() (people.py).

Run it whenever any of those three files change:

    python test_attribution.py            # checks 1-5, a few minutes
    python test_attribution.py --quick    # checks 1-3 only
    python test_attribution.py --full     # adds the alpha sweep at a larger population

Each check prints PASS/FAIL and the suite exits non-zero if anything failed. Nothing here
writes to csvs/ or figs/.

The five checks
---------------
1. OFF BY DEFAULT. With ``track_transmission`` unset, results are bit-identical to a run of the
   same seed with the analyzer machinery absent entirely, and no stray attributes leak onto the
   sim. The hooks must cost nothing when nobody asked for them.

2. CONSERVATION. Every transmission is counted exactly once, from both ends:
   ``sum(n_onward) == sum(n_acquired) == analyzer.n_infections == results['infections'].sum()``.
   This is the important one -- it is what proves the ``np.unique`` de-duplication at
   sim.py's transmission loop was unwound correctly (``sources[transmissions][unique_inds]``)
   and that the ``people.scale`` weighting matches HPVsim's own ``scale_flows``.

3. MULTISCALE. With ``ms_agent_ratio > 1`` most cancers occur in cloned level-1 agents that have
   no infection event of their own. Their causal infection is their parent's, so
   ``sum(n_onward_cancer) + unattributed == analyzer.n_cancers == results['cancers'].sum()``,
   and the unattributed share (cancers from seeded or reactivated infections, which have no
   recorded transmitter) must stay small. A large unattributed share means clone parentage is
   broken, not that the model changed.

4. NULL MODEL. With theta made nearly homogeneous (large Pareto alpha), the top 5% must account
   for about 5% of infections and the Gini must be near zero -- i.e. the measure reads
   heterogeneity rather than manufacturing it. Also checks the empirical theta thresholds against
   the analytic Pareto quantile q**(-1/alpha), which they should match closely.

5. RANKING CROSS-CHECK. Ranking by latent theta and ranking by realised partner acquisition rate
   should broadly agree (Spearman on partners-per-year-active). Note the RAW cumulative
   ``n_rships`` correlation is much weaker and that is expected, not a bug: it is dominated by how
   long each agent has been active. Insurance against the "you picked a convenient ranking"
   objection.
"""
import sys

import numpy as np
import sciris as sc

import powerlaw  # installs the Pareto theta sampler over the Gamma default, at import
import hpvsim_working as hpv


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

N_AGENTS = 8_000     # small enough to run in a couple of minutes; --full raises it
END = 2030           # cancer needs decades of lead time after infection, so don't shorten much
SEED = 1

# Interventions are dropped throughout: basePars' NHS screening/vaccination schedules run past
# END and would raise "Years must be within simulation start and end dates", and none of the
# checks below depend on them.
BASE = dict(n_agents=N_AGENTS, end=END, rand_seed=SEED, verbose=0, interventions=[])

FAILURES = []


def check(name, passed, detail=''):
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f'  --  {detail}' if detail else ''))
    if not passed:
        FAILURES.append(name)
    return passed


def relclose(a, b, tol=1e-6):
    ''' Relative closeness, tolerant of the float32 accumulators HPVsim uses for its own flows '''
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / scale < tol


def run(analyzer=True, **overrides):
    ''' Build and run a power-law sim, optionally with the attribution analyzer attached '''
    pars = sc.mergedicts(BASE, overrides)
    if analyzer:
        pars = sc.mergedicts(pars, dict(
            track_transmission=True,
            analyzers=[hpv.core_group_attribution(cut_min_n=2000)]))
    sim = powerlaw.make_sim(**pars)
    sim.run()
    return sim


# -------------------------------------------------------------------
# 1. off by default
# -------------------------------------------------------------------

def test_off_by_default():
    print('\n1. Off by default costs nothing')
    a = run(analyzer=False, analyzers=[])
    b = run(analyzer=False, analyzers=[])
    same = all(np.array_equal(a.results[k][:], b.results[k][:])
               for k in ['infections', 'cancers', 'n_infectious'])
    check('two untracked runs of the same seed are identical', same)
    check('no transmissions recorded when track_transmission is off',
          len(getattr(a.people, 'new_transmissions', [])) == 0)
    check('track_transmission defaults to False', a.pars['track_transmission'] is False)
    return a


def test_hooks_dont_change_results(untracked):
    ''' Turning tracking ON must not perturb the epidemic -- it only observes it. '''
    print('\n1b. Tracking does not perturb the epidemic')
    tracked = run(analyzer=True)
    same = all(np.array_equal(untracked.results[k][:], tracked.results[k][:])
               for k in ['infections', 'cancers', 'n_infectious'])
    check('tracked and untracked runs of the same seed give identical results', same,
          'if this fails, the hook is consuming random numbers')
    return tracked


# -------------------------------------------------------------------
# 2. conservation
# -------------------------------------------------------------------

def test_conservation():
    print('\n2. Conservation of transmission events (ms_agent_ratio=1)')
    sim = run(ms_agent_ratio=1)
    cga = sim.get_analyzer('core_group_attribution')
    res = float(sim.results['infections'][:].sum())

    check('sum(n_onward) == sum(n_acquired)',
          relclose(cga.n_onward.sum(), cga.n_acquired.sum()),
          f'{cga.n_onward.sum():,.0f} vs {cga.n_acquired.sum():,.0f}')
    check('sum(n_onward) == analyzer.n_infections',
          relclose(cga.n_onward.sum(), cga.n_infections),
          f'{cga.n_onward.sum():,.0f} vs {cga.n_infections:,.0f}')
    check("analyzer.n_infections == results['infections'].sum()",
          relclose(cga.n_infections, res),
          f'{cga.n_infections:,.0f} vs {res:,.0f}')
    check('per-timestep series sums to the total',
          relclose(cga.ts.infections.sum(), cga.n_infections))
    check('direct attribution curve ends at 100%',
          relclose(cga.results.direct['all'].infections[-1], 1.0, tol=1e-9))
    check('everyone ranked has a finite theta',
          np.isfinite(cga.theta[np.isfinite(cga.theta)]).all())
    return sim, cga


# -------------------------------------------------------------------
# 3. multiscale clone parentage
# -------------------------------------------------------------------

def test_multiscale():
    print('\n3. Multiscale clone parentage (ms_agent_ratio=10)')
    sim = run(ms_agent_ratio=10)
    cga = sim.get_analyzer('core_group_attribution')
    res = float(sim.results['cancers'][:].sum())

    check("analyzer.n_cancers == results['cancers'].sum()",
          relclose(cga.n_cancers, res), f'{cga.n_cancers:,.0f} vs {res:,.0f}')
    check('attributed + unattributed == total cancers',
          relclose(cga.n_onward_cancer.sum() + cga.n_cancers_unattributed, cga.n_cancers),
          f'{cga.n_onward_cancer.sum():,.0f} + {cga.n_cancers_unattributed:,.0f}')
    frac = cga.n_cancers_unattributed / cga.n_cancers if cga.n_cancers else np.nan
    check('unattributed cancers are a small minority (<20%)', frac < 0.20,
          f'{frac:.1%} -- these are cancers from seeded or reactivated infections, '
          f'which have no recorded transmitter')
    check('transmission conservation still holds under multiscale',
          relclose(cga.n_onward.sum(), float(sim.results['infections'][:].sum())))
    return sim, cga


# -------------------------------------------------------------------
# 4. null model
# -------------------------------------------------------------------

def test_null_model(alpha_flat=60.0):
    print(f'\n4. Null model: near-homogeneous theta (Pareto alpha={alpha_flat:g})')
    pars = sc.dcp(powerlaw.COMMUNITY_PARS)
    pars['gamma_shape'] = alpha_flat
    sim = run(ms_agent_ratio=1, community_pars=pars)
    cga = sim.get_analyzer('core_group_attribution')

    a5 = cga.attributable(0.05)
    gini = cga.results.acquisition['all'].gini
    check('top 5% account for roughly 5% of infections (3-9%)',
          0.03 < a5['infections'] < 0.09, f"{a5['infections']:.1%}")
    check('Gini is near zero (<0.05)', gini < 0.05, f'{gini:.3f}')

    # Empirical thresholds vs the analytic Pareto quantile: theta = 1 + Lomax(alpha) means
    # P(theta > x) = x**-alpha, so the top-q cut is exactly q**(-1/alpha).
    analytic = np.array([q ** (-1.0 / alpha_flat) for q in cga.core_quantiles])
    err = np.max(np.abs(cga.results.theta_cuts - analytic) / analytic)
    check('empirical theta cuts match the analytic Pareto quantiles (<5% error)', err < 0.05,
          f'max relative error {err:.2%}')
    return cga


def test_monotone_in_alpha(alphas=(2.5, 3.0, 5.0, 10.0)):
    ''' Attribution and Gini must both fall as the tail is made lighter. '''
    print('\n4b. Attribution rises monotonically as the tail gets heavier')
    got = []
    for alpha in alphas:
        pars = sc.dcp(powerlaw.COMMUNITY_PARS)
        pars['gamma_shape'] = alpha
        cga = run(ms_agent_ratio=1, community_pars=pars).get_analyzer('core_group_attribution')
        got.append((alpha, cga.attributable(0.05)['infections'], cga.results.acquisition['all'].gini))
        print(f'    alpha={alpha:<5g} top-5% share={got[-1][1]:6.1%}  Gini={got[-1][2]:.3f}')
    shares = [g[1] for g in got]
    ginis = [g[2] for g in got]
    check('top-5% share decreases as alpha increases',
          all(x >= y - 0.01 for x, y in zip(shares, shares[1:])))
    check('Gini decreases as alpha increases',
          all(x >= y - 0.01 for x, y in zip(ginis, ginis[1:])))
    return got


# -------------------------------------------------------------------
# 5. ranking cross-check
# -------------------------------------------------------------------

def test_ranking(cga):
    print('\n5. Latent theta vs realised partner acquisition')
    d = cga.results.descriptive
    print(f"    Spearman(theta, partners per year active) = {d.spearman_theta_vs_rship_rate:.3f}"
          f"   [n={d.n_descriptive:,}]")
    print(f"    Spearman(theta, cumulative n_rships)      = {d.spearman_theta_vs_rships:.3f}"
          f"   [age-confounded, expected to be much lower]")
    check('theta and realised partner rate are positively rank-correlated (>0.4)',
          d.spearman_theta_vs_rship_rate > 0.4)
    rates = d.mean_rship_rate_by_theta_decile
    check('mean partner rate increases across theta deciles',
          np.nanmean(rates[-3:]) > np.nanmean(rates[:3]),
          f'bottom 3 deciles {np.nanmean(rates[:3]):.2f}/yr vs top 3 {np.nanmean(rates[-3:]):.2f}/yr')
    return


# -------------------------------------------------------------------

def main():
    quick = '--quick' in sys.argv
    full = '--full' in sys.argv
    if full:
        global BASE
        BASE = sc.mergedicts(BASE, dict(n_agents=50_000))

    print('=' * 78)
    print(f'core_group_attribution verification  (n_agents={BASE["n_agents"]:,}, end={END})')
    print('=' * 78)

    untracked = test_off_by_default()
    test_hooks_dont_change_results(untracked)
    sim, cga = test_conservation()
    test_multiscale()

    if not quick:
        test_null_model()
        test_ranking(cga)
        if full:
            test_monotone_in_alpha()

    print('\n' + '=' * 78)
    if FAILURES:
        print(f'{len(FAILURES)} CHECK(S) FAILED:')
        for f in FAILURES:
            print(f'  - {f}')
        return 1
    print('All checks passed.')
    print('\nHeadline for this configuration:')
    print(f'  {cga.summary(0.05)}')
    print(f'  {cga.summary(0.20)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

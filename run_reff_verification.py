"""
run_reff_verification.py
=========================

Analytic R0 and time-varying R_eff by genotype (reff_hpv.py), checked against what actually
happens in small simulations with transmission tracking.

Setup: the default network exactly as basePars.py / run_default_meandeg1p4.py configure it, small
population, 1950-END, two scenarios:

    vacc     NHS_Vacc.vaccinations (the non-targeted NHS programme), no screening
    novacc   no interventions

Screening is left out on purpose: the analytic model has no treatment step, and the earlier
factorial runs showed screening is not what drives the post-vaccination changes.

For each snapshot year Y and genotype g it computes, from the SAME simulation:

    R0            immune-naive, uniform index events (r0_hpv's window construction), all-events and
                  distinct counts
    R_vx          the same with vaccine immunity only (no natural immunity, nobody infected): what
                  the vaccination programme alone does to R
    R_intro       the same index events, but partners' real immunity/infection state at Y
                  (a case introduced now; the only option for hi5, which is extinct)
    analytic      the real cohort of cases infected in year Y, expected offspring from the
                  partnership timelines and the state at Y (reff_hpv.expected_offspring)
    analytic_no_cross   the same with natural cross-immunity removed -> how much it suppresses R
    analytic_no_comp    the same without competition from partners' other partners
    analytic_real_dur   diagnostic: the cohort's realised infection lengths instead of sampled ones
    empirical     the same cohort's realised offspring (TxLog)
    mean_field    R0 x sqrt(sigma_f sigma_m delta_f)

Usage:
    python run_reff_verification.py            # full: SEEDS=0,1,2 x vacc/novacc, N_AGENTS=20000
    QUICK=1 python run_reff_verification.py    # 5k agents, 2 years, 1 seed -- pipeline check

Outputs:
    csvs/reff_verification/<TAG>_summary.csv       pooled R by scenario/year/genotype/estimator
    csvs/reff_verification/<TAG>_prevalence.csv    yearly prevalence by genotype
    csvs/reff_verification/<TAG>_cases.pkl          per-case offspring (for bootstrap CIs)
    figs/Default/reff/*.png
"""
import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import sciris as sc

import hpvsim_working as hpv
import NHS_Vacc
import r0_hpv
import reff_hpv
import run_default_meandeg1p4 as rd

QUICK = bool(int(os.environ.get('QUICK', 0)))
N_AGENTS = int(os.environ.get('N_AGENTS', 5_000 if QUICK else 20_000))
START = 1950
END = int(os.environ.get('END', 2030 if QUICK else 2090))
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0' if QUICK else '0,1,2').split(',')]
SCENS = os.environ.get('SCENS', 'vacc,novacc').split(',')
YEARS = [1995, 2010] if QUICK else list(range(1995, 2051, 5))
REC_T0 = 1980
N_REP = int(os.environ.get('N_REP', 10))            # Monte Carlo repeats: R0, R_intro, variants
N_REP_MAIN = int(os.environ.get('N_REP_MAIN', 30))  # ... and the main cohort estimate (women's
                                                    # infection lengths are very heavy-tailed)
NCPUS = int(os.environ.get('NCPUS', 6))
TAG = os.environ.get('TAG', 'quick' if QUICK else f'default_{N_AGENTS // 1000}k_{START}_{END}')

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / 'csvs' / 'reff_verification'
FIGS = ROOT / 'figs' / 'Default' / 'reff'


def make_pars(scen, seed):
    interventions = sc.dcp(NHS_Vacc.vaccinations) if scen == 'vacc' else []
    return rd.make_pars(seed, n_agents=N_AGENTS, start=START, end=END,
                        interventions=interventions,
                        analyzers=[r0_hpv.R0Recorder(t0_year=REC_T0), reff_hpv.TxLog(),
                                   reff_hpv.StateSnapshots(YEARS)],
                        track_transmission=True,
                        ms_agent_ratio=1,  # no multiscale cancer clones (as validate_r0.py)
                        verbose=-1)


def _cases_frame(ev, off, **tags):
    cls = np.where(ev['female'].to_numpy(), 2, 0) + ev['B'].to_numpy().astype(int)
    df = off.drop(columns='ev').assign(cls=cls)
    for k, v in tags.items():
        df[k] = v
    return df


def analyse(sim, scen, seed):
    rec = sim.get_analyzer('R0Recorder')
    tx = sim.get_analyzer('TxLog').df
    ss = sim.get_analyzer('StateSnapshots')
    ctx = reff_hpv.make_context(sim, rec, ss)
    dt = sim['dt']
    W = int(round(1 / dt))
    gnames = list(sim['genotype_map'].values())
    sums, cases, mf = [], [], []

    def add_sums(est, Y, gname, ev, off, **extra):
        ks = reff_hpv.k_sums(ev, off)
        for c in ('all', 'dist'):
            sums.append(dict(scen=scen, seed=seed, year=Y, genotype=gname, estimator=est, count=c,
                             **{f'S{i}{j}': ks[c][i, j] for i in range(4) for j in range(4)},
                             **{f'n{j}': ks['n'][j] for j in range(4)}, **extra))

    for step, snap in sorted(ss.snaps.items()):
        Y = snap['year']
        uev = reff_hpv.uniform_events(ctx, step, W)
        for g, gname in enumerate(gnames):
            hist = reff_hpv.infected_history(ctx, tx, snap, g)
            # --- R0 and R_intro: uniform index events ---
            ev = uev.copy()
            ev['infector_avail'] = reff_hpv.naive_infector_avail(ctx, g, ev, None)
            off0 = reff_hpv.expected_offspring(ctx, g, ev, None, n_rep=N_REP)
            add_sums('R0', Y, gname, ev, off0)
            ks0 = reff_hpv.k_sums(ev, off0)
            R0_all = reff_hpv.r_from_sums(ks0['all'], ks0['n'])['rho']
            if scen == 'vacc':
                offv = reff_hpv.expected_offspring(ctx, g, ev, None, variant='vx_only', n_rep=N_REP)
                add_sums('R_vx', Y, gname, ev, offv)
            offi = reff_hpv.expected_offspring(ctx, g, ev, snap, n_rep=N_REP, hist=hist)
            add_sums('R_intro', Y, gname, ev, offi)
            m = reff_hpv.mean_field_reff(ctx, g, snap, R0_all, W)
            mf.append(dict(scen=scen, seed=seed, year=Y, genotype=gname, R0_all=R0_all, **m))

            # --- the real cohort infected in the year after the snapshot ---
            cev = reff_hpv.cohort_events(ctx, tx, g, step, step + W)
            if len(cev) == 0:
                continue
            for est, kw in (('analytic', dict(n_rep=N_REP_MAIN)),
                            ('analytic_no_cross', dict(variant='no_cross')),
                            ('analytic_no_comp', dict(competition=False)),
                            ('analytic_real_dur', dict(fixed_end=cev['real_end'].to_numpy()))):
                kw = {'n_rep': N_REP, **kw}
                off = reff_hpv.expected_offspring(ctx, g, cev, snap, hist=hist, **kw)
                add_sums(est, Y, gname, cev, off)
                if est == 'analytic':
                    cases.append(_cases_frame(cev, off, scen=scen, seed=seed, year=Y,
                                              genotype=gname, estimator=est))
            offe = reff_hpv.empirical_offspring(ctx, tx, g, cev)
            add_sums('empirical', Y, gname, cev, offe)
            cases.append(_cases_frame(cev, offe, scen=scen, seed=seed, year=Y, genotype=gname,
                                      estimator='empirical'))

    res = sim.results
    years = np.round(np.asarray(res['year'])).astype(int)
    prev = pd.DataFrame({f'prev_{g}': np.asarray(res['hpv_prevalence_by_genotype'][gi, :])
                         for gi, g in enumerate(gnames)}, index=years)
    prev['scen'], prev['seed'] = scen, seed
    return dict(sums=pd.DataFrame(sums), cases=pd.concat(cases) if cases else pd.DataFrame(),
                mf=pd.DataFrame(mf), prev=prev.reset_index().rename(columns={'index': 'year'}))


def run_one(scen, seed):
    np.random.seed(10_000 + seed)
    T = sc.timer()
    sim = hpv.Sim(make_pars(scen, seed), label=f'{scen}_{seed}')
    sim.run()
    t_run = T.toc(output=True)
    out = analyse(sim, scen, seed)
    print(f'{scen} seed {seed}: sim {t_run:.0f}s, analysis {T.toc(output=True) - t_run:.0f}s', flush=True)
    return out


def pool(sums):
    ''' Pool sufficient statistics over seeds -> K and R per scenario/year/genotype/estimator. '''
    keys = ['scen', 'year', 'genotype', 'estimator', 'count']
    Scols = [f'S{i}{j}' for i in range(4) for j in range(4)]
    ncols = [f'n{j}' for j in range(4)]
    g = sums.groupby(keys)[Scols + ncols].sum().reset_index()
    rows = []
    for _, r in g.iterrows():
        S = r[Scols].to_numpy(float).reshape(4, 4)
        n = r[ncols].to_numpy(float)
        R = reff_hpv.r_from_sums(S, n)
        rows.append({**{k: r[k] for k in keys}, 'rho': R['rho'], 'R_m': R['R_m'], 'R_f': R['R_f'],
                     'gen': R['gen'], 'n_m': n[:2].sum(), 'n_f': n[2:].sum()})
    return pd.DataFrame(rows)


def bootstrap(cases, n_boot=300, seed=0):
    ''' Case-resampling 95% interval for rho (all-events and distinct) of the cohort estimators. '''
    rng = np.random.default_rng(seed)
    rows = []
    for key, d in cases.groupby(['scen', 'year', 'genotype', 'estimator']):
        cls = d['cls'].to_numpy()
        by = [np.flatnonzero(cls == X) for X in range(4)]
        for c in ('all', 'dist'):
            A, B = d[f'offA_{c}'].to_numpy(), d[f'offB_{c}'].to_numpy()

            def rho(idx_by):
                S = np.zeros((4, 4))
                n = np.zeros(4)
                for X, ix in enumerate(idx_by):
                    rws = (2, 3) if X < 2 else (0, 1)
                    S[rws[0], X], S[rws[1], X], n[X] = A[ix].sum(), B[ix].sum(), len(ix)
                return reff_hpv.r_from_sums(S, n)['rho']

            bs = [rho([rng.choice(ix, len(ix)) if len(ix) else ix for ix in by]) for _ in range(n_boot)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append(dict(zip(['scen', 'year', 'genotype', 'estimator'], key), count=c, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(s, seed) for s in SCENS for seed in SEEDS]
    print(f'{len(jobs)} sims: {N_AGENTS} agents, {START}-{END}, snapshots {YEARS}', flush=True)
    T = sc.timer()
    outs = sc.parallelize(run_one, iterarg=jobs, ncpus=min(NCPUS, len(jobs)))
    sums = pd.concat([o['sums'] for o in outs])
    cases = pd.concat([o['cases'] for o in outs])
    mf = pd.concat([o['mf'] for o in outs])
    prev = pd.concat([o['prev'] for o in outs])
    summ = pool(sums)
    ci = bootstrap(cases[cases['estimator'].isin(['analytic', 'empirical'])])
    summ = summ.merge(ci, on=['scen', 'year', 'genotype', 'estimator', 'count'], how='left')
    mfp = mf.groupby(['scen', 'year', 'genotype'])[['R0_all', 'sigma_f', 'sigma_m', 'delta_f', 'R_mf']].mean().reset_index()
    summ.to_csv(OUT / f'{TAG}_summary.csv', index=False)
    mfp.to_csv(OUT / f'{TAG}_meanfield.csv', index=False)
    prev.to_csv(OUT / f'{TAG}_prevalence.csv', index=False)
    sums.to_csv(OUT / f'{TAG}_sums_by_seed.csv', index=False)
    with open(OUT / f'{TAG}_cases.pkl', 'wb') as f:
        pickle.dump(cases, f)
    T.toc('all done')
    show = summ[summ['count'] == 'all'].pivot_table(index=['scen', 'genotype', 'year'],
                                                    columns='estimator', values='rho')
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(show.round(3))
    return


if __name__ == '__main__':
    main()

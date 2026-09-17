"""
pool_r0_validation.py
=====================

Pools validate_r0.py runs across seeds: the ground-truth cases of every run go into one empirical
next-generation matrix (bootstrap SE over cases), next to the mean window / closed-form estimates.
With --diag it also evaluates the window estimator on the ground-truth cases themselves, which
separates "the estimator's mechanics are wrong" (realized != expected for the same cases) from
"the outbreak sampled different cases" (e.g. saturated hubs).

    python pool_r0_validation.py powerlaw --seed_frac 0.002 --rec_dir <dir given to --save_rec> --diag
"""
import argparse
import glob
import json
import os

import numpy as np
import sciris as sc

import r0_hpv


def rho(M):
    return float(np.max(np.abs(np.linalg.eigvals(M))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--genotype', default='hpv16')
    ap.add_argument('--seed_frac', default='0.01')
    ap.add_argument('--rec_dir', required=True)
    ap.add_argument('--diag', action='store_true')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    fs = sorted(glob.glob(os.path.join(here, 'csvs', f'r0_validation_{args.model}_{args.genotype}_seed*_sf{args.seed_frac}.json')))
    if not fs:
        raise SystemExit('no runs found')
    runs = []
    for f in fs:
        d = json.load(open(f))
        o = sc.load(os.path.join(args.rec_dir, os.path.basename(f).replace('r0_validation_', 'r0rec_').replace('.json', '.obj')))
        cases = o['rec'].ground_truth_ngm(o['pars'], n_boot=1)['cases']
        runs.append((d, o, cases))

    typ = np.concatenate([c['type'] for _, _, c in runs])
    A = np.concatenate([c['offA'] for _, _, c in runs])
    B = np.concatenate([c['offB'] for _, _, c in runs])

    def K_of(idx):
        K = np.zeros((4, 4))
        for X in range(4):
            m = idx[typ[idx] == X]
            rows = (2, 3) if X < 2 else (0, 1)
            if len(m):
                K[rows[0], X] = A[m].mean()
                K[rows[1], X] = B[m].mean()
        return K

    K = K_of(np.arange(len(typ)))
    rng = np.random.default_rng(1)
    boot = [rho(K_of(np.concatenate([rng.choice(np.flatnonzero(typ == X), (typ == X).sum()) for X in range(4)])))
            for _ in range(1000)]
    Kw = np.mean([np.array(d['estimate']['window']['K']) for d, _, _ in runs], 0)
    Kc = np.mean([np.array(d['estimate']['closed']['K']) for d, _, _ in runs], 0)
    par = [d['parameters_only']['R0'] for d, _, _ in runs if 'parameters_only' in d]

    np.set_printoptions(precision=3, suppress=True)
    print(f'{args.model} {args.genotype}, seed_frac {args.seed_frac}: {len(fs)} runs, cases by type '
          f'{dict(zip(r0_hpv.TYPES, np.bincount(typ, minlength=4)))}')
    print(f'  ground truth R0 = {rho(K):.3f} +- {np.std(boot):.3f}  (95% {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f})')
    print(f'  window R0 = {rho(Kw):.3f}   closed R0 = {rho(Kc):.3f}' + (f'   parameters-only R0 = {np.mean(par):.3f}' if par else ''))
    print('  K rows/cols = offspring/parent', r0_hpv.TYPES)
    print('  truth\n', K, '\n  window\n', Kw)

    if args.diag:
        tot = {name: np.zeros(5) for name in r0_hpv.TYPES}
        for _, o, c in runs:
            for X, name in enumerate(r0_hpv.TYPES):
                sel = c['type'] == X
                eA, eB = r0_hpv.event_offspring(o['rec'], o['pars'], 0, 'f' if X >= 2 else 'm',
                                                c['person'][sel], c['infector'][sel], c['t'][sel], n_rep=40)
                tot[name] += [sel.sum(), c['offA'][sel].sum(), eA.sum(), c['offB'][sel].sum(), eB.sum()]
        print('  same cases, realized vs window-expected offspring:')
        for name, t in tot.items():
            n = max(t[0], 1)
            print(f'    {name}: n={t[0]:.0f}  A {t[1]/n:.3f} vs {t[2]/n:.3f} | B {t[3]/n:.3f} vs {t[4]/n:.3f}')


if __name__ == '__main__':
    main()

"""
run_equilibrium_1900.py
=======================

Long warm-up check: simulate one network from 1900 to 2070 so the epidemic has ~80 years to reach
its own equilibrium before screening starts (GlobalScreeningParameters.screening_start_year = 1980)
and ~108 years before vaccination (2008). Everything else is the network's usual basePars
configuration; only the start year, the seed and (for the sweep) gamma_shape change.

One network and one seed per process -- importing basePars_community_powerlaw installs the
power-law theta sampler for the whole process, so power-law and Gamma runs must never share one.

    python run_equilibrium_1900.py default 0
    python run_equilibrium_1900.py gamma_0.25 3
    python run_equilibrium_1900.py powerlaw_3 1 --end 1905 --n_agents 200000   # quick pilot

Writes csvs/equilibrium_1900/<network>_seed<seed>.csv with one row per year, and prints the
run time and peak memory.
"""
import argparse
import pathlib
import time

import numpy as np
import pandas as pd
import psutil
import sciris as sc

import hpvsim_working as hpv

OUT_DIR = pathlib.Path(__file__).parent / 'csvs' / 'equilibrium_1900'


class Progress(hpv.Analyzer):
    ''' Prints year, agent count, memory and elapsed time every `every` simulated years. '''

    def __init__(self, every=10, **kwargs):
        super().__init__(**kwargs)
        self.every = every
        self.t0 = time.time()

    def apply(self, sim):
        year = sim.yearvec[sim.t]
        if sim.t % int(round(self.every / sim['dt'])) == 0:
            rss = psutil.Process().memory_info().rss / 1e6
            print(f'  {year:.0f}: agents {len(sim.people)}, alive {int(sim.people.alive.sum())}, '
                  f'memory {rss:.0f} MB, {time.time() - self.t0:.0f}s', flush=True)


def make_pars(network):
    if network == 'default':
        from basePars import base_pars_geno
        pars = sc.dcp(base_pars_geno)
    elif network.startswith('gamma_'):
        from basePars_community import base_pars_geno
        pars = sc.dcp(base_pars_geno)
        pars['community_pars']['gamma_shape'] = float(network.split('_')[1])
    elif network.startswith('powerlaw_'):
        from basePars_community_powerlaw import base_pars_geno  # installs the power-law sampler
        pars = sc.dcp(base_pars_geno)
        pars['community_pars']['gamma_shape'] = float(network.split('_')[1])
    else:
        raise ValueError(f'unknown network {network!r}')
    return pars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('network')
    ap.add_argument('seed', type=int)
    ap.add_argument('--start', type=int, default=1900)
    ap.add_argument('--end', type=int, default=2070)
    ap.add_argument('--n_agents', type=int, default=200_000)
    args = ap.parse_args()

    pars = make_pars(args.network)
    pars.update(start=args.start, end=args.end, n_agents=args.n_agents, rand_seed=args.seed,
                analyzers=[Progress()], verbose=-1)  # no network_history: it stores every step's edge deltas
    if args.start < 1950:
        # UK demographic data start in 1950. Death rates (nearest year) and birth rates (clamped
        # interpolation) already fall back to 1950 before then, but the initial age distribution
        # is looked up for the exact start year, so supply 1950's; and People.check_migration()
        # raises NotImplementedError for a start before the data, so migration must be off
        # (unlike the usual 1980-start runs).
        pars['age_datafile'] = str(OUT_DIR / 'uk_age_distribution_1950.csv')
        pars['use_migration'] = False
    if args.end < 2023:
        # Short pilot: screening (1980) and vaccination (2008-2022) start dates would fall outside
        # the sim, which HPVsim rejects
        pars['interventions'] = []

    T = time.time()
    sim = hpv.Sim(pars)
    sim.run()
    run_s = time.time() - T

    r = sim.results
    df = pd.DataFrame({'year': np.asarray(r['year'], dtype=float)})
    for key in ('hpv_prevalence', 'infections', 'n_infectious', 'n_alive', 'cin_prevalence',
                'cancers', 'cancer_incidence', 'asr_cancer_incidence'):
        df[key] = np.asarray(r[key][:], dtype=float)
    for gi, g in enumerate(sim['genotype_map'].values()):
        df[f'hpv_prevalence_{g}'] = np.asarray(r['hpv_prevalence_by_genotype'][gi, :], dtype=float)
    df['network'] = args.network
    df['seed'] = args.seed

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f'{args.network}_seed{args.seed}' + ('' if (args.start, args.end) == (1900, 2070) else f'_{args.start}-{args.end}')
    df.to_csv(OUT_DIR / f'{tag}.csv', index=False)

    peak_mb = psutil.Process().memory_info().peak_wset / 1e6
    print(f'DONE {tag}: {run_s:.0f}s, peak memory {peak_mb:.0f} MB, final agents {len(sim.people)}', flush=True)


if __name__ == '__main__':
    main()

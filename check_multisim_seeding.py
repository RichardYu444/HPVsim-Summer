"""
Checks how MultiSim seeds its runs on the community network, i.e. the assumption every run
script here makes when it writes Seed = seed + i: that MultiSim(sim).run(n_runs=N) runs seeds
sim['rand_seed'], sim['rand_seed']+1, ..., sim['rand_seed']+N-1, in that order.

1. Runs a MultiSim of N_RUNS from BASE_SEED and records, from inside each run, the rand_seed it
   used plus fingerprints of its population, community tags, initial network and partner
   propensities (theta), and its results.
2. Reruns seed BASE_SEED+1 as a plain Sim and checks it reproduces MultiSim run 1 exactly.
3. Runs the same MultiSim again and checks it reproduces itself exactly.

How the seed is used (run.single_run and Sim.initialize/run):
    single_run         rand_seed += run index
    Sim.initialize()   np.random.seed(rand_seed)   -> population, init_hpv_prev infections
                       default_rng(rand_seed)      -> community network backend (communities,
                                                      theta, all partnership formation/dissolution)
                       network calibration uses its own fixed seed (acbnm.calibrate seed=12345),
                       so rho differs between runs only through the population size
    Sim.run()          np.random.seed(rand_seed+1) -> everything during the run

Findings (2026-09-24): seeds are base+i in order, every run differs in all of the above, and a
given seed reproduces bit-for-bit, whether run alone or inside a MultiSim.
"""
import sys, pathlib, hashlib
import numpy as np
import sciris as sc

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import hpvsim_working as hpv
from basePars_community import base_pars_geno

BASE_SEED = 7
N_RUNS = 3
N_AGENTS = 5000
END = 1990


def digest(*arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:10]


class Fingerprint:
    ''' Analyzer: on the first step, fingerprint the population and network this run was built with '''

    def __init__(self):
        self.info = None

    def __call__(self, sim):
        if self.info is not None:
            return
        p = sim.people
        backend = sim.network_backend
        self.info = dict(
            rand_seed=sim['rand_seed'],
            population=digest(p.age, p.sex, p.debut),
            community=digest(p.community),
            network_t0=digest(backend.initial_snapshot.added_edges['f'],
                              backend.initial_snapshot.added_edges['m']),
            theta=digest(backend._state['u_theta'][:50]),
            rho=float(backend._params['rho']),
        )


def make_sim(seed):
    pars = sc.dcp(base_pars_geno)
    pars.update(n_agents=N_AGENTS, start=1980, end=END, rand_seed=seed, verbose=-1,
                interventions=[], analyzers=[Fingerprint()])
    return hpv.Sim(pars)


def summarise(sim):
    # Matched by name: sims come back from the worker processes with a pickled copy of the class
    fp = [a for a in sim.analyzers if type(a).__name__ == 'Fingerprint'][0].info
    prev = sim.results['hpv_prevalence'].values
    return dict(fp, final_rand_seed=sim['rand_seed'], results=digest(prev),
                prev_1985=round(float(prev[5]), 4), prev_1990=round(float(prev[-1]), 4))


def show(title, rows):
    print(f'\n{title}')
    for r in rows:
        print('  ' + '  '.join(f'{k}={v}' for k, v in r.items()))


if __name__ == '__main__':
    msim = hpv.MultiSim(make_sim(BASE_SEED))
    msim.run(n_runs=N_RUNS, n_cpus=N_RUNS)
    runs = [summarise(s) for s in msim.sims]
    show(f'MultiSim, base seed {BASE_SEED}, n_runs={N_RUNS}:', runs)

    single = make_sim(BASE_SEED + 1)
    single.run()
    one = summarise(single)
    show(f'Plain Sim(rand_seed={BASE_SEED + 1}).run():', [one])

    msim2 = hpv.MultiSim(make_sim(BASE_SEED))
    msim2.run(n_runs=N_RUNS, n_cpus=N_RUNS)
    runs2 = [summarise(s) for s in msim2.sims]

    print('\nChecks')
    print(f'  seeds used by the runs, in order : {[r["rand_seed"] for r in runs]}  '
          f'(expected {list(range(BASE_SEED, BASE_SEED + N_RUNS))})')
    print(f'  sim["rand_seed"] after the run   : {[r["final_rand_seed"] for r in runs]}')
    for key in ('population', 'community', 'network_t0', 'theta', 'results'):
        vals = [r[key] for r in runs]
        print(f'  {key:12s} differs across runs: {len(set(vals)) == len(vals)}')
    same = all(one[k] == runs[1][k] for k in ('population', 'community', 'network_t0', 'theta', 'results'))
    print(f'  Sim(seed {BASE_SEED + 1}) alone == MultiSim run 1 : {same}')
    print(f'  second MultiSim == first MultiSim : {runs == runs2}')

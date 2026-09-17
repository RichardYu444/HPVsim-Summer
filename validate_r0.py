"""
validate_r0.py
==============

Checks r0_hpv.py's R0 against a ground-truth one-generation outbreak in the same sim (see
r0_hpv.R0Recorder): random seeds infect a generation-1 cohort, only that cohort may transmit
onwards, and its distinct offspring -- split by sex and by whether each case was infected by an
existing (A) or a new (B) partner -- give an empirical 4x4 next-generation matrix whose spectral
radius is R0. pool_r0_validation.py pools several seeds.

One model and one genotype per process -- importing basePars_community_powerlaw installs the
power-law theta sampler for the whole process, and R0Recorder's generation control works on
rel_trans, which is per person, not per genotype.

    python validate_r0.py default  --genotype hpv16
    python validate_r0.py gamma    --genotype hpv16
    python validate_r0.py gamma1   --genotype hpv16     # gamma, but a single community
    python validate_r0.py powerlaw --genotype hpv16

Writes csvs/r0_validation_<model>_<genotype>_seed<seed>_sf<seed_frac>.json; with --save_rec DIR
also saves the recorder, which pool_r0_validation.py needs.
"""
import argparse
import json
import pathlib

import numpy as np
import sciris as sc

import hpvsim_working as hpv
import r0_hpv

OUT_DIR = pathlib.Path(__file__).parent / 'csvs'


def base_pars_for(model):
    if model == 'default':
        from basePars import base_pars_geno
        pars = sc.dcp(base_pars_geno)
        burn = 15  # the default network starts from scratch at `start`
    elif model in ('gamma', 'gamma1'):
        from basePars_community import base_pars_geno
        pars = sc.dcp(base_pars_geno)
        if model == 'gamma1':
            pars['community_pars'] = sc.mergedicts(pars['community_pars'], dict(
                n_communities=1, community_probs=np.array([1.0]),
                community_mixing=np.array([[1.0]]), community_labels=['All']))
        burn = 2  # CommunityNetworkBackend burns the network in before t=0
    elif model == 'powerlaw':
        from basePars_community_powerlaw import base_pars_geno
        pars = sc.dcp(base_pars_geno)
        burn = 2
    else:
        raise ValueError(f'unknown model {model!r}')
    return pars, burn


def make_pars(model, genotype, n_agents, seed, years, seed_frac, cohort_years):
    pars, burn = base_pars_for(model)
    t0_year = pars['start'] + burn
    rec = r0_hpv.R0Recorder(t0_year=t0_year, ground_truth=True, genotype=0,
                            seed_frac=seed_frac, cohort_years=cohort_years)
    pars.update(
        n_agents=n_agents,
        rand_seed=seed,
        end=t0_year + years,
        genotypes=[genotype],
        init_hpv_dist={genotype: 1.0},
        rel_init_prev=0.0,          # no infections except the recorder's seeds
        interventions=[],           # R0: no screening, no vaccination
        analyzers=[rec],
        track_transmission=True,
        ms_agent_ratio=1,           # no multiscale cancer clones
        verbose=0,
    )
    pars['genotype_pars'] = sc.objdict({genotype: pars['genotype_pars'][genotype]})
    return pars


def _estimator_pars(sim):
    ''' The sim parameters r0_hpv's estimators read, as a plain dict usable in place of the sim. '''
    keys = ['dt', 'beta', 'transf2m', 'transm2f', 'condoms', 'eff_condoms', 'genotype_pars',
            'genotype_map', 'n_genotypes', 'dur_infection_male', 'sev_dist', 'age_risk', 'acts',
            'community_pars']
    return {k: sc.dcp(sim.pars[k]) for k in keys if k in sim.pars}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=['default', 'gamma', 'gamma1', 'powerlaw'])
    ap.add_argument('--genotype', default='hpv16')
    ap.add_argument('--n_agents', type=int, default=100_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--years', type=float, default=35)
    ap.add_argument('--seed_frac', type=float, default=0.01)
    ap.add_argument('--cohort_years', type=float, default=1.0)
    ap.add_argument('--save_rec', default=None, help='directory to save the recorder in')
    args = ap.parse_args()

    T = sc.timer()
    pars = make_pars(args.model, args.genotype, args.n_agents, args.seed, args.years,
                     args.seed_frac, args.cohort_years)
    sim = hpv.Sim(pars)
    sim.run()
    rec = sim.get_analyzer('R0Recorder')
    t_run = T.toc(output=True)

    truth = rec.ground_truth_ngm(sim)
    T2 = sc.timer()
    est, stats = r0_hpv.estimate_r0(rec, sim)
    t_est = T2.toc(output=True)
    g = args.genotype
    out = dict(model=args.model, genotype=g, n_agents=args.n_agents, seed=args.seed,
               years=args.years, run_seconds=t_run, estimate=est[g], network=stats,
               truth={k: v for k, v in truth.items() if k != 'cases'})
    if args.model in ('gamma', 'gamma1', 'powerlaw'):
        par_only, info = r0_hpv.r0_from_parameters(sim, powerlaw=(args.model == 'powerlaw'))
        out['parameters_only'] = par_only[g]
        out['parameters_only_info'] = info

    np.set_printoptions(precision=3, suppress=True)
    print(f'\n=== {args.model} / {g}  ({args.n_agents} agents, seed {args.seed}, sim {t_run:.0f}s, estimate {t_est:.0f}s) ===')
    print(f"ground truth R0 = {truth['R0']:.3f} ± {truth['R0_se']:.3f}   (cases by type {dict(zip(r0_hpv.TYPES, truth['n_by_type'].astype(int)))})")
    print(f"   counting reinfections too: R0 = {truth['R0_all']:.3f};  prevalence after cohort {truth['prev_at_end_cohort']:.2%}")
    print(f"window  R0 = {est[g]['window']['R0']:.3f}")
    print(f"closed  R0 = {est[g]['closed']['R0']:.3f}")
    if 'parameters_only' in out:
        print(f"params  R0 = {out['parameters_only']['R0']:.3f}   (h = 1+CV^2 = {info['h']:.2f})")
    print('K rows/cols = offspring/parent types', r0_hpv.TYPES)
    for label, K in (('truth', truth['K']), ('window', est[g]['window']['K']), ('closed', est[g]['closed']['K'])):
        print(f'{label}:\n{K}')
    for sex in ('m', 'f'):
        for X in ('A', 'B'):
            st = stats[sex][X]
            print(f"  {X}_{sex}: n={st['n_events']}  S={ {k: round(v, 3) for k, v in st['S'].items()} }  "
                  f"C={ {k: round(v, 3) for k, v in st['C'].items()} }")
    print('  q per year:', {k: round(v, 3) for k, v in stats['q'].items()})

    OUT_DIR.mkdir(exist_ok=True)
    tag = f'{args.model}_{g}_seed{args.seed}_sf{args.seed_frac:g}'
    if args.save_rec:  # recorder + minimal sim pars, to re-run the estimators without the sim
        sc.save(pathlib.Path(args.save_rec) / f'r0rec_{tag}.obj', dict(rec=rec, pars=_estimator_pars(sim)))
    fn = OUT_DIR / f'r0_validation_{tag}.json'
    with open(fn, 'w') as f:
        json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
    print(f'wrote {fn}')


if __name__ == '__main__':
    main()

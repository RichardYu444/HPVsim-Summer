"""
run_powerlaw_attribution.py
===========================

Core-group attribution over 50 runs of the power-law network: how much of the epidemic the most
sexually active tail of the population is responsible for.

Same shape as run_sim_community_50.py -- MultiSim in batches of 5 across 10 base seeds, one
stacked CSV with a Seed column and a single header row -- but built on powerlaw.make_sim() and
carrying hpv.core_group_attribution instead of the by-community results. See that analyzer's
docstring (hpvsim_working/analysis.py) for what each measure means and the literature behind it.

Three outputs into OUTPUT_DIR:

    <TAG>_attribution.csv   one tidy row per (run, measure, quantile); the direct-attribution
                            shares, the downstream-by-generation shares, and the acquisition
                            Gini. This is what plot_attribution.py reads.
    <TAG>_curves.npz        the full attribution and Lorenz CURVES, one row per run, on a shared
                            200-point log grid of population fractions. Too wide for the CSV, and
                            what the headline figure is actually drawn from.
    <TAG>_summary.txt       the human-readable headline lines, median across runs.

Each batch is released as soon as its curves have been extracted, so no per-agent array is
retained past the end of the batch it came from.
"""
import pathlib

import numpy as np
import sciris as sc

import powerlaw  # installs the Pareto theta sampler at import -- must precede make_sim()
import hpvsim_working as hpv


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

SIM_LABEL = 'powerlaw attribution'
TAG = 'powerlaw_alpha3_200k_50runs'  # IMPORTANT TO CHANGE EVERY TIME
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

N_RUNS = 5  # due to multisim stuff I think 5 is max I can run on a 6 core cpu
N_CPUS = 5

seeds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 seeds gets us to 5 * 10 = 50 total runs (0-49)

# Top-q cuts the DOWNSTREAM (multi-generation) measure is tracked for. These have to be fixed
# before the run because generation counting needs the thresholds live. The direct-attribution
# and Lorenz curves are computed post hoc from per-agent counters and so are available at ANY
# percentile regardless of what is listed here -- see CURVE_GRID below.
CORE_QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20)

# Generation depths N for the downstream measure: "descended from a top-q transmitter within N
# generations". N=0 (the direct measure) is always included.
GEN_CAPS = (1, 2, 3)

# Population fractions the exported curves are sampled at, 0.1% to 100%, log-spaced.
CURVE_GRID = np.logspace(-3, 0, 200)

# Percentiles written into the tidy CSV as explicit rows.
REPORT_QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)

# Rank within sex rather than pooled. Worth a second run with this True: the two sides of the
# bipartite network have different sizes, so equal theta does not mean equal degree.
BY_SEX = False

# Extra pars merged into every sim, on top of powerlaw.make_sim()'s own. Empty = use basePars as
# is, i.e. the full 200_000 agents over 1980-2055 with the NHS interventions in place.
#
# MEMORY HISTORY: an early attempt at five concurrent 200k-agent workers reached 24.1 GB of commit
# charge against a 25.4 GB limit ~13 minutes in, with 75 simulated years of population growth and
# multiscale cancer clones still to accumulate, and n_agents was temporarily halved to work around
# it. That baseline memory problem has since been resolved, so the full 200k runs as intended and
# no override is needed. HPVsim's own People arrays dominate the footprint; the attribution
# analyzer contributes only ~100 MB per worker.
#
# If you do need to shrink it again, note that every attribution measure is a RATIO (share of
# infections, share of cancers, Gini), so pop_scale cancels out of all of them and absolute counts
# stay comparable. The only cost is Monte Carlo noise, which widens the CANCER curve's IQR first:
# cancer is the rare endpoint, so it thins out fastest as agents are removed. For a quick
# shakedown before committing to the full 50 runs:
#     SIM_OVERRIDES = dict(n_agents=8_000, end=2030, interventions=[])
SIM_OVERRIDES = {}

# NB basePars.py's own analyzers=[hpv.network_history()] is REPLACED, not extended, by the
# analyzers= override passed to make_sim() below -- which is what we want. The attribution
# analyzer works off the transmission buffer, not the network deltas, and at 200k agents x 50
# runs keeping a NetworkDelta per timestep is a lot of memory for nothing.


def curves_for(cga):
    """(direct-infections, direct-cancers, lorenz-acquired) sampled on CURVE_GRID."""
    direct_inf = np.array([cga.attributable(q)['infections'] for q in CURVE_GRID])
    direct_can = np.array([cga.attributable(q)['cancers'] for q in CURVE_GRID])
    acq = cga.results.acquisition['female' if BY_SEX else 'all']
    lorenz = np.interp(CURVE_GRID, acq.pop_frac, acq.acquired)
    return direct_inf, direct_can, lorenz


def main():
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f'{TAG}_attribution.csv'
    npz_path = outdir / f'{TAG}_curves.npz'
    txt_path = outdir / f'{TAG}_summary.txt'
    print(f'Outputs will be saved to: {csv_path}')

    for path in (csv_path, npz_path):
        if path.exists():
            # Appending onto an existing file would interleave two different runs' results
            errormsg = (f'{path} already exists -- move or delete it first, otherwise these '
                        f'runs would be appended onto the previous ones.')
            raise FileExistsError(errormsg)

    group = 'female' if BY_SEX else 'all'
    header_written = False
    curves = dict(grid=CURVE_GRID, seeds=[], direct_infections=[], direct_cancers=[],
                  lorenz_acquired=[], gini=[], theta_cuts=[])

    for seed in seeds:
        sim = powerlaw.make_sim(
            rand_seed=seed,
            track_transmission=True,  # tells Sim.step() to publish infector identities
            analyzers=[hpv.core_group_attribution(
                core_quantiles=CORE_QUANTILES, gen_caps=GEN_CAPS, by_sex=BY_SEX)],
            **SIM_OVERRIDES,
        )
        sim.label = SIM_LABEL  # a Sim attribute, not a par -- make_sim() merges everything into pars
        print('Created HPVsim simulation.')

        print(f'Running MultiSim with n_runs = {N_RUNS}  (seeds {seed}-{seed + N_RUNS - 1}) ...')
        msim = hpv.MultiSim(sim)
        msim.run(n_runs=N_RUNS, n_cpus=N_CPUS)
        print('MultiSim run complete.')

        for i, run_sim in enumerate(msim.sims):
            this_seed = seed + i
            try:
                cga = run_sim.get_analyzer('core_group_attribution')
                temp_df = cga.to_df(quantiles=REPORT_QUANTILES, group=group)
            except Exception as e:
                print(f'Could not extract attribution for seed {this_seed}: {e}')
                continue
            temp_df['Seed'] = this_seed
            temp_df.to_csv(csv_path, mode='a', index=False, header=not header_written)
            header_written = True

            di, dc, lz = curves_for(cga)
            curves['seeds'].append(this_seed)
            curves['direct_infections'].append(di)
            curves['direct_cancers'].append(dc)
            curves['lorenz_acquired'].append(lz)
            curves['gini'].append(cga.results.acquisition[group].gini)
            curves['theta_cuts'].append(cga.results.theta_cuts)
            print(f'Seed:{this_seed} is done  --  {cga.summary(0.05, group=group)}')

        # Release the batch before building the next one: each worker's analyzer carries per-agent
        # arrays, and holding ten batches' worth at once is what would actually blow the memory.
        del msim

    curves['core_quantiles'] = np.array(CORE_QUANTILES)
    np.savez_compressed(npz_path, **{k: np.array(v) for k, v in curves.items()})

    # Headline lines, median and IQR across every run that made it into the curves
    n_done = len(curves['seeds'])
    di = np.array(curves['direct_infections'])
    dc = np.array(curves['direct_cancers'])
    gini = np.array(curves['gini'])
    lines = [f'{TAG}  ({n_done} runs, ranked by latent theta, group={group})', '']
    for q in (0.01, 0.05, 0.10, 0.20):
        j = int(np.argmin(np.abs(CURVE_GRID - q)))
        lines.append(
            f'  top {q:>5.1%}:  {np.median(di[:, j]):5.1%} of infections '
            f'[IQR {np.quantile(di[:, j], 0.25):.1%}-{np.quantile(di[:, j], 0.75):.1%}],  '
            f'{np.median(dc[:, j]):5.1%} of cancers '
            f'[IQR {np.quantile(dc[:, j], 0.25):.1%}-{np.quantile(dc[:, j], 0.75):.1%}]')
    lines += ['', f'  acquisition Gini: {np.median(gini):.3f} '
                  f'[IQR {np.quantile(gini, 0.25):.3f}-{np.quantile(gini, 0.75):.3f}]']
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))

    print(f'\nDone. Wrote {csv_path}, {npz_path} and {txt_path}')


if __name__ == '__main__':
    main()

"""
run_core_vacc.py
================

50 full runs of the power-law network under one of two vaccination arms:

    'baseline'  NHS_Vacc.py            -- the NHS programme as modelled today
    'core'      core_vacc.py           -- 100% of the top 10% most active, general coverage
                                          reduced to hold the total dose count fixed

Run it once per arm -- either by editing ARM below, or by passing the arm as an argument:

    python run_core_vacc.py baseline
    python run_core_vacc.py core

That gives two CSVs in csvs/ that are directly comparable and in the same format plot_IQR.py
already reads, so any comparison you want to draw is a pandas join away. No comparison script is
produced here on purpose.

WHY THE BASELINE HAS TO BE RE-RUN
---------------------------------
Both arms import theta_predraw, which fixes each agent's activity propensity theta at birth
rather than at debut so the core group is identifiable at vaccination age (12-13). That changes
the theta VALUES the network sees -- not its rng consumption, but the values -- so earlier
NHS_Vacc outputs produced without the patch are NOT a valid control. Run ARM='baseline' here.

The imports of powerlaw, theta_predraw and core_vacc are at MODULE level and must stay there:
both powerlaw's Pareto sampler and theta_predraw's patch are installed at import, and sciris
parallelize spawns workers on Windows that re-import __main__. An import tucked inside main()
would leave every worker unpatched.

Shaped on run_powerlaw_attribution.py -- MultiSim in batches of 5 across 10 base seeds, one
stacked CSV with a Seed column and a single header row -- but exporting the standard results
dataframe rather than attribution rows.

Interruptible: batches are appended to the CSV as they finish, and RESUME (below) makes a
re-invocation skip whatever is already in the file. A run cut short by a machine sleeping or a
session ending is continued simply by running the same command again.

Two outputs into OUTPUT_DIR:

    <TAG>.csv         50 runs stacked; year index, t, every 1-D result, Seed. Includes
                      cum_doses / cum_vaccinated alongside infections / cancer_incidence /
                      hpv_prevalence, so dose-neutrality can be checked rather than trusted.
    <TAG>_doses.txt   the headline dose counts, median and IQR across runs.

To also carry the core-group attribution measures, pass track_transmission=True and
analyzers=[hpv.core_group_attribution(...)] to make_sim() below -- but that is a separate
question from this one and costs memory, so it is off here.
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sciris as sc

import powerlaw       # installs the Pareto theta sampler at import -- must precede make_sim()
import theta_predraw  # installs the pre-debut theta patch at import -- must precede make_sim()
import hpvsim_working as hpv
import NHS_2025_lambdamu
import NHS_Vacc
import core_vacc


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

# 'core' or 'baseline' -- run once with each. Overridable from the command line
# (`python run_core_vacc.py baseline`) so both arms can be run back to back without editing this
# file between them. Read defensively: spawned workers re-import this module and their argv is not
# guaranteed to match the parent's, but they only ever run the already-pickled sim, so a
# disagreement here is harmless.
ARM = sys.argv[1] if len(sys.argv) > 1 else 'core'

# Output name. The baseline arm is plain NHS_Vacc and does NOT depend on core_vacc.CORE_FRAC, so
# it keeps one fixed name and one file: a single baseline run is a valid control for every core
# fraction, and re-running it per fraction would be pure waste. The core arm is named by its
# fraction instead, so 10% and 20% land in different files.
# Pareto tail index for the propensity distribution. Environment-set for the same reason
# CORE_FRAC is (inherited by spawned workers where an argv-derived constant would not be).
#
# powerlaw.COMMUNITY_PARS' own value of 3 is an UNCALIBRATED carryover -- it came from
# recreation.py where 3 was a Poisson lambda, and was never revised. The calibration against
# Natsal's cv_degree_annual=1.831 selected 2.05, which is GAMMA_SHAPE_FLOOR, i.e. the guardrail
# just above the finite-variance pole at alpha=2 (see calibrate_community_powerlaw.py:95). It is a
# censored boundary value rather than a converged fit: even there the realised annual-degree CV
# only reaches 0.988. Lower alpha = heavier tail = more heterogeneity.
#
#     ALPHA=2.05 CORE_FRAC=0.20 python run_core_vacc.py core
#
# COMMUNITY_PARS itself is deliberately NOT edited -- calibrate_community_powerlaw.py reads it as
# the calibration's starting point, and run_powerlaw_attribution.py / test_attribution.py depend
# on the current default.
ALPHA = float(os.environ.get('ALPHA', powerlaw.COMMUNITY_PARS['gamma_shape']))
_A = f'{ALPHA:g}'.replace('.', 'p')

# Output name. The baseline arm is plain NHS_Vacc and does NOT depend on core_vacc.CORE_FRAC, so
# one baseline run serves every core fraction at the same alpha; re-running it per fraction would
# be pure waste. The core arm is named by its fraction as well.
TAG = (f'powerlaw_alpha{_A}_200k_baseline_vacc' if ARM == 'baseline'
       else f'powerlaw_alpha{_A}_200k_top{round(core_vacc.CORE_FRAC * 100)}_core_vacc')
if ALPHA == 3 and ARM == 'baseline':
    TAG = 'powerlaw_alpha3_200k_50runs_baseline_vacc'  # the 50 runs already on disk
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

N_RUNS = 5  # due to multisim stuff I think 5 is max I can run on a 6 core cpu
N_CPUS = 5

seeds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 seeds gets us to 5 * 10 = 50 total runs (0-49)

# Number of leading seed batches to run, N_RUNS each. 10 (the default) is the full 50 runs; set it
# lower for a quick look. Because RESUME skips whatever is already in the CSV, a short run can be
# extended to the full 50 later just by re-running with a larger value -- only the missing batches
# are computed.
#
#     N_BATCHES=3 CORE_FRAC=0.20 python run_core_vacc.py core
seeds = seeds[:int(os.environ.get('N_BATCHES', len(seeds)))]

# Extra pars merged into every sim, on top of powerlaw.make_sim()'s own. Empty = the full 200_000
# agents over 1980-2055. For a quick shakedown before committing to the full 50 runs:
#     SIM_OVERRIDES = dict(n_agents=20_000, end=2030)
SIM_OVERRIDES = {}

# basePars.py's analyzers=[hpv.network_history()] keeps a NetworkDelta for every timestep of every
# run. Nothing here reads it, and at 200k agents x 50 runs it is a lot of memory for nothing --
# same reasoning as run_sim_community_50.py's DROP_NETWORK_HISTORY.
DROP_NETWORK_HISTORY = True

# If the output CSV already exists, continue from where a previous attempt stopped rather than
# refusing to start. These runs take hours and the batches are written incrementally, so an
# interruption (machine sleeping, session ending, Ctrl-C) otherwise throws away everything done so
# far. Seeds already present in the CSV are skipped and never rewritten, so resuming cannot
# duplicate or interleave rows. Set False to get the old hard failure instead.
RESUME = True


VACCINATIONS = {'baseline': NHS_Vacc.vaccinations, 'core': core_vacc.vaccinations}


def make_sim(seed):
    if ARM not in VACCINATIONS:
        raise ValueError(f'ARM must be one of {sorted(VACCINATIONS)}, got {ARM!r}')
    overrides = dict(SIM_OVERRIDES)
    if DROP_NETWORK_HISTORY:
        overrides['analyzers'] = []
    return powerlaw.make_sim(
        rand_seed=seed,
        community_pars=sc.mergedicts(powerlaw.COMMUNITY_PARS, dict(gamma_shape=ALPHA)),
        interventions=NHS_2025_lambdamu.get_interventions(l=1, m=1) + VACCINATIONS[ARM],
        **overrides,
    )


def main():
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f'{TAG}.csv'
    txt_path = outdir / f'{TAG}_doses.txt'
    frac = f', core fraction {core_vacc.CORE_FRAC:.0%}' if ARM == 'core' else ''
    print(f"Arm: {ARM!r}  ({len(VACCINATIONS[ARM])} vaccination interventions{frac})")
    print(f'Pareto alpha: {ALPHA}  (theta CV = {1/np.sqrt(ALPHA*(ALPHA-2)):.3f})')
    print(f'Seed batches: {seeds}  -> up to {len(seeds) * N_RUNS} runs')
    print(f'Outputs will be saved to: {csv_path}')

    done_seeds = set()
    header_written = False
    if csv_path.exists():
        if not RESUME:
            # Appending onto an existing file would interleave two different runs' results
            errormsg = (f'{csv_path} already exists -- move or delete it first, otherwise these '
                        f'runs would be appended onto the previous ones.')
            raise FileExistsError(errormsg)
        prev = pd.read_csv(csv_path, index_col=0)
        done_seeds = {int(x) for x in prev['Seed'].unique()}
        header_written = True
        print(f'RESUME: {len(done_seeds)} run(s) already present, seeds '
              f'{min(done_seeds)}-{max(done_seeds)}. Those will be skipped.')

    for seed in seeds:
        batch = [seed + i for i in range(N_RUNS)]
        if all(s in done_seeds for s in batch):
            print(f'Skipping seeds {batch[0]}-{batch[-1]} (already in the CSV).')
            continue
        sim = make_sim(seed)
        sim.label = f'{ARM} vaccination'  # a Sim attribute, not a par
        print('Created HPVsim simulation.')

        print(f'Running MultiSim with n_runs = {N_RUNS}  (seeds {seed}-{seed + N_RUNS - 1}) ...')
        msim = hpv.MultiSim(sim)
        msim.run(n_runs=N_RUNS, n_cpus=N_CPUS)
        print('MultiSim run complete.')

        for i, run_sim in enumerate(msim.sims):
            this_seed = seed + i
            if this_seed in done_seeds:  # partially-written batch: keep the rows already on disk
                print(f'Seed:{this_seed} already in the CSV, not rewriting.')
                continue
            try:
                temp_df = run_sim.to_df(date_index=True)
            except Exception as e:
                print(f'Could not save run results to df: {e}')
                continue
            temp_df['Seed'] = this_seed
            # Header on the first write only, so the CSV is directly readable with
            # pd.read_csv(..., index_col=0) and needs no cleaning pass
            temp_df.to_csv(csv_path, mode='a', index=True, header=not header_written)
            header_written = True

            doses = float(run_sim.results['cum_doses'][-1])
            vacc = float(run_sim.results['cum_vaccinated'][-1])
            print(f'Seed:{this_seed} is done  --  cum_doses={doses:,.0f}  cum_vaccinated={vacc:,.0f}')

        del msim

    # Headline dose counts, so dose-neutrality between the arms is visible without post-processing.
    # Read back off the CSV rather than accumulated in memory, so a resumed run summarises every
    # seed in the file and not just the ones this invocation happened to run.
    final = pd.read_csv(csv_path, index_col=0).groupby('Seed')[['cum_doses', 'cum_vaccinated']].last()
    d, v = final['cum_doses'].to_numpy(), final['cum_vaccinated'].to_numpy()
    lines = [f'{TAG}  ({len(d)} runs, arm={ARM})', '']
    for name, arr in (('cum_doses', d), ('cum_vaccinated', v)):
        lines.append(f'  {name:<16} median {np.median(arr):>14,.0f}  '
                     f'[IQR {np.quantile(arr, 0.25):,.0f}-{np.quantile(arr, 0.75):,.0f}]')
    lines += ['', '  Compare cum_doses against the other arm: the two should agree to within ~1%,',
              '  the residual being the timestep/mortality boundary effects documented in',
              "  core_vacc.py's module docstring (section 2)."]
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print()
    print('\n'.join(lines))

    print(f'\nDone. Wrote {csv_path} and {txt_path}')


if __name__ == '__main__':
    main()

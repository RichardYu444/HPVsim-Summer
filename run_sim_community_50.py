"""
50 full runs of the community (gamma) simulation with community-stratified results on.

Same shape as run_sim.py -- MultiSim in batches of 5 across 10 base seeds -- but with
pars['community_results'] set, so every run's dataframe carries the by-community network and
epidemic results as well as the usual ones. Which of those appear is controlled by
COMMUNITY_RESULTS (what the sim stores) and EXPORT_BY_COMMUNITY (what reaches the CSV) below.

Output: one CSV at OUTPUT_DIR/ALLRUNS with a single header row, holding all 50 runs
stacked. Column order is

    year (index), t, <all the existing 1-D results>, <all the by-community results>, Seed

i.e. the by-community columns come in series after the original results, one column per
community per result, named <result>_<community label> (e.g. hpv_prevalence_by_community_White,
mean_degree_by_community_Asian). Labels come from basePars_community.ETHNICITIES via
community_pars['community_labels'].

Unlike run_sim.py this writes the header only once, so the output does NOT need
clean_sweep_csv.py afterwards.
"""
import pathlib
import numpy as np
import hpvsim_working as hpv
from basePars_community import base_pars_geno

# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

SIM_LABEL = 'community network'
ALLRUNS = 'community_gamma2_50runs.csv'  # IMPORTANT TO CHANGE EVERY TIME
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

N_RUNS = 5  # due to multisim stuff I think 5 is max I can run on a 6 core cpu
N_CPUS = 5

seeds = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 seeds gets us to 5 * 10 = 50 total runs (0-49)

# Which results to stratify by community. True gives all 28 of them (112 columns, which is a lot);
# False turns the whole feature off. Can also be a preset name -- 'epi', 'rates', 'flows',
# 'stocks', 'network', 'minimal', 'all' -- or an explicit list like this one. A list is widened
# automatically to include whatever each entry needs to be computed (e.g. asking for
# hpv_prevalence also stores n_infectious and n_alive), so more results are stored than listed.
COMMUNITY_RESULTS = ['infections', 'cancers', 'hpv_prevalence', 'cancer_incidence',
                     'mean_degree', 'within_edge_frac', 'single_frac']

# Which of the stored results actually get written to the CSV, in the same formats as above.
# True writes everything that was stored, including the dependencies pulled in above; the list
# here writes exactly these and nothing else. Must be a subset of what COMMUNITY_RESULTS stored.
EXPORT_BY_COMMUNITY = COMMUNITY_RESULTS

# network_history stores a NetworkDelta for every timestep of every run. None of the
# by-community results need it (they are computed from people.community + people.contacts
# during the run), and at 200k agents x 50 runs it is a lot of memory for nothing, so it
# is dropped here. Set to False if you do want the deltas kept.
DROP_NETWORK_HISTORY = True


def main():
    # Ensure output directory exists
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    allruns_path = outdir / ALLRUNS
    print(f'Outputs will be saved to: {allruns_path}')

    if allruns_path.exists():
        # Appending onto an existing file would interleave two different runs' results
        errormsg = (f'{allruns_path} already exists -- move or delete it first, otherwise these '
                    f'runs would be appended onto the previous ones.')
        raise FileExistsError(errormsg)

    base_pars_geno['community_results'] = COMMUNITY_RESULTS  # the toggle: adds the by-community results
    if DROP_NETWORK_HISTORY:
        base_pars_geno['analyzers'] = []

    # Community labels (ethnicities) are what name the CSV columns; warn early rather than
    # silently falling back to 'Community 0', 'Community 1', ... after hours of running
    labels = base_pars_geno['community_pars'].get('community_labels')
    if labels is None:
        print('WARNING: community_pars has no community_labels; columns will be named by index.')
    else:
        print(f'Communities: {list(labels)}')

    # Run through seeds to run sim, 5 at a time, hence the gaps in seeds
    header_written = False
    for seed in seeds:
        base_pars_geno['rand_seed'] = seed
        # Build simulation
        sim = hpv.Sim(base_pars_geno, label=SIM_LABEL)
        print('Created HPVsim simulation.')
        # Run MultiSim
        print(f'Running MultiSim with n_runs = {N_RUNS}  (seeds {seed}-{seed + N_RUNS - 1}) ...')
        msim = hpv.MultiSim(sim)
        msim.run(n_runs=N_RUNS, n_cpus=N_CPUS)
        print('MultiSim run complete.')

        # i helps keep track of which seed we are on within this batch
        for i, run_sim in enumerate(msim.sims):
            try:
                temp_df = run_sim.to_df(date_index=True, by_community=EXPORT_BY_COMMUNITY)
            except Exception as e:
                print(f'Could not save run results to df: {e}')
                continue
            temp_df['Seed'] = seed + i
            # Header on the first write only, so the CSV is directly readable with
            # pd.read_csv(..., index_col=0) and needs no cleaning pass
            temp_df.to_csv(allruns_path, mode='a', index=True, header=not header_written)
            header_written = True
            ncomm = len([c for c in temp_df.columns if 'by_community' in c])
            print(f'Seed:{seed + i} is done ({len(temp_df.columns)} columns, {ncomm} by-community)')

    print(f'Done. Wrote {allruns_path}')


if __name__ == '__main__':
    main()

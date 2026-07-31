import pathlib
import sciris as sc
import hpvsim_working as hpv
import NHS_2025_lambdamu
from basePars_community import base_pars_geno

# -------------------------------------------------------------------
# Sweep the community network's gamma_shape (partner-propensity
# heterogeneity) while holding every other base_pars_community.py
# setting constant. Same run_sim.py logic/CSV format, one MultiSim at
# a time (no overlapping runs), one output CSV per shape value.
# -------------------------------------------------------------------

SIM_LABEL = 'community network'
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer'

N_RUNS = 5  # due to multisim stuff I think 5 is max I can run on a 6 core cpu
seeds = [0]  # 3 seeds * 5 runs = 15 sims per shape value

# 5 steps from 0.05 to 5, denser at the low end since CV = 1/sqrt(shape)
# changes fastest there (see gamma-shape discussion earlier in session).
GAMMA_SHAPES = [0.05, 0.25, 1.0, 2.5, 5.0]


def shape_tag(shape):
    return str(shape).replace('.', 'p')


def run_for_shape(shape, outdir):
    base_pars_geno['community_pars']['gamma_shape'] = shape
    allruns_path = outdir / f'gammashape_{shape_tag(shape)}.csv'
    print(f'=== gamma_shape = {shape} -> {allruns_path.name} ===')

    for seed in seeds:
        base_pars_geno['rand_seed'] = seed
        # Build simulation
        sim = hpv.Sim(base_pars_geno, label=SIM_LABEL)
        print('Created HPVsim simulation.')
        # Run MultiSim
        print(f'Running MultiSim with n_runs = {N_RUNS}  ...')
        msim = hpv.MultiSim(sim)
        msim.run(n_runs=N_RUNS, n_cpus=N_RUNS)
        print('MultiSim run complete.')

        i = 0
        for sim in msim.sims:
            try:
                temp_df = sim.to_df(date_index=True)
                print('sim made into df')
            except Exception as e:
                print(f'Could not save run results to df: {e}')
                i += 1
                continue
            temp_df['Seed'] = seed + i
            temp_df.to_csv(allruns_path, mode='a', index=True)
            print(f'Seed:{seed + i} is done')
            i += 1


def main():
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Outputs will be saved to: {outdir.resolve()}')

    for shape in GAMMA_SHAPES:
        run_for_shape(shape, outdir)

    print('Sweep done.')


if __name__ == '__main__':
    main()

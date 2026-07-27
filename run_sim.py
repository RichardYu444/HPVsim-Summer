import pathlib
import sciris as sc
import matplotlib.pyplot as plt
import hpvsim_working as hpv
import NHS_2025_lambdamu
import pickle
# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

NETWORK = 'community'  # 'default' (HPVsim's built-in network, basePars.py) or 'community' (basePars_community.py)

if NETWORK == 'community':
    from basePars_community import base_pars_geno
    SIM_LABEL = 'community network'
    ALLRUNS = 'test_community.csv'
else:
    from basePars import base_pars_geno
    SIM_LABEL = 'default network'
    ALLRUNS = 'default.csv'  #IMPORTANT TO CHANGE EVERYTIME (maybe?)

OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer'

N_RUNS = 5 #due to multisim stuff I think 5 is max I can run on a 6 core cpu

seeds = [0]#, #6, 12, 18, 24, 30, 36, 42, 48, 54] #10 seeds gets us to 5 * 10 = 50 total runs (in theory)

def main():
    #Ensure output directory exists
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f'Outputs will be saved to: {outdir.resolve()}')

    #run through seeds to run sim, 5 at a time, hence the gaps in seeds
    for seed in seeds:
        base_pars_geno['rand_seed'] = seed
        #Build simulation
        sim = hpv.Sim(base_pars_geno, label=SIM_LABEL)
        print('Created HPVsim simulation.')
        #Run MultiSim
        print(f'Running MultiSim with n_runs = {N_RUNS}  ...')
        msim = hpv.MultiSim(sim)
        msim.run(n_runs = N_RUNS, n_cpus = 5) 
        print('MultiSim run complete.')

        #We keep the important indicators in line with what is detectable irl

        skipped_pars = ['genotype_map', 'vaccine map'] #these are skipped as their datatype doesn't convert to df
        # available = []
        allruns_path = outdir / ALLRUNS
        #i helps keep track of which seed we are on, but is onlly for debugging purposes really
        i = 0
        #run through each sim done and 
        for sim in msim.sims: 


            try:
                temp_df = sim.to_df(date_index = True)
                print('sim made into df')
            except Exception as e:
                print(f'Could not save run results to df: {e}')
            temp_df['Seed'] = seed + i
            temp_df.to_csv(ALLRUNS, mode = 'a', index = True)
            print(f'Seed:{seed + i} is done')
            i += 1


    print('Done.')


if __name__ == '__main__':
    main()

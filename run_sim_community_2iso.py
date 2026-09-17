"""
Two completely isolated communities with different vaccination coverage.

Same shape as run_sim_community_50.py -- MultiSim in batches of 5, by-community results on,
one CSV with a single header row -- and the same network and epidemic parameters
(basePars_community: 200k agents, gamma_shape 2, 1.5 partners/yr, the Natsal durations, the
casual age-mixing kernel, NHS_2025_lambdamu screening with l=m=1), with two changes:

1. The community structure. The four ethnicity communities are replaced by two:
       'Large'   COMMUNITY_PROBS[0] = 80% of network members
       'Small'   COMMUNITY_PROBS[1] = 20%
   and community_mixing is diagonal, so no partnership can ever form between them -- both
   _sample_edges() and the force-pairing step allocate across blocks in proportion to W, and
   W = kron(C, A) is zero on every cross-community block. The diagonal is 1/p_c, built the same
   way basePars_community builds its matrix (joint / outer(p, p), with the joint here being
   diag(p)), which gives both communities the same mean degree: the sampler's total edge count
   is fixed by the propensity sums and C only splits it, so each community gets edges in
   proportion to its size.

2. Vaccination coverage by community. Every first-dose offer in NHS_Vacc.py's programme -- all
   four eras, 2008 onwards -- is accepted with ANNUAL probability VACC_COVERAGE[c] for an agent
   in community c (90% Large, 30% Small), in place of the year-by-year national figure. Second
   and third doses keep NHS_Vacc.py's conditional continuation probabilities (so from 2023,
   when the programme is one dose, coverage is exactly these numbers). Screening is unchanged
   and identical in both communities.

   The acceptance step is NHS_ethnicity_uptake's: it clones NHS_Vacc.py's interventions,
   intercepts only the draw, and applies the coverage on the annual scale before converting to
   per-timestep (see that file's _per_agent_probs()). Importing it also installs its patch that
   makes an agent's community readable from birth -- vaccination is at 12-13, but the network
   only tags a community at sexual debut (~16) -- which is why the import is at module level:
   sciris spawns workers on Windows that re-import this file, and they need the patch too.

   That patch changes the network realisation relative to an unpatched run, so these runs are
   not seed-for-seed comparable with community_gamma2_*.csv (they were never going to be with a
   different community structure anyway).

Output: OUTPUT_DIR/ALLRUNS, columns
    year (index), t, <all the existing 1-D results>, <by-community results>, Seed
with by-community columns named <result>_by_community_Large / _Small.
"""
import pathlib
import numpy as np
import hpvsim_working as hpv
import NHS_2025_lambdamu
import NHS_Vacc
import NHS_ethnicity_uptake as eth   # MODULE LEVEL: installs the pre-debut community patch
from basePars_community import base_pars_geno

# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

SIM_LABEL = 'two isolated communities (80/20), vaccination 90%/30%'
ALLRUNS = 'community_2iso_80-20_vacc90-30_2070_5runs.csv'  # IMPORTANT TO CHANGE EVERY TIME
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

N_RUNS = 5  # due to multisim stuff I think 5 is max I can run on a 6 core cpu
N_CPUS = 5

seeds = [0]  # 1 seed gets us to 5 * 1 = 5 total runs (0-4)

END_YEAR = 2070

# Community order is index order: community 0 is COMMUNITY_LABELS[0] and so on
COMMUNITY_LABELS = ['Large', 'Small']
COMMUNITY_PROBS = np.array([0.8, 0.2])
VACC_COVERAGE = np.array([0.9, 0.3])  # annual first-dose acceptance probability, per community

# Same by-community results as run_sim_community_50.py, plus n_vaccinated so the coverage
# split can be checked in the output
COMMUNITY_RESULTS = ['infections', 'cancers', 'hpv_prevalence', 'cancer_incidence',
                     'mean_degree', 'within_edge_frac', 'single_frac', 'n_vaccinated']
EXPORT_BY_COMMUNITY = COMMUNITY_RESULTS

DROP_NETWORK_HISTORY = True


class CommunityCoverage:
    '''
    Stands in for NHS_ethnicity_uptake.EthnicityUptake: same interface (name, is_null, probs()),
    but returns an absolute coverage per community rather than a multiplier on the national
    probability, and works for any number of communities (EthnicityUptake is tied to the four
    ethnicities).
    '''
    name = 'vaccination'
    is_null = False

    def __init__(self, coverage):
        self.coverage = np.asarray(coverage, dtype=float)
        return

    def probs(self, sim, inds, base):
        inds = np.asarray(inds, dtype=np.int64)
        comm = np.asarray(sim.people.community[inds], dtype=np.int64)
        unknown = comm < 0  # not yet debuted -- nearly everyone at 12-13
        if unknown.any():
            comm[unknown] = eth.ethnicity_for(inds[unknown], eth._seed_of(sim),
                                              eth._community_probs_of(sim))
        return self.coverage[comm]


VACC_UPTAKE = CommunityCoverage(VACC_COVERAGE)


def community_pars():
    probs = COMMUNITY_PROBS / COMMUNITY_PROBS.sum()
    joint = np.diag(probs)  # no cross-community partnership ends at all
    pars = dict(base_pars_geno['community_pars'])
    pars.update(n_communities=len(probs),
                community_probs=probs,
                community_mixing=joint / np.outer(probs, probs),
                community_labels=COMMUNITY_LABELS)
    return pars


def get_vaccinations():
    ''' NHS_Vacc.vaccinations, with first doses accepted at VACC_COVERAGE by community. '''
    out = []
    for src in NHS_Vacc.vaccinations:
        first_dose = src.eligibility is NHS_Vacc.eligible_first_dose
        out.append(eth._clone_vx(src, VACC_UPTAKE if first_dose else None))
    eth._check_imm_source_ordering(out)
    return out


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

    base_pars_geno['community_pars'] = community_pars()
    base_pars_geno['community_results'] = COMMUNITY_RESULTS
    base_pars_geno['interventions'] = NHS_2025_lambdamu.get_interventions(l=1, m=1) + get_vaccinations()
    base_pars_geno['end'] = END_YEAR
    if DROP_NETWORK_HISTORY:
        base_pars_geno['analyzers'] = []

    cp = base_pars_geno['community_pars']
    print(f"gamma_shape: {cp['gamma_shape']}")
    print(f"Years: {base_pars_geno['start']}-{base_pars_geno['end']}")
    print(f"Communities: {list(cp['community_labels'])}  shares {cp['community_probs'].tolist()}")
    print(f"community_mixing:\n{cp['community_mixing']}")
    print(f"First-dose vaccination coverage: {dict(zip(COMMUNITY_LABELS, VACC_COVERAGE.tolist()))}")
    print(f"Interventions: {len(base_pars_geno['interventions'])}")

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
            temp_df.to_csv(allruns_path, mode='a', index=True, header=not header_written)
            header_written = True
            ncomm = len([c for c in temp_df.columns if 'by_community' in c])
            print(f'Seed:{seed + i} is done ({len(temp_df.columns)} columns, {ncomm} by-community)')

    print(f'Done. Wrote {allruns_path}')


if __name__ == '__main__':
    main()

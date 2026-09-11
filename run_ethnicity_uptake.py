"""
run_ethnicity_uptake.py
=======================

Two arms of the community-network sim, differing only in who takes up what:

    'equal'     every ethnicity accepts at the national rate (all multipliers 1.0). This is the
                CONTROL, and it has to come from NHS_ethnicity_uptake too rather than from
                NHS_2025_lambdamu + NHS_Vacc directly -- importing that module patches the network
                so ethnicity is known before sexual debut, which changes the network realisation.
                With the multipliers at 1 the two are otherwise identical (its
                check_null_equivalence() runs both and compares).

    'observed'  the uptake supplied for this study, applied as ratios to the White group:
                    vaccination   White 74.1  Black 70.1  Asian 63.8  Other 66.4  (%)
                    screening     White 74.4  Black 49.2  Asian 54.9  Other 36.0  (%)
                See NHS_ethnicity_uptake.multipliers_from_uptake() for why these become ratios
                rather than absolute acceptance probabilities.

Run with no argument for both arms, or name one to run it alone:

    python -u run_ethnicity_uptake.py            # both arms
    python -u run_ethnicity_uptake.py observed   # one arm
    N_BATCHES=20 python -u run_ethnicity_uptake.py    # extend to 100 runs per arm

Defaults are 200,000 agents to 2070, in batches of 5 runs, 12 batches (60 runs) per arm. Batches
are appended to the CSV as they finish and already-present seeds are skipped, so raising
N_BATCHES and re-running continues from where the last invocation stopped rather than repeating
it. Batches alternate between the arms, so both always hold the same seeds and a run stopped
part-way still gives a balanced comparison.


CENSUS SHARES (the change from the first 200k pass)
---------------------------------------------------
basePars_community.py tags agents from PARTNERSHIP-END shares (91.2 / 4.5 / 3.6 / 0.7), which
makes the Asian community about half its census size. Here that is switched to the England &
Wales census composition (86.0 / 9.1 / 4.2 / 0.7, renormalised over the four groups), because at
partnership-end shares the minority communities were too small to resolve anything: over 10 runs
the smallest detectable change in Asian cancers was 7.2%, and in the fourth community 20%.

This is NOT purely a variance reduction, and the trade-off is worth stating plainly.
basePars_community's own comment explains why: the gap between the two share sets IS the survey's
implied degree gradient. With partnership-end shares the IPF step is close to a no-op and every
community ends up with the SAME mean degree, so the gradient is expressed as community size. Move
to census shares and the same gradient reappears as mean DEGREE instead -- Asian mean degree
comes out around 0.49x White. So the minority communities roughly double in size (less noise) but
also become less sexually active (lower incidence, hence fewer cancer events per head). The two
effects push the event count in opposite directions, so the net gain in precision is an empirical
question -- analyse_ethnicity_uptake.py reports the realised MDE, which is where to read it off
rather than assume it.

Both arms use the identical network parameterisation, so the between-arm comparison is unaffected
either way; what changes is the baseline epidemiology of each community, and how precisely a
given number of runs can measure a difference within it.

community_mixing is rebuilt for the new margins with basePars_community's own
_rescale_to_margins() (biproportional/IPF), which preserves every odds ratio -- i.e. the whole
assortativity structure measured from Natsal -- while making both margins equal the census
shares. Reusing that function rather than reimplementing it is deliberate: it is the piece that
defines what "the same mixing, different margins" means.


Output, in the house format (as run_sim_community_50.py, and readable by plot_IQR.py):

    csvs/ethnicity_<arm>_<tag>_<n>k.csv
        year (index), t, <all the 1-D results>, <the by-community results>, Seed

one row per (run, year), all runs of that arm stacked under a single header row. The
by-community columns are named <result>_by_community_<label>, e.g.
hpv_prevalence_by_community_White.

Results are trimmed on purpose -- stratifying by community multiplies the stored results by four,
so pars['community_results'] selects only the stems in COMMUNITY_RESULTS below: prevalence,
infections, infectious stock, vaccinated stock, cancers, cancer stock, plus cancer incidence and
cancer deaths. No CIN, no per-genotype anything. HPVsim widens the selection automatically to
whatever a rate needs as a denominator (n_alive, n_females_alive), so those appear too.

One caveat on what "by community" counts. HPVsim's by-community results cover NETWORK MEMBERS
only -- people.community is -1 until an agent's sexual debut, and People.bin_communities() drops
anyone below zero. Agents are vaccinated at 12-13 and debut at ~16, so a freshly vaccinated
cohort does not show up in n_vaccinated_by_community for about three years, and the by-community
denominators do not sum to the whole population. This affects both arms identically and does not
touch the uptake logic itself (that reads ethnicity through NHS_ethnicity_uptake.ethnicity_of(),
which knows an agent's ethnicity from birth), but it does mean n_vaccinated_by_community is a
count of vaccinated ADULTS, not of vaccinated children.

Then:
    python analyse_ethnicity_uptake.py     # is the difference bigger than the run-to-run spread?
    python plot_ethnicity_uptake.py        # both arms overlaid, per community
"""

import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sciris as sc

import hpvsim_working as hpv
import NHS_ethnicity_uptake as eth   # MODULE LEVEL: installs the pre-debut ethnicity patch
import basePars_community


# -------------------------------------------------------------------
# settings
# -------------------------------------------------------------------

ARMS = ['equal', 'observed']
WHICH = sys.argv[1:] or ARMS

N_AGENTS = int(os.environ.get('N_AGENTS', 200_000))
END = int(os.environ.get('END', 2070))

# England & Wales census composition over the four groups, renormalised (the figure quoted in
# basePars_community.py's comment above eth_partner_end_shares). Set CENSUS_SHARES=0 to fall back
# to that file's partnership-end shares.
USE_CENSUS_SHARES = os.environ.get('CENSUS_SHARES', '1') != '0'
CENSUS_SHARES = np.array([0.860, 0.091, 0.042, 0.007])
CENSUS_SHARES = CENSUS_SHARES / CENSUS_SHARES.sum()

# Runs are done in batches of RUNS_PER_BATCH, and each batch is appended to the CSV as it
# finishes -- so a run cut short by the machine sleeping or a session ending is continued simply
# by invoking the same command again (RESUME below skips whatever is already on disk). Same
# shape as run_core_vacc.py, and for the same reason: at 200k agents this takes hours.
RUNS_PER_BATCH = int(os.environ.get('RUNS_PER_BATCH', 5))
N_CPUS = int(os.environ.get('N_CPUS', RUNS_PER_BATCH))
N_BATCHES = int(os.environ.get('N_BATCHES', 12))          # 12 x 5 = 60 runs per arm
SEEDS = list(range(0, N_BATCHES * RUNS_PER_BATCH, RUNS_PER_BATCH))

# Skip seeds already present in the output rather than refusing to start or duplicating them.
RESUME = os.environ.get('RESUME', '1') != '0'

TAG = os.environ.get('TAG', 'census' if USE_CENSUS_SHARES else 'partnership')
OUTPUT_DIR = pathlib.Path(__file__).with_name('csvs')

# Which results to stratify by community. See the module docstring.
COMMUNITY_RESULTS = [
    'hpv_prevalence',    # pulls in n_infectious, n_alive
    'infections',
    'n_infectious',      # the "n_infected" stock; HPVsim's community stem is n_infectious
    'n_vaccinated',
    'cancers',
    'n_cancerous',
    'cancer_incidence',  # pulls in n_females_alive
    'cancer_deaths',
]
# True exports everything that was stored, i.e. the above plus the denominators pulled in with
# them -- which analyse_ethnicity_uptake.py needs for the per-100k figures.
EXPORT_BY_COMMUNITY = True


def community_pars():
    '''
    basePars_community's community_pars, with the margins swapped for census shares and the
    mixing matrix rebuilt for them. See the module docstring for what that does and does not do.
    '''
    pars = dict(basePars_community.community_pars)
    if not USE_CENSUS_SHARES:
        return pars
    probs = CENSUS_SHARES
    mixing = (basePars_community._rescale_to_margins(basePars_community.eth_joint, probs)
              / np.outer(probs, probs))
    pars.update(community_probs=probs, community_mixing=mixing)
    return pars


def apply_arm(arm):
    ''' Set the multipliers for this arm. Returns the two dicts actually in force. '''
    if arm == 'equal':
        eth.set_multipliers(vaccination=[1.0] * eth.N_ETH, screening=[1.0] * eth.N_ETH)
    elif arm == 'observed':
        eth.use_observed()
    else:
        raise ValueError(f'arm must be one of {ARMS}, got {arm!r}')
    return eth.VACC_UPTAKE.as_dict(), eth.SCREEN_UPTAKE.as_dict()


def make_sim(arm, seed=0):
    apply_arm(arm)
    pars = sc.mergedicts(basePars_community.base_pars_geno, dict(
        n_agents=N_AGENTS,
        end=END,
        rand_seed=seed,
        verbose=0,                    # 0, not -1: brief() prints a character cp1252 can't encode
        analyzers=[],                 # network_history is pure memory cost here
        community_pars=community_pars(),
        community_results=COMMUNITY_RESULTS,
        interventions=eth.get_interventions(l=1, m=1),
    ))
    return hpv.Sim(pars, label=f'{arm} uptake')


def csv_path(arm):
    return OUTPUT_DIR / f'ethnicity_{arm}_{TAG}_{round(N_AGENTS / 1000)}k.csv'


def done_seeds(arm):
    ''' Seeds already in an output file, so a re-invocation continues instead of duplicating. '''
    path = csv_path(arm)
    if not RESUME or not path.exists():
        return set()
    return {int(x) for x in pd.read_csv(path, usecols=['Seed'])['Seed'].unique()}


def run_batch(arm, seed):
    ''' One batch of RUNS_PER_BATCH runs of one arm, appended to that arm's CSV. '''
    path = csv_path(arm)
    batch = [seed + i for i in range(RUNS_PER_BATCH)]
    already = done_seeds(arm)
    if all(s in already for s in batch):
        print(f'  [{arm}] seeds {batch[0]}-{batch[-1]} already on disk, skipping', flush=True)
        return

    print(f'  [{arm}] seeds {batch[0]}-{batch[-1]} ...', flush=True)
    msim = hpv.MultiSim(make_sim(arm, seed=seed))
    msim.run(n_runs=RUNS_PER_BATCH, n_cpus=N_CPUS, keep_people=False)

    written = 0
    for i, run_sim in enumerate(msim.sims):
        this_seed = batch[i]
        if this_seed in already:  # partially-written batch: keep what is there
            continue
        df = run_sim.to_df(date_index=True, by_community=EXPORT_BY_COMMUNITY)
        df['Seed'] = this_seed
        # Header on the first write only, so the CSV reads back with pd.read_csv(index_col=0)
        # and needs no cleaning pass
        df.to_csv(path, mode='a', index=True, header=not path.exists())
        written += 1
        ncomm = len([c for c in df.columns if 'by_community' in c])
    print(f'  [{arm}] wrote {written} run(s), seeds {batch[0]}-{batch[-1]}  '
          f'({len(df.columns)} columns, {ncomm} by-community)', flush=True)
    return


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probs = community_pars()['community_probs']
    print(f'Agents: {N_AGENTS:,}   years: {basePars_community.start}-{END}   arms: {WHICH}   '
          f'runs per arm: up to {len(SEEDS) * RUNS_PER_BATCH} '
          f'({len(SEEDS)} batches of {RUNS_PER_BATCH})')
    print(f'Community shares: {"CENSUS" if USE_CENSUS_SHARES else "partnership-end"} '
          f'{np.round(probs, 4).tolist()}')
    print(f'  agents per community: '
          f'{ {e: int(round(f * N_AGENTS)) for e, f in zip(eth.ETHNICITIES, probs)} }')
    print(f'By-community results stored: {COMMUNITY_RESULTS}')
    for arm in WHICH:
        vacc, screen = apply_arm(arm)
        print(f'  {arm:<9} vaccination { {k: round(v, 3) for k, v in vacc.items()} }  '
              f'screening { {k: round(v, 3) for k, v in screen.items()} }')
    for arm in WHICH:
        print(f'  -> {csv_path(arm)}')
    print(flush=True)

    # Batches outer, arms inner, so both arms always hold the SAME seeds. If the run is stopped
    # part-way the comparison is still balanced, rather than one arm being 60 runs deep and the
    # other not started.
    for seed in SEEDS:
        for arm in WHICH:
            run_batch(arm, seed)
    print('\nDone. Next:\n  python analyse_ethnicity_uptake.py\n  python plot_ethnicity_uptake.py')
    return


if __name__ == '__main__':
    main()

"""
run_gamma2_ethnicity_1900.py
============================

The community sibling of run_gamma2_nocomm_1900.py: same window, same agent count, same export,
same batching -- but WITH the four ethnic communities in place, and with ETHNICITY-DIFFERENTIAL
uptake of vaccination and screening.

    run_gamma2_nocomm_1900.py      n_communities = 1, national uptake       (no communities)
    run_gamma2_ethnicity_1900.py   n_communities = 4, uptake by ethnicity   (this file)

Gamma partner-propensity at shape 2, 100,000 agents, 1900-2050, current init_hpv_prev, 50 runs
per arm, with the same wide per-genotype export -- plus the by-community columns, which are the
point of this one.

TWO ARMS
--------
Ethnicity-differential uptake is only interpretable against a control that is identical in every
other respect, so this runs two arms on the SAME seeds:

    'equal'     every ethnicity accepts at the national rate (all multipliers 1.0). The control.
                It still comes from NHS_ethnicity_uptake rather than NHS_2025_lambdamu + NHS_Vacc
                directly, because importing that module patches the network so ethnicity is known
                before sexual debut, which changes the network realisation -- so the control has
                to be patched too or the two arms would differ by more than uptake. With the
                multipliers at 1 the two are otherwise identical (eth.check_null_equivalence()
                runs both and compares).

    'observed'  the uptake supplied for this study, applied as ratios to the White group:
                    vaccination   White 74.1  Black 70.1  Asian 63.8  Other 66.4  (%)
                    screening     White 74.4  Black 49.2  Asian 54.9  Other 36.0  (%)
                See NHS_ethnicity_uptake.multipliers_from_uptake() for why these become ratios to
                White rather than absolute acceptance probabilities: only the GRADIENT comes from
                the data, the national level stays where the calibration put it.

Batches alternate between arms (batches outer, arms inner), exactly as run_ethnicity_uptake.py
does it, so both arms always hold the same seeds. A run stopped part-way still gives a balanced
comparison rather than one arm 50 deep and the other not started.

    python run_gamma2_ethnicity_1900.py                 # both arms
    python run_gamma2_ethnicity_1900.py observed        # one arm only
    N_BATCHES=4 python run_gamma2_ethnicity_1900.py     # 20 runs per arm, extendable later

CENSUS SHARES, NOT PARTNERSHIP-END SHARES
-----------------------------------------
basePars_community.py tags agents from PARTNERSHIP-END shares (91.2 / 4.5 / 3.6 / 0.7), which
makes the Asian community about half its census size. This file follows run_ethnicity_uptake.py
in switching to the England & Wales census composition (86.0 / 9.1 / 4.2 / 0.7, renormalised),
because at partnership-end shares the minority communities are too small to resolve anything --
over 10 runs the smallest detectable change in Asian cancers was 7.2%, and in the fourth
community 20%.

run_ethnicity_uptake.py and basePars_community.py both warn that this swap costs something: that
the gap between the two share sets IS the survey's implied degree gradient, and that moving to
census shares makes it reappear as mean DEGREE (Asian ~0.49x White) rather than as community
size. THAT WARNING IS WRONG, and this run's --diagnose is what shows it.

The reason is what IPF does. _rescale_to_margins() makes both margins of the joint equal the
target shares, so for every community c the row sums to p_c, and the relative degree implied by
the mixing kernel is

    sum_j p_j * C[c, j]  =  (sum_j J[c, j]) / p_c  =  p_c / p_c  =  1

identically, for EVERY community and under BOTH share sets -- verifiable in three lines of numpy,
and exactly 1.0000 for all four groups either way. The mixing kernel decides who you partner
with, not how many partners you have; mean degree is set by mean_partners_per_year and theta,
which carry no community index. --diagnose measures it in the real sim and finds White 0.874,
Asian 0.850, Black 0.861, Chinese 0.830 -- within ~5%, i.e. flat to within age structure and
sampling noise, not a 2x gradient.

So census shares are a straight improvement here: the minority communities roughly double in size
(less noise, smaller detectable effects) at no cost in per-head sexual activity. Both arms use
the identical network parameterisation, so the between-arm comparison is unaffected either way.
Set CENSUS_SHARES=0 to fall back to basePars_community's partnership-end shares.

community_mixing is rebuilt for the new margins with basePars_community's own
_rescale_to_margins() (biproportional/IPF), which preserves every odds ratio -- the whole
assortativity structure measured from Natsal -- while making both margins equal the census
shares. Reusing that function rather than reimplementing it is deliberate: it is the piece that
defines what "same mixing, different margins" means.

STARTING IN 1900 NEEDS TWO ADJUSTMENTS
--------------------------------------
Identical to run_gamma2_nocomm_1900.py, and for the same reasons -- UK demographic data start in
1950, so make_pars() supplies pars['age_datafile'] (the population starts on 1950's age
structure, since the distribution is looked up for the exact start year) and sets
pars['use_migration'] = False (People.check_migration() raises NotImplementedError outright when
the sim starts before the data). Population size is then set by births and deaths alone and
drifts upward: ~100k agents at 1900 becomes ~300k by 2050, which is where the memory goes.

Both are applied only when start < 1950, and an explicit override of either wins, so shrinking
the window for --selftest/--diagnose does not silently change them.

WHAT GETS EXPORTED
------------------
Everything run_gamma2_nocomm_1900.py exports -- every 1-D result, every '*_by_genotype' result
flattened one column per genotype, plus infections_star / prevalence_star -- AND the by-community
columns, named <result>_by_community_<label> (e.g. hpv_prevalence_by_community_White).

The by-community selection is run_ethnicity_uptake.py's, so the output stays readable by
analyse_ethnicity_uptake.py and plot_ethnicity_uptake.py. HPVsim widens the selection
automatically to whatever a rate needs as a denominator (n_alive, n_females_alive), so those
appear too.

One caveat on what "by community" counts, carried over from run_ethnicity_uptake.py: HPVsim's
by-community results cover NETWORK MEMBERS only -- people.community is -1 until an agent's sexual
debut, and People.bin_communities() drops anyone below zero. Agents are vaccinated at 12-13 and
debut at ~16, so a freshly vaccinated cohort does not appear in n_vaccinated_by_community for
about three years, and the by-community denominators do not sum to the whole population. This
affects both arms identically and does not touch the uptake logic itself -- that reads ethnicity
through NHS_ethnicity_uptake.ethnicity_of(), which knows an agent's ethnicity from birth -- but
n_vaccinated_by_community is a count of vaccinated ADULTS, not of vaccinated children.

WHY NOTHING IS EDITED OUTSIDE THIS FILE
---------------------------------------
basePars_community.py is imported by run_sim_community_50.py, validate_r0.py, network.py and the
NHS_* analyses and must keep its current behaviour, so every change is layered onto a deep copy
at run time.

Imports stay at MODULE level, and NHS_ethnicity_uptake especially so: importing it installs the
pre-debut ethnicity patch, and sciris parallelize spawns workers on Windows that re-import
__main__. An import inside main() would leave workers unpatched. basePars_community_powerlaw and
powerlaw are deliberately NOT imported anywhere in this file or its import graph -- importing
either installs the Pareto theta sampler process-wide, replacing the Gamma propensity.

The arm's multipliers are applied by apply_arm() before get_interventions() builds the
intervention list, so the uptake objects travel with the pickled Sim into each worker. --selftest
verifies that round trip end-to-end rather than assuming it: every non-White multiplier is <= 1,
so the observed arm must come out with fewer doses and fewer screens than the equal-uptake
control. If the multipliers ever failed to survive process spawn, both arms would run at the
national rate and those two checks would fail.

THREE MODES
-----------
    python run_gamma2_ethnicity_1900.py --selftest    ~10 min   pipeline + column + arm checks
    python run_gamma2_ethnicity_1900.py --diagnose    ~10 min   realised network + communities
    python run_gamma2_ethnicity_1900.py               ~11 h     50 runs x 2 arms

Run --selftest and --diagnose first: --diagnose reports the realised mean degree per community
and the within-community edge fraction, which is worth reading before spending hours.

Known, pre-existing and not introduced here: NHS_Vacc's vx_1921_d2 carries a probability above 1
(0.64*2/(0.60+0.53) = 1.13 -- flagged in NHS_Vacc.py's own TODO), so interventions.py's
annual_prob conversion prints "RuntimeWarning: invalid value encountered in power".

Also worth knowing before comparing against the default-network or power-law runs:
basePars_community.py has its calibration block commented out, so beta, f/m_cross_layer and the
genotype progression pars (cin_fn.k, dur_cin.par1, rel_beta) are HPVsim package defaults rather
than this project's calibrated values. Cancer outcomes here are comparable to the other gamma
runs, not to basePars.py's.

If stdout is redirected to a file, set PYTHONIOENCODING=utf-8.

Interruptible: batches are appended to each arm's CSV as they finish, and RESUME makes a
re-invocation skip whatever is already in the file.

Outputs into OUTPUT_DIR:

    <TAG>_<arm>.csv                 50 runs stacked per arm; year index, t, every 1-D result,
                                    every by-genotype result, infections_star, prevalence_star,
                                    the by-community columns, Seed
    <TAG>_network_diagnostic.txt    --diagnose only
    <TAG>_<arm>_selftest.csv        --selftest only
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sciris as sc

import basePars                      # only to check init_hpv_prev has not drifted (--selftest)
import basePars_community as bpc     # the Gamma community network; NEVER the powerlaw sibling
import NHS_ethnicity_uptake as eth   # MODULE LEVEL: installs the pre-debut ethnicity patch
import hpvsim_working as hpv


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

ARMS = ['equal', 'observed']
WHICH = [a for a in sys.argv[1:] if not a.startswith('--')] or ARMS

# Environment-set rather than argv-derived so spawned workers inherit them (see run_core_vacc.py)
GAMMA_SHAPE = float(os.environ.get('GAMMA_SHAPE', 2.0))  # basePars_community's own value

N_AGENTS = int(os.environ.get('N_AGENTS', 100_000))
START = int(os.environ.get('START', 1900))
END = int(os.environ.get('END', 2050))

# England & Wales census composition over the four groups, renormalised. Set CENSUS_SHARES=0 to
# fall back to basePars_community's partnership-end shares. See the module docstring.
USE_CENSUS_SHARES = os.environ.get('CENSUS_SHARES', '1') != '0'
CENSUS_SHARES = np.array([0.860, 0.091, 0.042, 0.007])
CENSUS_SHARES = CENSUS_SHARES / CENSUS_SHARES.sum()

TAG = os.environ.get('TAG', 'gamma2_eth_100k_1900_2050_50runs')
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

# First year of the UK demographic data. A start before this needs the two adjustments in
# make_pars() -- see the module docstring.
DATA_START_YEAR = 1950
AGE_DATAFILE = (pathlib.Path(__file__).parent / 'csvs' / 'equilibrium_1900'
                / 'uk_age_distribution_1950.csv')

# Seeds per batch, and how many of them run at once. N_CPUS defaults to 3 for the reason measured
# on run_gamma2_nocomm_1900.py: a 1900 start forces use_migration=False, the population grows on
# births and deaths alone (~100k -> ~300k by 2050), and a worker peaks near 1.0 GB rather than the
# ~0.5 GB a 1950-start run uses. On this machine (15.4 GB total, ~10 GB already in use) five
# workers peak at 5.5 GB with 0.9 GB free and run at ~49% CPU efficiency -- half the wall time is
# paging -- while three fit in 2.9 GB at ~100%. Throughput is roughly a wash; 3 is the default
# because it leaves the machine usable and keeps clear of an OOM.
#
#     N_CPUS=5 python run_gamma2_ethnicity_1900.py
N_RUNS = int(os.environ.get('N_RUNS', 5))
N_CPUS = int(os.environ.get('N_CPUS', 3))

SEEDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 seeds gets us to 5 * 10 = 50 runs (0-49)/arm

# Number of leading seed batches to run, N_RUNS each, PER ARM. 10 (the default) is the full 50.
# Because RESUME skips whatever is already in the CSV, a short run can be extended later just by
# re-running with a larger value -- only the missing batches are computed.
#
#     N_BATCHES=2 python run_gamma2_ethnicity_1900.py
SEEDS = SEEDS[:int(os.environ.get('N_BATCHES', len(SEEDS)))]

RESUME = os.environ.get('RESUME', '1') != '0'

# Which results to stratify by community -- run_ethnicity_uptake.py's selection, so the output
# stays readable by analyse_ethnicity_uptake.py. HPVsim widens this to include whatever each rate
# needs as a denominator (n_alive, n_females_alive), so those are stored too.
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

# --diagnose settings. The epidemic is irrelevant to network structure, so this stops well before
# 2050; interventions are dropped because screening's 1980 start would then fall outside the sim.
DIAG_SEED = int(os.environ.get('DIAG_SEED', 0))
DIAG_AGENTS = int(os.environ.get('DIAG_AGENTS', N_AGENTS))
DIAG_END = int(os.environ.get('DIAG_END', 1930))

# --selftest settings: small and short, but through exactly the same code path as the real run.
# SELFTEST_END has to clear 2008 for the vaccination and arm checks to mean anything.
SELFTEST_AGENTS = int(os.environ.get('SELFTEST_AGENTS', 20_000))
SELFTEST_END = int(os.environ.get('SELFTEST_END', 2030))
SELFTEST_RUNS = int(os.environ.get('SELFTEST_RUNS', 2))

LKEY_SHORT, LKEY_LONG = 's', 'l'  # community network's layer keys (<-> default's casual/marital)

N_GENOTYPE_RESULTS = 21  # how many '*_by_genotype' results HPVsim stores; checked by --selftest


# -------------------------------------------------------------------
# parameters
# -------------------------------------------------------------------

def community_pars(gamma_shape=None):
    """
    basePars_community's community_pars with gamma_shape made explicit and, by default, the
    margins swapped for census shares with the mixing matrix rebuilt for them.

    The four communities are KEPT -- that is the difference from run_gamma2_nocomm_1900.py, which
    collapses them to one. Deep-copied so nothing here can reach back into the module-level dict
    that run_sim_community_50.py, validate_r0.py and the NHS_* analyses read.
    """
    pars = sc.dcp(sc.mergedicts(bpc.community_pars, dict(
        gamma_shape=float(GAMMA_SHAPE if gamma_shape is None else gamma_shape),
    )))
    if USE_CENSUS_SHARES:
        probs = CENSUS_SHARES
        # basePars_community's own IPF, so "same mixing, different margins" means what that file
        # means by it -- every odds ratio preserved, both margins set to the census shares
        mixing = (bpc._rescale_to_margins(bpc.eth_joint, probs) / np.outer(probs, probs))
        pars.update(community_probs=probs, community_mixing=mixing)
    return pars


def apply_arm(arm):
    """
    Set the uptake multipliers for this arm. Returns the two dicts actually in force.

    Called before get_interventions() builds the list, so the uptake objects the interventions
    hold references to travel with the pickled Sim into every spawned worker -- which is what
    makes this survive process spawn on Windows. --selftest verifies that round trip.
    """
    if arm == 'equal':
        eth.set_multipliers(vaccination=[1.0] * eth.N_ETH, screening=[1.0] * eth.N_ETH)
    elif arm == 'observed':
        eth.use_observed()
    else:
        errormsg = f'arm must be one of {ARMS}, got {arm!r}'
        raise ValueError(errormsg)
    return eth.VACC_UPTAKE.as_dict(), eth.SCREEN_UPTAKE.as_dict()


def make_pars(arm, seed, **overrides):
    """
    basePars_community's Gamma community-network pars with the four ethnic communities, census
    margins, gamma_shape made explicit, ethnicity-differential uptake, and the 1900-2050 / 100k
    setup.

    ``overrides`` win over everything and are how --diagnose and --selftest shrink the run without
    touching the network configuration being measured.
    """
    apply_arm(arm)

    pars = sc.dcp(bpc.base_pars_geno)
    pars['community_pars'] = community_pars()

    pars.update(
        n_agents=N_AGENTS,
        start=START,
        end=END,
        rand_seed=seed,
        # Drop-in for NHS_2025_lambdamu.get_interventions(l=1, m=1) + NHS_Vacc.vaccinations, with
        # every acceptance probability scaled by the agent's ethnicity. Rebuilt per call because
        # the clones hold per-sim state.
        interventions=eth.get_interventions(l=1, m=1),
        community_results=COMMUNITY_RESULTS,
        # basePars_community has its network_history analyzer commented out; set explicitly so a
        # future uncomment there cannot quietly start keeping a NetworkDelta per timestep per run.
        # --diagnose puts one back, for one seed.
        analyzers=[],
        verbose=-1,
    )
    pars.update(overrides)

    # Pre-1950 start: see the module docstring. setdefault, not update, so an explicit override of
    # either key wins -- and note this keys off the FINAL start year, after ``overrides``.
    if pars['start'] < DATA_START_YEAR:
        if not AGE_DATAFILE.exists():
            errormsg = (f'A start year of {pars["start"]} is before the UK demographic data '
                        f'({DATA_START_YEAR}), so the initial age distribution must be supplied, '
                        f'but {AGE_DATAFILE} does not exist.')
            raise FileNotFoundError(errormsg)
        pars.setdefault('age_datafile', str(AGE_DATAFILE))
        pars.setdefault('use_migration', False)

    return pars


def describe(pars, arm):
    """ One block at startup, so a CSV can always be traced back to what produced it """
    cp = pars['community_pars']
    ihp = pars['init_hpv_prev']
    k = float(cp['gamma_shape'])
    theta_cv = 1 / np.sqrt(k)  # Gamma(shape=k) CV -- contrast powerlaw.py's Pareto 1/sqrt(a(a-2))
    probs = np.asarray(cp['community_probs'], dtype=float)
    labels = list(cp.get('community_labels', eth.ETHNICITIES))
    vacc, screen = eth.VACC_UPTAKE.as_dict(), eth.SCREEN_UPTAKE.as_dict()
    print('Configuration:')
    print(f"  arm                {arm}")
    print(f"  network            {pars['network']}, n_communities={cp['n_communities']}")
    print(f"  community shares   {'CENSUS' if USE_CENSUS_SHARES else 'partnership-end'}  "
          f"{ {l: round(float(p), 4) for l, p in zip(labels, probs)} }")
    print(f"  agents/community   "
          f"{ {l: int(round(float(p) * pars['n_agents'])) for l, p in zip(labels, probs)} }")
    print(f"  vaccination mult   { {kk: round(v, 3) for kk, v in vacc.items()} }")
    print(f"  screening mult     { {kk: round(v, 3) for kk, v in screen.items()} }")
    print(f"  Gamma shape        {k}  (theta CV = {theta_cv:.3f})")
    print(f"  p_single_annual    {cp.get('p_single_annual', 0.0)}  "
          f"({'gate on' if cp.get('p_single_annual', 0.0) else 'gate OFF'})")
    print(f"  partners/yr target {cp['mean_partners_per_year']}  (including singles)")
    print(f"  frac_long          {cp['frac_long']}   D_short={cp['D_mean_short']} mo, "
          f"D_long={cp['D_mean_long']} mo")
    print(f"  years              {pars['start']}-{pars['end']}   n_agents={pars['n_agents']:,}   "
          f"dt={pars['dt']}")
    print(f"  use_migration      {pars.get('use_migration', True)}"
          f"{'  (off: start precedes the demographic data)' if not pars.get('use_migration', True) else ''}")
    print(f"  interventions      {len(pars['interventions'])}  (ethnicity-aware screening + vacc)")
    print(f"  by-community       {COMMUNITY_RESULTS}")
    print(f"  genotypes          {list(pars['genotypes'])}")
    print(f"  init_hpv_prev f    {np.asarray(ihp['f'])}")
    print(f"  init_hpv_prev m    {np.asarray(ihp['m'])}", flush=True)
    return


# -------------------------------------------------------------------
# export
# -------------------------------------------------------------------

def export_df(run_sim):
    """
    run_sim.to_df(date_index=True, by_community=...), widened with every by-genotype result plus
    infections_star and prevalence_star. See the module docstring for what those two are.

    The extra columns are built in one frame and joined in a single call, the way Sim.to_df's own
    by_community block does -- inserting them one at a time fragments the frame.
    """
    df = run_sim.to_df(date_index=True, by_community=EXPORT_BY_COMMUNITY)
    res = run_sim.results
    genotypes = list(run_sim['genotype_map'].values())

    newcols = {}
    suffix = '_by_genotype'
    for key in run_sim.result_keys('genotype'):
        stem = key[:-len(suffix)] if key.endswith(suffix) else key
        vals = np.asarray(res[key][:], dtype=float)
        for gi, g in enumerate(genotypes):
            newcols[f'{stem}_{g}'] = vals[gi, :]

    n_infected = np.asarray(res['n_infected'][:], dtype=float)
    n_alive = np.asarray(res['n_alive'][:], dtype=float)
    newcols['infections_star'] = n_infected
    newcols['prevalence_star'] = sc.safedivide(n_infected, n_alive)

    clashes = sorted(set(newcols) & set(df.columns))
    if clashes:
        # join() would raise anyway, but say which column and why rather than letting pandas do it
        errormsg = f'new columns collide with existing to_df() columns: {sc.strjoin(clashes)}'
        raise ValueError(errormsg)

    return df.join(pd.DataFrame(newcols, index=df.index))


# -------------------------------------------------------------------
# the run itself
# -------------------------------------------------------------------

def csv_path(arm, outdir=None, suffix=''):
    outdir = pathlib.Path(outdir or OUTPUT_DIR)
    return outdir / f'{TAG}_{arm}{suffix}.csv'


def done_seeds(path, resume=True):
    """ Seeds already in an output file, so a re-invocation continues instead of duplicating. """
    if not path.exists():
        return set(), False
    if not resume:
        # Appending onto an existing file would interleave two different runs' results
        errormsg = (f'{path} already exists -- move or delete it first, otherwise these runs '
                    f'would be appended onto the previous ones.')
        raise FileExistsError(errormsg)
    prev = pd.read_csv(path, usecols=['Seed'])
    return {int(x) for x in prev['Seed'].unique()}, True


def run_batch(arm, seed, path, n_runs, n_cpus, resume=True, label=None, **par_overrides):
    """ One batch of n_runs runs of one arm, appended to that arm's CSV. """
    already, header_written = done_seeds(path, resume=resume)
    batch = [seed + i for i in range(n_runs)]
    if all(s in already for s in batch):
        print(f'  [{arm}] seeds {batch[0]}-{batch[-1]} already on disk, skipping', flush=True)
        return

    # MultiSim varies rand_seed by run index, so this batch covers seeds seed..seed+n_runs-1
    sim = hpv.Sim(make_pars(arm, seed, **par_overrides), label=label or f'{TAG} {arm}')
    print(f'  [{arm}] seeds {batch[0]}-{batch[-1]} ...', flush=True)
    msim = hpv.MultiSim(sim)
    msim.run(n_runs=n_runs, n_cpus=n_cpus)

    written = 0
    for i, run_sim in enumerate(msim.sims):
        this_seed = batch[i]
        if this_seed in already:  # partially-written batch: keep the rows already on disk
            print(f'  [{arm}] seed {this_seed} already in the CSV, not rewriting.', flush=True)
            continue
        try:
            temp_df = export_df(run_sim)
        except Exception as e:
            print(f'  [{arm}] could not save seed {this_seed} to df: {e}', flush=True)
            continue
        temp_df['Seed'] = this_seed
        # Header on the first write only, so the CSV is directly readable with
        # pd.read_csv(..., index_col=0) and needs no cleaning pass
        temp_df.to_csv(path, mode='a', index=True, header=not header_written)
        header_written = True
        written += 1

        doses = float(run_sim.results['cum_doses'][-1])
        vacc = float(run_sim.results['cum_vaccinated'][-1])
        print(f'  [{arm}] Seed:{this_seed} is done  --  {len(temp_df.columns)} columns, '
              f'cum_doses={doses:,.0f}, cum_vaccinated={vacc:,.0f}', flush=True)

    ncomm = len([c for c in temp_df.columns if 'by_community' in c]) if written else 0
    print(f'  [{arm}] wrote {written} run(s), seeds {batch[0]}-{batch[-1]}  ({ncomm} by-community '
          f'columns)', flush=True)
    del msim
    return


def main():
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    for arm in WHICH:
        describe(make_pars(arm, SEEDS[0]), arm)
        print()
    print(f'Seed batches: {SEEDS}  -> up to {len(SEEDS) * N_RUNS} runs per arm, arms {WHICH}')
    for arm in WHICH:
        print(f'  -> {csv_path(arm)}')
    print(flush=True)

    # Batches outer, arms inner, so both arms always hold the SAME seeds. If the run is stopped
    # part-way the comparison is still balanced, rather than one arm 50 runs deep and the other
    # not started.
    for seed in SEEDS:
        for arm in WHICH:
            run_batch(arm, seed, csv_path(arm), N_RUNS, N_CPUS, resume=RESUME)

    print('\nDone. Wrote:')
    for arm in WHICH:
        print(f'  {csv_path(arm)}')
    print('\nNext:\n  python analyse_ethnicity_uptake.py\n  python plot_ethnicity_uptake.py')
    return


# -------------------------------------------------------------------
# --diagnose: what network did we actually get, and how separated are the communities?
# -------------------------------------------------------------------

def diagnose():
    """
    Measure the realised network structure of the configuration main() runs, including the
    per-community breakdown that run_gamma2_nocomm_1900.py has nothing to report.

    Why this exists even though basePars_community's knobs are settled: acbnm.calibrate() fits rho
    on a 2,500-node TOY model with no sexual debut, no mortality and no migration, over a single
    12-month window. The real sim has debut and mortality, plus CommunityNetworkBackend's
    mortality-aware inflation of q_short/q_long -- and here it also has migration switched OFF, a
    growing population, and four communities with census margins rather than the survey's own.

    The arm does not matter to network structure (uptake changes who gets vaccinated, not who
    partners whom), so this runs the control arm only.
    """
    # calibrate_default_poisson has no import-time side effects (its only raise is inside a
    # function) -- the sibling run scripts import from it the same way
    from calibrate_default_poisson import (
        TARGETS, _ActiveTracker, degree_from_edges, union_edges_window, active_union_window,
        pooled_mean_degree_excl_singles,
    )

    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'{TAG}_network_diagnostic.txt'

    # verbose > 0 so CommunityNetworkBackend prints its own calibration trace, which is half the
    # point of this mode; 0.1 means a progress line every 10 timesteps
    pars = make_pars('equal', DIAG_SEED, n_agents=DIAG_AGENTS, end=DIAG_END, interventions=[],
                     analyzers=[hpv.network_history(), _ActiveTracker()], verbose=0.1)
    describe(pars, 'equal')
    print(f'\nDiagnostic run: seed {DIAG_SEED}, {DIAG_AGENTS:,} agents, {pars["start"]}-{DIAG_END}, '
          f'no interventions.', flush=True)

    sim = hpv.Sim(pars, label=f'{TAG} network diagnostic')
    sim.run()

    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    n = len(sim.people)
    t_end = sim.npts - 1
    spy = int(round(1 / sim['dt']))

    # --- Instantaneous snapshot at the last timestep: relationship-status proportions ---
    edges_now = nh.edges_at(t_end)
    active_f, active_m = act.active_female[t_end], act.active_male[t_end]
    deg_long = degree_from_edges(edges_now, n, lkey=LKEY_LONG)
    deg_short = degree_from_edges(edges_now, n, lkey=LKEY_SHORT)
    deg_total = deg_long + deg_short  # the two layers are disjoint edge sets, so this is exact

    def pooled(mask):
        pool = np.concatenate([mask[active_f], mask[active_m]])
        return float(pool.mean()) if pool.size else float('nan')

    p_long = pooled(deg_long >= 1)
    p_short = pooled(deg_short >= 1)
    p_single = pooled(deg_total == 0)

    # --- Windowed unions: distinct partners over the final 1- and 5-year periods ---
    mean_deg_excl = pooled_mean_degree_excl_singles(nh, act, t_end, spy, 1, n)
    mean_deg_5yr = pooled_mean_degree_excl_singles(nh, act, t_end, spy, 5, n)

    annual_edges = union_edges_window(nh, t_end, spy, 1)
    annual_deg = degree_from_edges(annual_edges, n)
    f_union, m_union = active_union_window(act, t_end, spy, 1)
    active_any = np.array(sorted(f_union | m_union), dtype=np.int64)
    ad = annual_deg[active_any]
    mean_deg_incl = float(ad.mean()) if ad.size else float('nan')
    frac_zero_in_year = float((ad == 0).mean()) if ad.size else float('nan')
    partnered = ad[ad >= 1]
    cv_annual = (float(partnered.std() / partnered.mean())
                 if partnered.size and partnered.mean() else float('nan'))

    cp = pars['community_pars']
    target_incl = float(cp['mean_partners_per_year'])
    labels = list(cp.get('community_labels', eth.ETHNICITIES))
    probs = np.asarray(cp['community_probs'], dtype=float)

    # --- Per-community realised size, mean annual degree, and assortativity ---
    comm = np.asarray(sim.people.community, dtype=np.int64)
    in_network = comm >= 0  # -1 until sexual debut; bin_communities() drops those
    comm_rows = []
    within = between = 0
    for f, m, _layer in annual_edges.values():
        cf, cm = comm[f], comm[m]
        if cf < 0 or cm < 0:
            continue
        if cf == cm:
            within += 1
        else:
            between += 1
    within_frac = within / (within + between) if (within + between) else float('nan')

    for ci, label in enumerate(labels):
        members = np.flatnonzero(in_network & (comm == ci))
        if not members.size:
            comm_rows.append(f'  {label:<10} {0:>8}  {"--":>10}  {"--":>10}  {"--":>10}')
            continue
        realised_share = members.size / max(int(in_network.sum()), 1)
        d = annual_deg[members]
        dp = d[d >= 1]
        comm_rows.append(
            f'  {label:<10} {members.size:>8}  {realised_share:>10.4f}  '
            f'{float(d.mean()):>10.3f}  {(float(dp.mean()) if dp.size else float("nan")):>10.3f}')

    migration_on = bool(pars.get('use_migration', True))
    mig_note = 'migration on' if migration_on else 'migration off'

    def row(label, value, target=None, note=''):
        tgt = f'  target {target:>6.3f}' if target is not None else ' ' * 14
        return f'  {label:<38} {value:>8.3f}{tgt}   {note}'

    lines = [
        f'{TAG}  --  network diagnostic',
        '',
        f"  gamma_shape={cp['gamma_shape']}  n_communities={cp['n_communities']}  "
        f"shares={'CENSUS' if USE_CENSUS_SHARES else 'partnership-end'}  "
        f"mean_partners_per_year={target_incl} (including singles)",
        f"  frac_long={cp['frac_long']}  D_short={cp['D_mean_short']} mo  "
        f"D_long={cp['D_mean_long']} mo  p_single_annual={cp.get('p_single_annual', 0.0)}",
        f'  seed {DIAG_SEED}, {DIAG_AGENTS:,} agents, {pars["start"]}-{DIAG_END}, {mig_note}, '
        f'measured at the final timestep',
        '',
        '  Annual distinct partners (whole network)',
        row('mean, EXCLUDING singles', mean_deg_excl, TARGETS['mean_degree_annual'],
            'Natsal convention; not what is fitted'),
        row('mean, INCLUDING singles', mean_deg_incl, target_incl,
            'community_pars convention  <-- FITTED'),
        row('CV among the partnered', cv_annual, 1.831,
            'Natsal cv_degree_annual; falls as gamma_shape rises'),
        row('fraction with 0 partners in the year', frac_zero_in_year),
        '',
        '  5-year distinct partners',
        row('mean, EXCLUDING singles', mean_deg_5yr, TARGETS['mean_degree_5yr']),
        '',
        '  Instantaneous snapshot',
        row('p_single', p_single, TARGETS['p_single']),
        row('p_long', p_long, TARGETS['p_long']),
        row('p_short', p_short, TARGETS['p_short']),
        '',
        '  By community (network members only -- community is -1 before sexual debut)',
        f'  {"label":<10} {"members":>8}  {"share":>10}  {"deg(all)":>10}  {"deg(ptnrd)":>10}',
        *comm_rows,
        f'  target shares: { {l: round(float(p), 4) for l, p in zip(labels, probs)} }',
        '',
        row('within-community edge fraction', within_frac, None,
            'annual union; 1.0 = fully assortative, ~sum(p^2) = random'),
        row('   random-mixing reference', float((probs ** 2).sum()), None,
            'what it would be with no assortativity'),
        '',
        '  The FITTED row is the one acbnm.calibrate() solved for, on its 2,500-node toy model',
        '  without debut/mortality/migration. If it is far off here, in the real sim, re-tune',
        "  community_pars['mean_partners_per_year'] in basePars_community.py and re-run this",
        '  diagnostic before starting the production runs. A large gap between the two annual',
        '  rows is expected, not a fault: it is just the single fraction reported two lines below',
        '  them.',
    ] + ([] if migration_on else [
        '  This run has migration OFF (its start precedes the demographic data), so the age',
        "  structure here drifts from the usual 1980-start runs' -- some movement against the",
        '  targets is expected. Re-run with START=1980 DIAG_END=2010 TAG=... for a control.',
    ])
    text = '\n'.join(lines) + '\n'
    out_path.write_text(text, encoding='utf-8')
    print()
    print(text)
    print(f'Wrote {out_path}')
    return


# -------------------------------------------------------------------
# --selftest: does the pipeline produce what it claims to?
# -------------------------------------------------------------------

def init_hpv_prev_matches_basepars():
    """
    basePars_community's init_hpv_prev is still the same as basePars.py's.

    Both files carry their own copy of the current ("new") arrays, so an edit to one and not the
    other would silently put this run on a different initial prevalence from its siblings.
    """
    a, b = bpc.base_pars['init_hpv_prev'], basePars.base_pars['init_hpv_prev']
    return bool(
        set(a) == set(b)
        and all(np.array_equal(np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float))
                for k in a)
    )


def selftest():
    """
    The real code path (make_pars -> MultiSim -> export_df -> CSV) at a size that finishes in a
    few minutes, then the column invariants checked on the CSV as read back off disk. Both arms,
    because the whole point of this script is that they differ.
    """
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    frames = {}
    for arm in ARMS:
        path = csv_path(arm, suffix='_selftest')
        if path.exists():
            path.unlink()  # a self-test always starts clean, unlike the production run
        pars = make_pars(arm, 0, n_agents=SELFTEST_AGENTS, end=SELFTEST_END)
        describe(pars, arm)
        print(f'\nSelf-test [{arm}]: {SELFTEST_RUNS} runs, {SELFTEST_AGENTS:,} agents, '
              f'{START}-{SELFTEST_END}', flush=True)
        run_batch(arm, 0, path, SELFTEST_RUNS, min(SELFTEST_RUNS, N_CPUS), resume=False,
                  label=f'{TAG} {arm} self-test', n_agents=SELFTEST_AGENTS, end=SELFTEST_END)
        frames[arm] = pd.read_csv(path, index_col=0)
        print()

    pars = make_pars('observed', 0, n_agents=SELFTEST_AGENTS, end=SELFTEST_END)
    df = frames['observed']
    ctl = frames['equal']
    gt = [str(g) for g in pars['genotypes']]
    gen_cols = [c for c in df.columns if any(c.endswith(f'_{g}') for g in gt)]
    comm_cols = [c for c in df.columns if 'by_community' in c]
    cp = pars['community_pars']
    labels = list(cp.get('community_labels', eth.ETHNICITIES))

    tol = 1e-9
    n_alive = df['n_alive'].to_numpy()

    # The two arms must differ, and in the expected direction: differential uptake is <= 1 for
    # every non-White group, so total doses should come out LOWER than the equal-uptake control
    doses_obs = float(df.loc[df.index == SELFTEST_END, 'cum_doses'].median())
    doses_eq = float(ctl.loc[ctl.index == SELFTEST_END, 'cum_doses'].median())
    screens_obs = float(df.loc[df.index == SELFTEST_END, 'cum_screens'].median())
    screens_eq = float(ctl.loc[ctl.index == SELFTEST_END, 'cum_screens'].median())

    checks = [
        ('network is the community one, Gamma propensity (powerlaw not imported)',
         pars['network'] == 'community' and 'powerlaw' not in sys.modules),
        (f'gamma_shape reached the sim as {GAMMA_SHAPE}',
         float(cp['gamma_shape']) == float(GAMMA_SHAPE)),
        ('four communities, with the ethnicity labels',
         int(cp['n_communities']) == 4 and labels == list(eth.ETHNICITIES)),
        (f'community shares are {"census" if USE_CENSUS_SHARES else "partnership-end"}',
         bool(np.allclose(np.asarray(cp['community_probs'], dtype=float),
                          CENSUS_SHARES if USE_CENSUS_SHARES
                          else np.asarray(bpc.community_probs, dtype=float)))),
        ('community_mixing is 4x4 and symmetric',
         np.asarray(cp['community_mixing']).shape == (4, 4)
         and bool(np.allclose(cp['community_mixing'], np.asarray(cp['community_mixing']).T))),
        ('init_hpv_prev matches basePars.py (the current arrays)',
         init_hpv_prev_matches_basepars()),
        (f'pre-{DATA_START_YEAR} start: age_datafile set and migration off',
         pars['start'] >= DATA_START_YEAR
         or (pars.get('age_datafile') == str(AGE_DATAFILE) and pars['use_migration'] is False)),
        ('index is years, single header row',
         pd.api.types.is_numeric_dtype(df.index) and float(df.index.min()) == float(START)),
        (f'{SELFTEST_RUNS} seeds present in both arms',
         df['Seed'].nunique() == SELFTEST_RUNS and ctl['Seed'].nunique() == SELFTEST_RUNS),
        ('both arms hold the same seeds',
         sorted(df['Seed'].unique()) == sorted(ctl['Seed'].unique())),
        ('infections_star and prevalence_star present',
         {'infections_star', 'prevalence_star'}.issubset(df.columns)),
        (f'{N_GENOTYPE_RESULTS * len(gt)} by-genotype columns '
         f'({len(gt)} genotypes x {N_GENOTYPE_RESULTS} results)',
         len(gen_cols) == N_GENOTYPE_RESULTS * len(gt)),
        ('by-community columns present, a multiple of 4',
         len(comm_cols) > 0 and len(comm_cols) % 4 == 0),
        ('every community label appears in the by-community columns',
         all(any(c.endswith(f'_{l}') for c in comm_cols) for l in labels)),
        ('prevalence_star == infections_star / n_alive',
         bool(np.allclose(df['prevalence_star'], df['infections_star'] / n_alive, atol=tol))),
        ('prevalence_star >= hpv_prevalence everywhere',
         bool((df['prevalence_star'] >= df['hpv_prevalence'] - tol).all())),
        ('hpv_prevalence == n_infectious / n_alive',
         bool(np.allclose(df['hpv_prevalence'], df['n_infectious'] / n_alive, atol=tol))),
        ('infections == sum over genotypes of infections_<g>',
         bool(np.allclose(df['infections'], sum(df[f'infections_{g}'] for g in gt), rtol=1e-6))),
        ('screening is live after 1980',
         bool(float(df.loc[df.index >= 1981, 'cum_screens'].max()) > 0)),
        ('vaccination is live after 2008',
         bool(float(df.loc[df.index >= 2009, 'cum_doses'].max()) > 0)),
        (f'observed arm gives FEWER doses than equal ({doses_obs:,.0f} < {doses_eq:,.0f})',
         doses_obs < doses_eq),
        (f'observed arm gives FEWER screens than equal ({screens_obs:,.0f} < {screens_eq:,.0f})',
         screens_obs < screens_eq),
    ]

    print()
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    n_fail = sum(1 for _, ok in checks if not ok)
    print(f'\n{len(checks) - n_fail}/{len(checks)} checks passed.')
    for arm in ARMS:
        p = csv_path(arm, suffix='_selftest')
        print(f'Wrote {p}  ({len(frames[arm].columns)} columns, {len(frames[arm])} rows).')
    if n_fail:
        sys.exit(1)
    return


if __name__ == '__main__':
    args = set(sys.argv[1:])
    if '--diagnose' in args:
        diagnose()
    elif '--selftest' in args:
        selftest()
    else:
        main()

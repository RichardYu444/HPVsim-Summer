"""
run_gamma2_nocomm_1900.py
=========================

50 full runs of the community network with the package's own GAMMA partner-propensity
distribution at shape 2 and NO community structure (n_communities = 1), over 1900-2050 at
100,000 agents, under the existing NON-TARGETED NHS vaccination programme
(NHS_Vacc.vaccinations), with the same wide per-genotype export as run_powerlaw_nogate.py and
run_default_meandeg1p4.py.

This is the Gamma sibling of

    csvs/powerlaw_alpha3p5_nogate_100k_1950_2070_50runs.csv   (Pareto propensity)
    csvs/default_meandeg1p4_100k_1950_2070_50runs.csv         (HPVsim's built-in network)

and is deliberately shaped like them -- same 10 x 5 seed batches, same interventions, same
exported columns, same resume behaviour. What differs is the network AND the window: 1900-2050
rather than 1950-2070, so the epidemic gets ~80 years to settle before screening starts (1980)
and ~108 before vaccination (2008), instead of ~30 and ~58.

WHAT "NO COMMUNITIES" MEANS HERE
--------------------------------
basePars_community.py runs four communities standing in for ethnic groups (ETHNICITIES, with a
Natsal-derived community_mixing kernel). This run collapses that to a single community:

    n_communities     4  ->  1
    community_probs   ethnic partnership-end shares  ->  [1.0]
    community_mixing  4x4 Natsal kernel              ->  [[1.0]]

which is the same single-community configuration validate_r0.py calls 'gamma1' and
basePars_community_powerlaw.py uses by default. With one community the mixing kernel is
degenerate (a 1x1 matrix of ones), so partner choice is driven by age and propensity alone --
exactly the comparison the power-law and default runs above are on.

gamma_shape is already 2 in basePars_community.py; it is restated here as a named module-level
value so the figure this run is named after appears in describe()'s output and can be re-tuned
from the environment without editing basePars_community.py (which run_sim_community_50.py,
validate_r0.py, network.py and the NHS_* analyses all import). --selftest asserts the value that
actually reached the sim.

STARTING IN 1900 NEEDS TWO ADJUSTMENTS
--------------------------------------
UK demographic data start in 1950, so a 1900 start is outside them. Death rates (nearest year)
and birth rates (clamped interpolation) already fall back to 1950 on their own, but two things do
not, and both are handled in make_pars() -- the same two run_equilibrium_1900.py handles:

1. The initial age distribution is looked up for the EXACT start year
   (population.py -> hpdata.get_age_distribution(location, year=sim['start'])), which selects
   `Time == 1900` from the UN table and comes back empty. pars['age_datafile'] therefore points
   at csvs/equilibrium_1900/uk_age_distribution_1950.csv, i.e. the population is started on
   1950's age structure.

2. People.check_migration() raises NotImplementedError outright when the sim starts before the
   data ("Starting the sim earlier than the data is not hard, but has not been done yet"), so
   pars['use_migration'] = False. Unlike the usual 1980-start runs, population size is then set
   by births and deaths alone and drifts upward: expect the agent count to roughly triple over
   150 years, which is where the memory goes.

Both are applied only when start < 1950, and an explicit override of either wins, so shrinking
the window for --selftest/--diagnose does not silently change them.

INITIAL PREVALENCE
------------------
basePars_community.py's own init_hpv_prev is the CURRENT one -- the f/m arrays that were swapped
and m revalued. It is taken as-is, unlike run_powerlaw_nogate.py, which had to borrow basePars'
copy because basePars_community_powerlaw.py's was stale. --selftest checks it still matches
basePars.py's, so a future edit to one and not the other cannot pass silently.

WHAT THIS CAPTURES THAT EARLIER COMMUNITY RUNS DID NOT
------------------------------------------------------
Identical to the two sibling scripts, and for the same reasons:

1. Per-genotype dynamics. Sim.to_df() keeps only 1-D results, so every '*_by_genotype' result was
   silently dropped from csvs/community_gamma2_50runs.csv. export_df() below adds all of them
   back, one column per result per genotype (e.g. hpv_prevalence_hpv16, infections_ohr,
   n_cin_hpv18).

2. infections* / prevalence*, exported as the columns `infections_star` and `prevalence_star`:

       infections_star = people.infected.any(axis=0), counted once per person, scale-weighted
       prevalence_star = infections_star / n_alive

   People.infected is a property (base.py) = infectious | inactive, i.e. currently infectious OR
   carrying an inactive/latent infection, including people with cancer. It is a strict superset
   of `infectious`, so prevalence_star >= hpv_prevalence always. The sim already stores
   people.count_any('infected') as the 1-D result n_infected, so infections_star is that result
   under a clearer name; prevalence_star is formed exactly the way hpv_prevalence is
   (sim.py: safedivide(n_infectious, n_alive)), and shares its denominator -- remove_people()
   zeroes infectious/inactive on death, so neither numerator carries the dead.

By-community results are deliberately left off (pars['community_results'] stays False): with one
community they would just duplicate the totals.

WHY NOTHING IS EDITED OUTSIDE THIS FILE
---------------------------------------
basePars_community.py is imported by run_sim_community_50.py, run_sim_community_2iso.py,
validate_r0.py, network.py, r0_hpv.py and the NHS_ethnicity_* analyses, and must keep its current
behaviour, so every change above is layered onto a deep copy at run time.

Imports stay at MODULE level: sciris parallelize spawns workers on Windows that re-import
__main__, and an import tucked inside main() would be re-executed per worker for no benefit.
basePars_community_powerlaw and powerlaw are deliberately NOT imported anywhere in this file or
its import graph -- importing either installs the Pareto theta sampler process-wide as a side
effect, which would silently replace the Gamma propensity this run is about.

THREE MODES
-----------
    python run_gamma2_nocomm_1900.py --selftest    ~5 min    pipeline + column checks, small sim
    python run_gamma2_nocomm_1900.py --diagnose    ~10 min   realised network structure, 1 seed
    python run_gamma2_nocomm_1900.py               ~5-9 h    the 50 production runs

Run --selftest and --diagnose first: --diagnose reports the realised mean degree, which is worth
reading before spending hours rather than after.

Known, pre-existing and not introduced here: NHS_Vacc's vx_1921_d2 carries a probability above 1
(0.64*2/(0.60+0.53) = 1.13 -- flagged in NHS_Vacc.py's own TODO), so interventions.py's
annual_prob conversion prints "RuntimeWarning: invalid value encountered in power". Every run
using the usual interventions has always printed it.

If stdout is redirected to a file, set PYTHONIOENCODING=utf-8 -- Sim.brief() prints a glyph cp1252
cannot encode, and the resulting UnicodeEncodeError surfaces as an opaque sciris "Task N failed".

Interruptible: batches are appended to the CSV as they finish, and RESUME makes a re-invocation
skip whatever is already in the file.

Outputs into OUTPUT_DIR:

    <TAG>.csv                       50 runs stacked; year index, t, every 1-D result, every
                                    by-genotype result, infections_star, prevalence_star, Seed
    <TAG>_network_diagnostic.txt    --diagnose only
    <TAG>_selftest.csv              --selftest only
"""
import os
import pathlib
import sys

import numpy as np
import pandas as pd
import sciris as sc

import basePars                    # only to check init_hpv_prev has not drifted (--selftest)
import basePars_community as bpc   # the Gamma community network; NEVER the powerlaw sibling
import hpvsim_working as hpv
import NHS_2025_lambdamu
import NHS_Vacc


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

# Environment-set rather than argv-derived so spawned workers inherit them (see run_core_vacc.py)
GAMMA_SHAPE = float(os.environ.get('GAMMA_SHAPE', 2.0))  # basePars_community's own value

N_AGENTS = int(os.environ.get('N_AGENTS', 100_000))
START = int(os.environ.get('START', 1900))
END = int(os.environ.get('END', 2050))

TAG = os.environ.get('TAG', 'gamma2_nocomm_100k_1900_2050_50runs')
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

# First year of the UK demographic data. A start before this needs the two adjustments in
# make_pars() -- see the module docstring.
DATA_START_YEAR = 1950
AGE_DATAFILE = (pathlib.Path(__file__).parent / 'csvs' / 'equilibrium_1900'
                / 'uk_age_distribution_1950.csv')

# Seeds per batch, and how many of them run at once. The siblings hard-code both at 5; here they
# are env-overridable because this run needs more memory per worker than they do. A 1900 start
# forces use_migration=False, so the population is not pinned to the UK trend and grows on births
# and deaths alone -- ~100k agents at 1900 becomes ~300k by 2050, and a worker peaks near 1.0 GB
# rather than the ~0.5 GB a 1950-start run uses.
#
# Measured on this machine (15.4 GB total, ~10 GB already in use by other apps). One 1900-2050
# sim costs roughly 20-30 min of CPU, and a batch is 5 seeds:
#
#     5 workers   5.5 GB resident, 0.9 GB free, ~49% CPU efficiency (half the wall time is
#                 paging)  ->  one wave at ~2x cost  ->  ~60 min/batch
#     3 workers   2.9 GB resident, 3.1 GB free, ~100% CPU efficiency (no page file at all)
#                 ->  two waves at full speed      ->  ~40-60 min/batch
#
# So throughput is roughly a WASH between the two. 3 is the default not because it is faster but
# because it leaves the machine usable and keeps well clear of an OOM -- at 5 workers the box has
# under 1 GB free, and the population is still growing at that point. Raise N_CPUS if the machine
# is otherwise idle or has more RAM; there is no gain above 5 either way, since N_RUNS is 5.
#
#     N_CPUS=5 python run_gamma2_nocomm_1900.py
N_RUNS = int(os.environ.get('N_RUNS', 5))
N_CPUS = int(os.environ.get('N_CPUS', 3))

SEEDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 seeds gets us to 5 * 10 = 50 total runs (0-49)

# Number of leading seed batches to run, N_RUNS each. 10 (the default) is the full 50 runs; set it
# lower for a quick look. Because RESUME skips whatever is already in the CSV, a short run can be
# extended to the full 50 later just by re-running with a larger value -- only the missing batches
# are computed.
#
#     N_BATCHES=2 python run_gamma2_nocomm_1900.py
SEEDS = SEEDS[:int(os.environ.get('N_BATCHES', len(SEEDS)))]

# If the output CSV already exists, continue from where a previous attempt stopped rather than
# refusing to start. These runs take hours and the batches are written incrementally, so an
# interruption (machine sleeping, session ending, Ctrl-C) otherwise throws away everything done so
# far. Seeds already present in the CSV are skipped and never rewritten, so resuming cannot
# duplicate or interleave rows. Set False to get a hard failure on an existing file instead.
RESUME = True

# --diagnose settings. The epidemic is irrelevant to network structure, so this stops well before
# 2050; interventions are dropped because screening's 1980 start would then fall outside the sim,
# which HPVsim rejects. 30 years of demography on top of the backend's own pre-t=0 network burn-in.
DIAG_SEED = int(os.environ.get('DIAG_SEED', 0))
DIAG_AGENTS = int(os.environ.get('DIAG_AGENTS', N_AGENTS))
DIAG_END = int(os.environ.get('DIAG_END', 1930))

# --selftest settings: small and short, but through exactly the same code path as the real run.
# SELFTEST_END has to clear 2008 for the vaccination check to mean anything.
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
    basePars_community's community_pars with the four ethnic communities collapsed to one and
    gamma_shape made explicit.

    Deep-copied so nothing here can reach back into the module-level dict that
    run_sim_community_50.py, validate_r0.py and the NHS_* analyses read. (CommunityNetworkBackend
    does take its own copy before popping keys off it, but relying on that from the outside is not
    worth the saved microsecond.)

    The single-community triple below is the one validate_r0.py's 'gamma1' model uses; with
    n_communities=1 the mixing kernel is a 1x1 matrix of ones, i.e. no community structure at all.
    community_labels is kept at length n_communities so anything that zips the two (e.g.
    NHS_ethnicity_uptake's check) stays consistent.
    """
    return sc.dcp(sc.mergedicts(bpc.community_pars, dict(
        gamma_shape=float(GAMMA_SHAPE if gamma_shape is None else gamma_shape),
        n_communities=1,
        community_probs=np.array([1.0]),
        community_mixing=np.array([[1.0]]),
        community_labels=['All'],
    )))


def make_pars(seed, **overrides):
    """
    basePars_community's Gamma community-network pars with no community structure, gamma_shape
    made explicit, and the 1900-2050 / 100k / non-targeted-vaccination setup.

    ``overrides`` win over everything and are how --diagnose and --selftest shrink the run without
    touching the network configuration being measured.
    """
    pars = sc.dcp(bpc.base_pars_geno)
    pars['community_pars'] = community_pars()

    pars.update(
        n_agents=N_AGENTS,
        start=START,
        end=END,
        rand_seed=seed,
        # "Old", non-targeted vaccination: the NHS programme as modelled today. Rebuilt/dcp'd
        # because basePars_community's list holds the module-level intervention objects shared
        # with every other importer of NHS_Vacc.
        interventions=NHS_2025_lambdamu.get_interventions(l=1, m=1) + sc.dcp(NHS_Vacc.vaccinations),
        # basePars_community has its network_history analyzer commented out; set explicitly so a
        # future uncomment there cannot quietly start keeping a NetworkDelta per timestep per run
        # (a lot of memory at 100k agents x 50 runs, and nothing here reads it). --diagnose puts
        # one back, for one seed.
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


def describe(pars):
    """ One block at startup, so a CSV can always be traced back to what produced it """
    cp = pars['community_pars']
    ihp = pars['init_hpv_prev']
    k = float(cp['gamma_shape'])
    theta_cv = 1 / np.sqrt(k)  # Gamma(shape=k) CV -- contrast powerlaw.py's Pareto 1/sqrt(a(a-2))
    n_comm = int(cp['n_communities'])
    print('Configuration:')
    print(f"  network            {pars['network']}, n_communities={n_comm}"
          f"{'  (no communities)' if n_comm == 1 else ''}")
    print(f"  Gamma shape        {k}  (theta CV = {theta_cv:.3f})")
    print(f"  p_single_annual    {cp.get('p_single_annual', 0.0)}  "
          f"({'gate on' if cp.get('p_single_annual', 0.0) else 'gate OFF'})")
    print(f"  partners/yr target {cp['mean_partners_per_year']}  (including singles)")
    print(f"  frac_long          {cp['frac_long']}   D_short={cp['D_mean_short']} mo, "
          f"D_long={cp['D_mean_long']} mo")
    print(f"  condoms            s={pars['condoms']['s']}, l={pars['condoms']['l']}")
    print(f"  years              {pars['start']}-{pars['end']}   n_agents={pars['n_agents']:,}   "
          f"dt={pars['dt']}")
    print(f"  use_migration      {pars.get('use_migration', True)}"
          f"{'  (off: start precedes the demographic data)' if not pars.get('use_migration', True) else ''}")
    print(f"  age_datafile       {pars.get('age_datafile') or 'package default (UN, by start year)'}")
    print(f"  interventions      {len(pars['interventions'])}  (screening + NHS_Vacc, non-targeted)")
    print(f"  genotypes          {list(pars['genotypes'])}")
    print(f"  init_hpv_prev f    {np.asarray(ihp['f'])}")
    print(f"  init_hpv_prev m    {np.asarray(ihp['m'])}", flush=True)
    return


# -------------------------------------------------------------------
# export
# -------------------------------------------------------------------

def export_df(run_sim):
    """
    run_sim.to_df(date_index=True), widened with every by-genotype result plus infections_star and
    prevalence_star. See the module docstring for what those two are.

    The extra columns are built in one frame and joined in a single call, the way Sim.to_df's own
    by_community block does -- inserting them one at a time fragments the frame.
    """
    df = run_sim.to_df(date_index=True)
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

def run_batches(csv_path, seed_list, n_runs, n_cpus, resume=True, label=None, **par_overrides):
    """ MultiSim in batches of n_runs, appending each finished run to one CSV. """
    done_seeds = set()
    header_written = False
    if csv_path.exists():
        if not resume:
            # Appending onto an existing file would interleave two different runs' results
            errormsg = (f'{csv_path} already exists -- move or delete it first, otherwise these '
                        f'runs would be appended onto the previous ones.')
            raise FileExistsError(errormsg)
        prev = pd.read_csv(csv_path, index_col=0)
        done_seeds = {int(x) for x in prev['Seed'].unique()}
        header_written = True
        print(f'RESUME: {len(done_seeds)} run(s) already present, seeds '
              f'{min(done_seeds)}-{max(done_seeds)}. Those will be skipped.', flush=True)

    for seed in seed_list:
        batch = [seed + i for i in range(n_runs)]
        if all(s in done_seeds for s in batch):
            print(f'Skipping seeds {batch[0]}-{batch[-1]} (already in the CSV).', flush=True)
            continue

        # MultiSim varies rand_seed by run index, so this batch covers seeds seed..seed+n_runs-1
        sim = hpv.Sim(make_pars(seed, **par_overrides), label=label or TAG)
        print(f'Running MultiSim with n_runs = {n_runs}  (seeds {seed}-{seed + n_runs - 1}) ...',
              flush=True)
        msim = hpv.MultiSim(sim)
        msim.run(n_runs=n_runs, n_cpus=n_cpus)
        print('MultiSim run complete.', flush=True)

        for i, run_sim in enumerate(msim.sims):
            this_seed = seed + i
            if this_seed in done_seeds:  # partially-written batch: keep the rows already on disk
                print(f'Seed:{this_seed} already in the CSV, not rewriting.', flush=True)
                continue
            try:
                temp_df = export_df(run_sim)
            except Exception as e:
                print(f'Could not save run results to df: {e}', flush=True)
                continue
            temp_df['Seed'] = this_seed
            # Header on the first write only, so the CSV is directly readable with
            # pd.read_csv(..., index_col=0) and needs no cleaning pass
            temp_df.to_csv(csv_path, mode='a', index=True, header=not header_written)
            header_written = True

            doses = float(run_sim.results['cum_doses'][-1])
            vacc = float(run_sim.results['cum_vaccinated'][-1])
            print(f'Seed:{this_seed} is done  --  {len(temp_df.columns)} columns, '
                  f'cum_doses={doses:,.0f}, cum_vaccinated={vacc:,.0f}', flush=True)

        del msim
    return


def main():
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f'{TAG}.csv'

    describe(make_pars(SEEDS[0]))
    print(f'Seed batches: {SEEDS}  -> up to {len(SEEDS) * N_RUNS} runs')
    print(f'Outputs will be saved to: {csv_path}', flush=True)

    run_batches(csv_path, SEEDS, N_RUNS, N_CPUS, resume=RESUME)
    print(f'\nDone. Wrote {csv_path}')
    return


# -------------------------------------------------------------------
# --diagnose: what mean degree did we actually get?
# -------------------------------------------------------------------

def diagnose():
    """
    Measure the realised network structure of the configuration main() runs.

    Why this exists even though basePars_community's knobs are settled: acbnm.calibrate() fits rho
    on a 2,500-node TOY model with no sexual debut, no mortality and no migration, and it measures
    over a single 12-month window. The real sim has debut and mortality, plus
    CommunityNetworkBackend's mortality-aware inflation of q_short/q_long -- and here it also has
    migration switched OFF and a growing population, which the toy model knows nothing about. So
    hitting the target on the toy model is necessary but not sufficient; this measures what the
    actual production configuration produces, using the same windowed-union machinery the
    project's own calibration harnesses use.

    Both conventions are reported. basePars_community's mean_partners_per_year is fitted INCLUDING
    singles (calibrate()'s exclude_singles defaults to False and nothing here overrides it), so
    that is the row marked FITTED; the Natsal target of 1.4 EXCLUDES singles and is shown against
    the other row for reference. The gap between them is the single fraction, reported below.
    """
    # calibrate_default_poisson has no import-time side effects (its only raise is inside a
    # function) -- run_powerlaw_nogate.py and run_default_meandeg1p4.py import from it the same way
    from calibrate_default_poisson import (
        TARGETS, _ActiveTracker, degree_from_edges, union_edges_window, active_union_window,
        pooled_mean_degree_excl_singles,
    )

    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'{TAG}_network_diagnostic.txt'

    # verbose > 0 so CommunityNetworkBackend prints its own calibration trace, which is half the
    # point of this mode; 0.1 means a progress line every 10 timesteps
    pars = make_pars(DIAG_SEED, n_agents=DIAG_AGENTS, end=DIAG_END, interventions=[],
                     analyzers=[hpv.network_history(), _ActiveTracker()], verbose=0.1)
    describe(pars)
    print(f'\nDiagnostic run: seed {DIAG_SEED}, {DIAG_AGENTS:,} agents, {START}-{DIAG_END}, '
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
    # Read off pars, not assumed: --diagnose is also how the 1900 start gets compared against a
    # normal 1980-start control (START=1980 DIAG_END=2010 ...), and that control has migration ON
    migration_on = bool(pars.get('use_migration', True))
    mig_note = 'migration on' if migration_on else 'migration off'

    def row(label, value, target=None, note=''):
        tgt = f'  target {target:>6.3f}' if target is not None else ' ' * 14
        return f'  {label:<38} {value:>8.3f}{tgt}   {note}'

    lines = [
        f'{TAG}  --  network diagnostic',
        '',
        f"  gamma_shape={cp['gamma_shape']}  n_communities={cp['n_communities']} (no communities)  "
        f"mean_partners_per_year={target_incl} (including singles)",
        f"  frac_long={cp['frac_long']}  D_short={cp['D_mean_short']} mo  "
        f"D_long={cp['D_mean_long']} mo  p_single_annual={cp.get('p_single_annual', 0.0)}",
        f'  seed {DIAG_SEED}, {DIAG_AGENTS:,} agents, {pars["start"]}-{DIAG_END}, {mig_note}, '
        f'measured at the final timestep',
        '',
        '  Annual distinct partners',
        row('mean, EXCLUDING singles', mean_deg_excl, TARGETS['mean_degree_annual'],
            'Natsal convention; not what is fitted'),
        row('mean, INCLUDING singles', mean_deg_incl, target_incl,
            "community_pars convention  <-- FITTED"),
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
        '  The FITTED row is the one acbnm.calibrate() solved for, on its 2,500-node toy model',
        '  without debut/mortality/migration. If it is far off here, in the real sim, re-tune',
        "  community_pars['mean_partners_per_year'] in basePars_community.py and re-run this",
        '  diagnostic before starting the 50 production runs. A large gap between the two annual',
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
    other would silently put this run on a different initial prevalence from its default-network
    sibling. run_powerlaw_nogate.py has to override its base file's copy for exactly that reason.
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
    few minutes, then the column invariants checked on the CSV as read back off disk.
    """
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f'{TAG}_selftest.csv'
    if csv_path.exists():
        csv_path.unlink()  # a self-test always starts clean, unlike the production run

    pars = make_pars(0, n_agents=SELFTEST_AGENTS, end=SELFTEST_END)
    describe(pars)
    print(f'\nSelf-test: {SELFTEST_RUNS} runs, {SELFTEST_AGENTS:,} agents, {START}-{SELFTEST_END}',
          flush=True)

    run_batches(csv_path, [0], SELFTEST_RUNS, min(SELFTEST_RUNS, N_CPUS), resume=False,
                label=f'{TAG} self-test', n_agents=SELFTEST_AGENTS, end=SELFTEST_END)

    df = pd.read_csv(csv_path, index_col=0)
    gt = [str(g) for g in pars['genotypes']]
    gen_cols = [c for c in df.columns if any(c.endswith(f'_{g}') for g in gt)]
    cp = pars['community_pars']

    tol = 1e-9
    n_alive = df['n_alive'].to_numpy()
    checks = [
        ('network is the community one, Gamma propensity (powerlaw not imported)',
         pars['network'] == 'community' and 'powerlaw' not in sys.modules),
        (f'gamma_shape reached the sim as {GAMMA_SHAPE}',
         float(cp['gamma_shape']) == float(GAMMA_SHAPE)),
        ('no communities: n_communities == 1 and the mixing kernel is 1x1',
         int(cp['n_communities']) == 1
         and np.asarray(cp['community_mixing']).shape == (1, 1)
         and np.asarray(cp['community_probs']).shape == (1,)),
        ('by-community results are off',
         not pars.get('community_results', False)),
        ('init_hpv_prev matches basePars.py (the current arrays)',
         init_hpv_prev_matches_basepars()),
        (f'pre-{DATA_START_YEAR} start: age_datafile set and migration off',
         pars['start'] >= DATA_START_YEAR
         or (pars.get('age_datafile') == str(AGE_DATAFILE) and pars['use_migration'] is False)),
        ('index is years, single header row',
         pd.api.types.is_numeric_dtype(df.index) and float(df.index.min()) == float(START)),
        (f'{SELFTEST_RUNS} seeds present',
         df['Seed'].nunique() == SELFTEST_RUNS),
        ('infections_star and prevalence_star present',
         {'infections_star', 'prevalence_star'}.issubset(df.columns)),
        (f'{N_GENOTYPE_RESULTS * len(gt)} by-genotype columns '
         f'({len(gt)} genotypes x {N_GENOTYPE_RESULTS} results)',
         len(gen_cols) == N_GENOTYPE_RESULTS * len(gt)),
        ('prevalence_star == infections_star / n_alive',
         bool(np.allclose(df['prevalence_star'], df['infections_star'] / n_alive, atol=tol))),
        ('prevalence_star >= hpv_prevalence everywhere',
         bool((df['prevalence_star'] >= df['hpv_prevalence'] - tol).all())),
        ('hpv_prevalence == n_infectious / n_alive',
         bool(np.allclose(df['hpv_prevalence'], df['n_infectious'] / n_alive, atol=tol))),
        ('every hpv_prevalence_<g> <= hpv_prevalence',
         bool(all((df[f'hpv_prevalence_{g}'] <= df['hpv_prevalence'] + tol).all() for g in gt))),
        ('infections == sum over genotypes of infections_<g>',
         bool(np.allclose(df['infections'], sum(df[f'infections_{g}'] for g in gt), rtol=1e-6))),
        ('screening is live after 1980',
         bool(float(df.loc[df.index >= 1981, 'cum_screens'].max()) > 0)),
        ('vaccination is live after 2008',
         bool(len(df.loc[df.index >= 2009]) > 0
              and float(df.loc[df.index >= 2009, 'cum_doses'].max()) > 0
              and float(df.loc[df.index >= 2009, 'cum_vaccinated'].max()) > 0)),
    ]

    print()
    for name, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}')
    n_fail = sum(1 for _, ok in checks if not ok)
    print(f'\n{len(checks) - n_fail}/{len(checks)} checks passed.')
    print(f'Wrote {csv_path}  ({len(df.columns)} columns, {len(df)} rows).')
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

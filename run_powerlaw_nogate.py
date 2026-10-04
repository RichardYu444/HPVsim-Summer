"""
run_powerlaw_nogate.py
======================

50 full runs of the single-community power-law network with three changes to the calibrated
config in basePars_community_powerlaw.py:

    Pareto alpha (community_pars['gamma_shape'])   2.05  ->  3.5    lighter tail
    annual singleness gate (p_single_annual)       0.20  ->  0.0    off entirely
    partner target (mean_partners_per_year)        1.5   ->  1.4

run over 1950-2070 at 100,000 agents, under the existing NON-TARGETED NHS vaccination programme
(NHS_Vacc.vaccinations -- not core_vacc, and no theta_predraw patch).

Shaped on run_core_vacc.py -- MultiSim in batches of 5 across 10 base seeds, one stacked CSV with
a Seed column and a single header row, resumable -- but with pars built from
basePars_community_powerlaw rather than powerlaw.make_sim(), and a wider export (below).

WHAT THIS CAPTURES THAT EARLIER RUNS DID NOT
--------------------------------------------
1. Per-genotype dynamics. Sim.to_df() keeps only 1-D results, so every '*_by_genotype' result was
   silently dropped from previous CSVs. export_df() below adds all of them back, one column per
   result per genotype (e.g. hpv_prevalence_hpv16, infections_ohr, n_cin_hpv18).

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
   Overall HPV only, as asked; the per-genotype n_infected_<g> columns are there too if the
   split is ever wanted.

WHY NOTHING ELSE IS EDITED
--------------------------
r0_hpv.py, validate_r0.py, run_sim.py, degree_timeseries.py and plot_powerlaw_network_stats.py all
import basePars_community_powerlaw and must keep their current behaviour, so every change above is
layered onto a deep copy at run time instead.

One exception worth knowing about: basePars_community_powerlaw.py carries its OWN copy of
init_hpv_prev, which is now stale -- basePars.py's has since been edited (the f/m arrays swapped
and m revalued) and is the one to use for the power law too. make_pars() overrides it from
basePars for this run only; the other scripts above still use the stale copy.

The imports of basePars_community_powerlaw (and, through it, powerlaw) are at MODULE level and
must stay there: the Pareto theta sampler is installed as an import side effect, and sciris
parallelize spawns workers on Windows that re-import __main__. An import tucked inside main()
would leave every worker unpatched and silently running the Gamma default.

THREE MODES
-----------
    python run_powerlaw_nogate.py --selftest    ~2 min   pipeline + column checks, small sim
    python run_powerlaw_nogate.py --diagnose    ~10 min  realised network structure, 1 seed
    python run_powerlaw_nogate.py               ~4-8 h   the 50 production runs

Run --selftest and --diagnose first: --diagnose reports the realised network structure, which is
worth reading before spending hours rather than after.

WHICH MEAN DEGREE 1.4 IS
------------------------
"Mean degree 1.4" is two different measurements. The backend's mean_partners_per_year has always
been fitted to the mean annual distinct-partner count over everyone active, INCLUDING people with
zero partners; the project's Natsal target (calibrate_default_poisson.TARGETS['mean_degree_annual'])
EXCLUDES singles. They differ by the single fraction, which itself moves with connectivity, so one
cannot be converted into the other after the fact. With the singleness gate off nothing pins that
fraction any more, and the two diverge badly.

This run therefore fits the EXCLUDING-singles figure, via the opt-in exclude_singles argument
added to age_community_bipartite_network_model.calibrate() and passed down through
community_pars['calibrate_kwargs']. That argument defaults to False, so every other script in the
repo keeps the old convention untouched (verified bit-identical). Set EXCLUDE_SINGLES=0 in the
environment to run this script the old way instead.

Be aware of what the switch costs: holding the excluding-singles mean at 1.4 rather than the
including-singles mean gives a markedly SPARSER network -- on the toy calibration model, rho lands
at ~0.22 of its analytic value rather than ~0.52, with ~60% of active people holding no partner in
a given year rather than ~30%. That is arithmetic, not a bug: 1.4 partners among the partnered,
with most people unpartnered, is far fewer partnerships than 1.4 partners averaged over everyone.
--diagnose measures what the real sim (with debut, mortality and migration) actually produces.

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

import basePars                            # for the current init_hpv_prev (see module docstring)
import basePars_community_powerlaw as bpw  # imports powerlaw -> installs the Pareto sampler
import hpvsim_working as hpv
import NHS_2025_lambdamu
import NHS_Vacc


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

# Environment-set rather than argv-derived so spawned workers inherit them (see run_core_vacc.py)
ALPHA = float(os.environ.get('ALPHA', 3.5))                        # Pareto tail index; must be > 2
P_SINGLE_ANNUAL = float(os.environ.get('P_SINGLE_ANNUAL', 0.0))    # 0.0 = gate off entirely
MEAN_PARTNERS_PER_YEAR = float(os.environ.get('MEAN_PARTNERS_PER_YEAR', 1.4))

# Which convention MEAN_PARTNERS_PER_YEAR is fitted to -- see the module docstring and
# age_community_bipartite_network_model.calibrate()'s exclude_singles argument. True (the default
# here) targets the Natsal "mean annual partners among people who have at least one" figure, which
# is what the project's 1.4 means; EXCLUDE_SINGLES=0 reverts to HPVsim's historical
# including-singles mean. The excluding-singles root-find is the harder of the two, so it gets
# more iterations than calibrate()'s own default of 6; each one costs a couple of seconds, once,
# at sim init.
EXCLUDE_SINGLES = os.environ.get('EXCLUDE_SINGLES', '1').lower() not in ('0', 'false', 'no')
CALIBRATE_MAX_ITERS = int(os.environ.get('CALIBRATE_MAX_ITERS', 8))

N_AGENTS = int(os.environ.get('N_AGENTS', 100_000))
START = int(os.environ.get('START', 1950))
END = int(os.environ.get('END', 2070))

TAG = os.environ.get('TAG', 'powerlaw_alpha3p5_nogate_100k_1950_2070_50runs')
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

N_RUNS = 5   # due to multisim stuff I think 5 is max I can run on a 6 core cpu
N_CPUS = 5

SEEDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 seeds gets us to 5 * 10 = 50 total runs (0-49)

# Number of leading seed batches to run, N_RUNS each. 10 (the default) is the full 50 runs; set it
# lower for a quick look. Because RESUME skips whatever is already in the CSV, a short run can be
# extended to the full 50 later just by re-running with a larger value -- only the missing batches
# are computed.
#
#     N_BATCHES=2 python run_powerlaw_nogate.py
SEEDS = SEEDS[:int(os.environ.get('N_BATCHES', len(SEEDS)))]

# If the output CSV already exists, continue from where a previous attempt stopped rather than
# refusing to start. These runs take hours and the batches are written incrementally, so an
# interruption (machine sleeping, session ending, Ctrl-C) otherwise throws away everything done so
# far. Seeds already present in the CSV are skipped and never rewritten, so resuming cannot
# duplicate or interleave rows. Set False to get a hard failure on an existing file instead.
RESUME = True

# --diagnose settings. The epidemic is irrelevant to network structure, so this stops well before
# 2070; interventions are dropped because vaccination's 2008 start would then fall outside the sim,
# which HPVsim rejects. 30 years is more burn-in than calibrate_default_poisson's own 20.
DIAG_SEED = int(os.environ.get('DIAG_SEED', 0))
DIAG_AGENTS = int(os.environ.get('DIAG_AGENTS', N_AGENTS))
DIAG_END = int(os.environ.get('DIAG_END', 1980))

# --selftest settings: small and short, but through exactly the same code path as the real run.
SELFTEST_AGENTS = int(os.environ.get('SELFTEST_AGENTS', 20_000))
SELFTEST_END = int(os.environ.get('SELFTEST_END', 2030))
SELFTEST_RUNS = int(os.environ.get('SELFTEST_RUNS', 2))

LKEY_SHORT, LKEY_LONG = 's', 'l'  # community network's layer keys (<-> default's casual/marital)

N_GENOTYPE_RESULTS = 21  # how many '*_by_genotype' results HPVsim stores; checked by --selftest


# -------------------------------------------------------------------
# parameters
# -------------------------------------------------------------------

def make_pars(seed, **overrides):
    """
    basePars_community_powerlaw's calibrated pars with this run's three network changes, the
    current init_hpv_prev, and the 1950-2070 / 100k / non-targeted-vaccination setup.

    ``overrides`` win over everything and are how --diagnose and --selftest shrink the run without
    touching the network configuration being measured.
    """
    pars = sc.dcp(bpw.base_pars_geno)

    # Deep-copied so nothing here can reach back into the module-level dict that r0_hpv.py and
    # friends read. (CommunityNetworkBackend does take its own copy before popping keys off it,
    # but relying on that from the outside is not worth the saved microsecond.)
    pars['community_pars'] = sc.dcp(sc.mergedicts(bpw.community_pars, dict(
        gamma_shape=ALPHA,
        p_single_annual=P_SINGLE_ANNUAL,
        mean_partners_per_year=MEAN_PARTNERS_PER_YEAR,
        # CommunityNetworkBackend pops this and forwards it straight to acbnm.calibrate(), so the
        # calibration convention is switchable from here without touching community_network.py
        calibrate_kwargs=dict(exclude_singles=EXCLUDE_SINGLES, max_iters=CALIBRATE_MAX_ITERS),
    )))

    # basePars_community_powerlaw's own copy is stale -- see the module docstring
    pars['init_hpv_prev'] = sc.dcp(basePars.base_pars['init_hpv_prev'])

    pars.update(
        n_agents=N_AGENTS,
        start=START,
        end=END,
        rand_seed=seed,
        # "Old", non-targeted vaccination: the NHS programme as modelled today. dcp'd because the
        # interventions are module-level objects shared with every other importer of NHS_Vacc.
        interventions=NHS_2025_lambdamu.get_interventions(l=1, m=1) + sc.dcp(NHS_Vacc.vaccinations),
        # basePars_community_powerlaw sets analyzers=[hpv.network_history()], which keeps a
        # NetworkDelta for every timestep of every run. Nothing here reads it, and at 100k agents
        # x 50 runs it is a lot of memory for nothing -- same reasoning as run_core_vacc.py's
        # DROP_NETWORK_HISTORY. --diagnose puts it back, for one seed.
        analyzers=[],
        verbose=-1,
    )
    pars.update(overrides)
    return pars


def describe(pars):
    """ One block at startup, so a CSV can always be traced back to what produced it """
    cp = pars['community_pars']
    ihp = pars['init_hpv_prev']
    a = float(cp['gamma_shape'])
    theta_cv = 1 / np.sqrt(a * (a - 2)) if a > 2 else float('nan')  # Pareto CV -- see powerlaw.py
    gate = 'gate OFF' if not cp['p_single_annual'] else 'gate on'
    print('Configuration:')
    print(f"  network            {pars['network']}, n_communities={cp['n_communities']} (no communities)")
    print(f"  Pareto alpha       {a}  (theta CV = {theta_cv:.3f})")
    print(f"  p_single_annual    {cp['p_single_annual']}  ({gate})")
    excl = cp.get('calibrate_kwargs', {}).get('exclude_singles', False)
    print(f"  partners/yr target {cp['mean_partners_per_year']}  "
          f"({'EXCLUDING' if excl else 'including'} singles)")
    print(f"  frac_long          {cp['frac_long']}   D_short={cp['D_mean_short']} mo, "
          f"D_long={cp['D_mean_long']} mo")
    print(f"  years              {pars['start']}-{pars['end']}   n_agents={pars['n_agents']:,}")
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

    Why this exists even now that calibrate() targets the right convention (EXCLUDE_SINGLES, see
    the module docstring): acbnm.calibrate() fits rho on a 2,500-node TOY model with no sexual
    debut, no mortality and no migration, and it measures over a single 12-month window. The real
    sim has all three, plus CommunityNetworkBackend's mortality-aware inflation of q_short/q_long.
    So hitting the target on the toy model is necessary but not sufficient -- this measures what
    the actual simulated network produces, using the same windowed-union machinery the project's
    own calibration harnesses use.

    Both conventions are reported, with the one being fitted marked, so the gap between them (the
    single fraction) is visible rather than implied. If the fitted figure lands far from its
    target here, that is the moment to re-tune mean_partners_per_year -- before the 50 runs, not
    after them.
    """
    # calibrate_default_poisson has no import-time side effects (its only raise is inside a
    # function) -- run_powerlaw_validation.py imports from it the same way
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

    def row(label, value, target=None, note=''):
        tgt = f'  target {target:>6.3f}' if target is not None else ' ' * 14
        return f'  {label:<38} {value:>8.3f}{tgt}   {note}'

    fitted = '<-- FITTED'  # marks whichever convention calibrate() was told to target
    lines = [
        f'{TAG}  --  network diagnostic',
        '',
        f'  alpha={ALPHA}  p_single_annual={P_SINGLE_ANNUAL}  '
        f'mean_partners_per_year={MEAN_PARTNERS_PER_YEAR} '
        f'({"EXCLUDING" if EXCLUDE_SINGLES else "including"} singles)',
        f'  seed {DIAG_SEED}, {DIAG_AGENTS:,} agents, {START}-{DIAG_END}, '
        f'measured at the final timestep',
        '',
        '  Annual distinct partners',
        row('mean, EXCLUDING singles', mean_deg_excl, TARGETS['mean_degree_annual'],
            f'Natsal convention  {fitted if EXCLUDE_SINGLES else ""}'),
        row('mean, INCLUDING singles', mean_deg_incl,
            MEAN_PARTNERS_PER_YEAR if not EXCLUDE_SINGLES else None,
            f'{"" if EXCLUDE_SINGLES else fitted}'),
        row('CV among the partnered', cv_annual, 1.831,
            'Natsal cv_degree_annual; falls as alpha rises'),
        row('fraction with 0 partners in the year', frac_zero_in_year),
        '',
        '  5-year distinct partners',
        row('mean, EXCLUDING singles', mean_deg_5yr, TARGETS['mean_degree_5yr']),
        '',
        '  Instantaneous snapshot',
        row('p_single', p_single, TARGETS['p_single'], 'no longer pinned -- the gate is off'),
        row('p_long', p_long, TARGETS['p_long']),
        row('p_short', p_short, TARGETS['p_short']),
        '',
        '  The FITTED row is the one acbnm.calibrate() solved for, on its 2,500-node toy model',
        '  without debut/mortality/migration. If it is far off here, in the real sim, re-tune',
        "  community_pars['mean_partners_per_year'] (MEAN_PARTNERS_PER_YEAR at the top of this",
        '  file, or the env var of the same name) and re-run this diagnostic before starting the',
        '  50 production runs. A large gap between the two rows is expected, not a fault: it is',
        '  just the single fraction reported two lines below.',
    ]
    text = '\n'.join(lines) + '\n'
    out_path.write_text(text, encoding='utf-8')
    print()
    print(text)
    print(f'Wrote {out_path}')
    return


# -------------------------------------------------------------------
# --selftest: does the pipeline produce what it claims to?
# -------------------------------------------------------------------

def selftest():
    """
    The real code path (make_pars -> MultiSim -> export_df -> CSV) at a size that finishes in a
    couple of minutes, then the column invariants checked on the CSV as read back off disk.
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

    tol = 1e-9
    n_alive = df['n_alive'].to_numpy()
    checks = [
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

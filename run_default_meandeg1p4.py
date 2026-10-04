"""
run_default_meandeg1p4.py
=========================

50 full runs of HPVsim's DEFAULT ('default') network -- basePars.py's calibrated configuration,
unchanged -- over 1950-2070 at 100,000 agents, under the existing NON-TARGETED NHS vaccination
programme (NHS_Vacc.vaccinations), with the same wide per-genotype export as
run_powerlaw_nogate.py.

This is the default-network companion to

    csvs/powerlaw_alpha3p5_nogate_100k_1950_2070_50runs.csv

and is deliberately matched to it: same years, same agent count, same 10 x 5 seed batches, same
interventions, same exported columns. The only thing that differs is the network.

WHICH MEAN DEGREE 1.4 IS
------------------------
"Mean degree 1.4" here is the Natsal convention -- the mean number of DISTINCT partners over a
one-year window, among people who had at least one (singles EXCLUDED). That is exactly
calibrate_default_poisson.TARGETS['mean_degree_annual'], and basePars.py's six partnership knobs

    m_scale             0.3       layer_probs['m'] scale   -> p_long
    c_scale            12         layer_probs['c'] scale   -> p_short
    m_partners_c_par1   0.5867    m_partners['c'] poisson1 -> annual mean degree
    f_partners_c_par1   1.17      f_partners['c'] poisson  -> annual mean degree
    dur_pship_c_par1    0.75      casual duration          -> 5-year mean degree
    dur_pship_m_par1   60         marital duration         -> 5-year mean degree

are that calibration's output. So NOTHING has to be changed to get mean degree 1.4 on the default
network: basePars.py is already there, and it is already the same convention the power-law nogate
run was fitted to (its EXCLUDE_SINGLES=1 path). The two runs are therefore on the same footing.

The knobs are nevertheless rebuilt here from named module-level values rather than taken silently
off basePars.base_pars_geno, so that the figure this run is named after is visible in describe()'s
output and can be re-tuned from the environment without editing basePars.py (which a dozen other
scripts import). With the knobs at their basePars values the rebuild reproduces basePars' arrays
bit-for-bit -- checked by --selftest.

Worth knowing: calibrate_default_poisson fitted those knobs at 200k agents, start 2000, a 20-year
burn-in and a single measurement window, on seed 1. This run is 100k agents from 1950. Hitting the
target there is necessary but not sufficient, so --diagnose measures what THIS configuration
actually produces before the hours are spent.

WHAT THIS CAPTURES THAT EARLIER DEFAULT-NETWORK RUNS DID NOT
------------------------------------------------------------
Identical to run_powerlaw_nogate.py, and for the same reasons:

1. Per-genotype dynamics. Sim.to_df() keeps only 1-D results, so every '*_by_genotype' result was
   silently dropped from csvs/default.csv and csvs/defaultinter.csv. export_df() below adds all of
   them back, one column per result per genotype (e.g. hpv_prevalence_hpv16, infections_ohr,
   n_cin_hpv18).

2. infections* / prevalence*, exported as the columns `infections_star` and `prevalence_star`:

       infections_star = people.infected.any(axis=0), counted once per person, scale-weighted
       prevalence_star = infections_star / n_alive

   People.infected is a property (base.py) = infectious | inactive, i.e. currently infectious OR
   carrying an inactive/latent infection, including people with cancer. It is a strict superset of
   `infectious`, so prevalence_star >= hpv_prevalence always. The sim already stores
   people.count_any('infected') as the 1-D result n_infected, so infections_star is that result
   under a clearer name; prevalence_star is formed exactly the way hpv_prevalence is
   (sim.py: safedivide(n_infectious, n_alive)), and shares its denominator -- remove_people()
   zeroes infectious/inactive on death, so neither numerator carries the dead.

WHY NOTHING IS EDITED OUTSIDE THIS FILE
---------------------------------------
basePars.py is imported by roughly a dozen scripts (run_sim.py, r0_hpv.py, plot_* and the NHS_*
analyses) and must keep its current behaviour, so the 1950/2070/100k/no-analyzer changes are
layered onto a deep copy at run time. basePars' own init_hpv_prev is the current one -- it is the
copy run_powerlaw_nogate.py had to borrow -- so unlike that script there is nothing to patch here.

Imports stay at MODULE level: sciris parallelize spawns workers on Windows that re-import
__main__, and an import tucked inside main() would be re-executed per worker for no benefit.

THREE MODES
-----------
    python run_default_meandeg1p4.py --selftest    ~2 min   pipeline + column checks, small sim
    python run_default_meandeg1p4.py --diagnose    ~10 min  realised network structure, 1 seed
    python run_default_meandeg1p4.py               ~4-8 h   the 50 production runs

Run --selftest and --diagnose first: --diagnose reports the realised mean degree, which is worth
reading before spending hours rather than after.

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

import basePars                  # the default network's calibrated configuration
import hpvsim_working as hpv
from hpvsim_working import parameters as hppar
import NHS_2025_lambdamu
import NHS_Vacc


# -------------------------------------------------------------------
# adjustable settings
# -------------------------------------------------------------------

# Environment-set rather than argv-derived so spawned workers inherit them (see run_core_vacc.py)
N_AGENTS = int(os.environ.get('N_AGENTS', 100_000))
START = int(os.environ.get('START', 1950))
END = int(os.environ.get('END', 2070))

# The six partnership knobs calibrate_default_poisson.py fits, defaulting to the values basePars.py
# already carries -- i.e. mean degree 1.4, annual distinct partners excluding singles. Override any
# of them from the environment to re-tune without touching basePars.py; --diagnose then reports what
# the override actually produced.
KNOBS = dict(
    m_scale=float(os.environ.get('M_SCALE', basePars.m_scale)),
    c_scale=float(os.environ.get('C_SCALE', basePars.c_scale)),
    m_partners_c_par1=float(os.environ.get('M_PARTNERS_C_PAR1', basePars.m_partners_c_par1)),
    f_partners_c_par1=float(os.environ.get('F_PARTNERS_C_PAR1', basePars.f_partners_c_par1)),
    dur_pship_c_par1=float(os.environ.get('DUR_PSHIP_C_PAR1', basePars.dur_pship_c_par1)),
    dur_pship_m_par1=float(os.environ.get('DUR_PSHIP_M_PAR1', basePars.dur_pship_m_par1)),
)

# Package defaults for the two dur_pship spread parameters; basePars.py uses these and
# calibrate_default_poisson.py never moves them (see parameters.py layer_defaults['default']).
DUR_PSHIP_C_PAR2 = 2.0
DUR_PSHIP_M_PAR2 = 3.0

# What the knobs above are claiming, reported by describe(). This is
# calibrate_default_poisson.TARGETS['mean_degree_annual'] -- kept as a literal so this file states
# the figure it is named after rather than importing it from a calibration harness.
MEAN_DEGREE_TARGET = float(os.environ.get('MEAN_DEGREE_TARGET', 1.4))

TAG = os.environ.get('TAG', 'default_meandeg1p4_100k_1950_2070_50runs')
OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs'

N_RUNS = 5   # due to multisim stuff I think 5 is max I can run on a 6 core cpu
N_CPUS = 5

SEEDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 seeds gets us to 5 * 10 = 50 total runs (0-49)

# Number of leading seed batches to run, N_RUNS each. 10 (the default) is the full 50 runs; set it
# lower for a quick look. Because RESUME skips whatever is already in the CSV, a short run can be
# extended to the full 50 later just by re-running with a larger value -- only the missing batches
# are computed.
#
#     N_BATCHES=2 python run_default_meandeg1p4.py
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

LKEY_CASUAL, LKEY_MARITAL = 'c', 'm'  # default network's layer keys (<-> community's 's'/'l')

N_GENOTYPE_RESULTS = 21  # how many '*_by_genotype' results HPVsim stores; checked by --selftest


# -------------------------------------------------------------------
# parameters
# -------------------------------------------------------------------

def partnership_pars(knobs):
    """
    basePars.py's own recipe for the six calibrated knobs, rebuilt from `knobs` so the mean-degree
    configuration is explicit here and overridable from the environment.

    With `knobs` at their basePars values this reproduces basePars' arrays exactly -- --selftest
    asserts that, so a drift between this recipe and basePars.py's cannot pass silently.

    reset_layer_pars() merges dict-of-layer-keys parameters key by key (see parameters.py), so
    passing only the 'c' entry for m_partners/f_partners leaves the marital-layer 'm' entry at its
    package default, exactly as basePars.py does.
    """
    _default_mixing, default_layer_probs = hppar.get_mixing('default')
    lp_m = sc.dcp(default_layer_probs['m'])
    lp_c = sc.dcp(default_layer_probs['c'])
    # Row 0 is the age-bracket header, hence the [1:, :] -- the same slice basePars.py uses
    lp_m[1:, :] = np.clip(lp_m[1:, :] * knobs['m_scale'], 0.0, 1.0)
    lp_c[1:, :] = np.clip(lp_c[1:, :] * knobs['c_scale'], 0.0, 1.0)
    return dict(
        layer_probs=dict(m=lp_m, c=lp_c),
        m_partners=dict(c=dict(dist='poisson1', par1=knobs['m_partners_c_par1'])),
        f_partners=dict(c=dict(dist='poisson', par1=knobs['f_partners_c_par1'])),
        dur_pship=dict(
            m=dict(dist='neg_binomial', par1=knobs['dur_pship_m_par1'], par2=DUR_PSHIP_M_PAR2),
            c=dict(dist='lognormal', par1=knobs['dur_pship_c_par1'], par2=DUR_PSHIP_C_PAR2),
        ),
    )


def make_pars(seed, **overrides):
    """
    basePars.py's calibrated default-network pars with the mean-degree-1.4 knobs made explicit and
    the 1950-2070 / 100k / non-targeted-vaccination setup.

    ``overrides`` win over everything and are how --diagnose and --selftest shrink the run without
    touching the network configuration being measured.
    """
    # Deep-copied so nothing here can reach back into the module-level dict that run_sim.py,
    # r0_hpv.py and the NHS_* analyses read
    pars = sc.dcp(basePars.base_pars_geno)

    pars.update(partnership_pars(KNOBS))

    pars.update(
        n_agents=N_AGENTS,
        start=START,
        end=END,
        rand_seed=seed,
        # "Old", non-targeted vaccination: the NHS programme as modelled today. Rebuilt/dcp'd
        # because basePars' list holds the module-level intervention objects shared with every
        # other importer of NHS_Vacc.
        interventions=NHS_2025_lambdamu.get_interventions(l=1, m=1) + sc.dcp(NHS_Vacc.vaccinations),
        # basePars sets analyzers=[hpv.network_history()], which keeps a NetworkDelta for every
        # timestep of every run. Nothing here reads it, and at 100k agents x 50 runs it is a lot of
        # memory for nothing -- same reasoning as run_core_vacc.py's DROP_NETWORK_HISTORY.
        # --diagnose puts it back, for one seed.
        analyzers=[],
        verbose=-1,
    )
    pars.update(overrides)
    return pars


def describe(pars):
    """ One block at startup, so a CSV can always be traced back to what produced it """
    ihp = pars['init_hpv_prev']
    dp_c, dp_m = pars['dur_pship']['c'], pars['dur_pship']['m']
    print('Configuration:')
    print(f"  network            {pars['network']}  (HPVsim built-in, basePars.py)")
    print(f"  mean degree        {MEAN_DEGREE_TARGET}  (annual distinct partners, EXCLUDING "
          f"singles -- Natsal convention)")
    print(f"  layer_probs scale  m={KNOBS['m_scale']}  c={KNOBS['c_scale']}")
    print(f"  casual partners    m poisson1 par1={KNOBS['m_partners_c_par1']}, "
          f"f poisson par1={KNOBS['f_partners_c_par1']}")
    print(f"  dur_pship          c lognormal par1={dp_c['par1']} yr, "
          f"m neg_binomial par1={dp_m['par1']}")
    print(f"  condoms            m={pars['condoms']['m']}, c={pars['condoms']['c']}")
    print(f"  beta               {pars['beta']}")
    print(f"  cross-layer        f={pars['f_cross_layer']}, m={pars['m_cross_layer']}")
    print(f"  years              {pars['start']}-{pars['end']}   n_agents={pars['n_agents']:,}   "
          f"dt={pars['dt']}")
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

    Why this exists even though basePars.py's knobs are already calibrated: that calibration ran at
    200k agents from 2000 with a 20-year burn-in on seed 1, and measured a single window. This run
    is 100k agents from 1950, with 70 more years of demography behind the measurement. Hitting the
    target on the calibration harness is necessary but not sufficient -- this measures what the
    actual production configuration produces, using the same windowed-union machinery the
    calibration itself used.

    Both conventions are reported, with the fitted one (EXCLUDING singles) marked, so the gap
    between them -- the single fraction -- is visible rather than implied.
    """
    # calibrate_default_poisson has no import-time side effects (its only raise is inside a
    # function) -- run_powerlaw_validation.py and run_powerlaw_nogate.py import from it the same way
    from calibrate_default_poisson import (
        TARGETS, _ActiveTracker, degree_from_edges, union_edges_window, active_union_window,
        pooled_mean_degree_excl_singles,
    )

    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'{TAG}_network_diagnostic.txt'

    # 0.1 means a progress line every 10 timesteps
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
    deg_long = degree_from_edges(edges_now, n, lkey=LKEY_MARITAL)
    deg_short = degree_from_edges(edges_now, n, lkey=LKEY_CASUAL)
    deg_total = deg_long + deg_short  # 'm' and 'c' are disjoint edge sets, so this is exact

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

    lines = [
        f'{TAG}  --  network diagnostic',
        '',
        f'  default network, basePars.py knobs: m_scale={KNOBS["m_scale"]} '
        f'c_scale={KNOBS["c_scale"]} '
        f'm_partners_c={KNOBS["m_partners_c_par1"]} f_partners_c={KNOBS["f_partners_c_par1"]} '
        f'dur_c={KNOBS["dur_pship_c_par1"]} dur_m={KNOBS["dur_pship_m_par1"]}',
        f'  seed {DIAG_SEED}, {DIAG_AGENTS:,} agents, {START}-{DIAG_END}, '
        f'measured at the final timestep',
        '',
        '  Annual distinct partners',
        row('mean, EXCLUDING singles', mean_deg_excl, TARGETS['mean_degree_annual'],
            'Natsal convention  <-- FITTED'),
        row('mean, INCLUDING singles', mean_deg_incl, None,
            'the other convention, for reference'),
        row('CV among the partnered', cv_annual, 1.831, 'Natsal cv_degree_annual'),
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
        '  The FITTED row is the one calibrate_default_poisson.py solved for, at 200k agents from',
        "  2000 with a 20-year burn-in on seed 1. If it is far off here, in this run's actual",
        '  configuration, re-tune m_partners_c_par1 / f_partners_c_par1 (the M_PARTNERS_C_PAR1 and',
        '  F_PARTNERS_C_PAR1 env vars) and re-run this diagnostic before starting the 50 production',
        '  runs. The gap between the two annual rows is just the single fraction reported two lines',
        '  below them, not a fault.',
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

def rebuild_matches_basepars():
    """ partnership_pars() at basePars' own knob values reproduces basePars' arrays exactly """
    built = partnership_pars(dict(
        m_scale=basePars.m_scale,
        c_scale=basePars.c_scale,
        m_partners_c_par1=basePars.m_partners_c_par1,
        f_partners_c_par1=basePars.f_partners_c_par1,
        dur_pship_c_par1=basePars.dur_pship_c_par1,
        dur_pship_m_par1=basePars.dur_pship_m_par1,
    ))
    bp = basePars.base_pars_geno
    return bool(
        np.array_equal(built['layer_probs']['m'], bp['layer_probs']['m'])
        and np.array_equal(built['layer_probs']['c'], bp['layer_probs']['c'])
        and built['m_partners'] == bp['m_partners']
        and built['f_partners'] == bp['f_partners']
        and built['dur_pship'] == bp['dur_pship']
    )


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
        ('partnership pars rebuild matches basePars.py bit-for-bit',
         rebuild_matches_basepars()),
        ('network is the default one',
         pars['network'] == 'default'),
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

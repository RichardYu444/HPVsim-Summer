"""
plot_summary_boxplots.py
========================

Summary box plots of cancer incidence (per 100,000 women) in 2040 against the UK elimination
target of 4, in the same style as DefaultCode/2040plots.py. One box per scenario, one value per
run (seed); whiskers are 1.5 IQR, fliers hidden, green triangle = mean.

    graph1_networks_interventions.png   Poisson vs gamma (3 shapes) vs power law, usual interventions
    graph2_networks_no_interventions.png  same, no interventions (only what has been run)
    graph3_core_group.png                power law with / without core-group vaccination
    graph3b_core_group_paired.png        the same, as the paired per-seed % change vs baseline
    graph4a_ethnicity_equal_uptake.png   4 ethnic communities, usual (national) uptake
    graph4b_ethnicity_observed_uptake.png  4 ethnic communities, ethnicity-based uptake
    graph5_gamma_sweep.png               every gamma shape, usual interventions
    graph6_networks_timeseries.png       graph 1's networks over 1980-2055 (median + IQR)
    graph7_ethnicity_effect.png          observed vs equal uptake, % change with 95% CI

All network runs are single-community unless stated. Scenarios whose CSV is missing are skipped
with a warning rather than plotted empty.

Usage
-----
    python plot_summary_boxplots.py
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

REPO = pathlib.Path(__file__).resolve().parent
CSV_DIR = REPO / 'csvs'
FIG_DIR = REPO / 'figs' / 'Summary'

YEAR_TARGET = 2040
TARGET = 4  # UK elimination target, cancers per 100,000 women
COL = 'cancer_incidence'

# (label, csv). Labels are what appears under each box.
NETWORKS_INTER = [
    ('Poisson',           'defaultinter.csv'),
    ('Gamma\nshape=0.25', 'gammashape_0p25.csv'),
    ('Gamma\nshape=1',    'gammashape_1p0.csv'),
    ('Gamma\nshape=3',    'gammashape3v1inter.csv'),
    ('Power law\nα=2.05', 'powerlawinter.csv'),
]
NETWORKS_NO_INTER = [
    ('Poisson',           'default.csv'),
    ('Gamma\nshape=0.25', None),  # not run without interventions
    ('Gamma\nshape=1',    None),
    ('Gamma\nshape=3',    'gammashape3v1.csv'),
    ('Power law\nα=2.05', None),
]
# Core-group arms: (label, csv, alpha). Arms at the same alpha share seeds.
CORE_ARMS = [
    ('Baseline', 'powerlaw_alpha2p05_200k_baseline_vacc.csv',   '2.05'),
    ('Top 10%',  'powerlaw_alpha2p05_200k_top10_core_vacc.csv', '2.05'),
    ('Top 20%',  'powerlaw_alpha2p05_200k_top20_core_vacc.csv', '2.05'),
    ('Baseline', 'powerlaw_alpha3_200k_50runs_baseline_vacc.csv', '3'),
    ('Top 10%',  'powerlaw_alpha3_200k_50runs_core_vacc.csv',     '3'),
]
ETHNICITIES = ['White', 'Asian', 'Black', 'Chinese']
ETHNICITY_ARMS = [
    ('equal',    'ethnicity_equal_census_200k.csv',
     'Usual (national) screening & vaccination uptake', 'graph4a_ethnicity_equal_uptake.png'),
    ('observed', 'ethnicity_observed_census_200k.csv',
     'Ethnicity-based screening & vaccination uptake', 'graph4b_ethnicity_observed_uptake.png'),
]


def load(name):
    """Read a run CSV, dropping any repeated header rows left by appending runs."""
    if name is None:
        return None
    path = CSV_DIR / name
    if not path.exists():
        print(f'WARNING: {path.name} not found, skipping')
        return None
    df = pd.read_csv(path, low_memory=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    return df[df['year'].notna()]


def at_year(df, col=COL, year=YEAR_TARGET):
    """One value per run at `year`."""
    return pd.to_numeric(df.loc[df['year'] == year, col], errors='coerce').dropna().to_numpy()


def boxplot(ax, samples, labels):
    """The house style from 2040plots.py, with the run count under each label."""
    ax.boxplot(samples,
               tick_labels=[f'{lab}\n(n={len(s)})' for lab, s in zip(labels, samples)],
               showmeans=True,
               showfliers=False)
    target_line = ax.axhline(y=TARGET, color='r', linestyle='-', label=f'{YEAR_TARGET} Target')
    ax.legend(handles=[target_line])
    ax.set_ylabel(f'Cancer incidence in {YEAR_TARGET}')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def ylim_top(samples, pad=2):
    """Top of the axis: just above the highest whisker (fliers are hidden)."""
    tops = []
    for s in samples:
        q1, q3 = np.percentile(s, [25, 75])
        tops.append(s[s <= q3 + 1.5 * (q3 - q1)].max())
    return np.ceil(max(tops) + pad)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches='tight')
    print(f'Saved {FIG_DIR / name}')
    plt.close(fig)


def network_plot(scenarios, title, outfile):
    present = [(lab, at_year(df)) for lab, name in scenarios if (df := load(name)) is not None]
    missing = [lab.replace('\n', ' ') for lab, name in scenarios if name is None]
    if missing:
        print(f'{outfile}: no runs for {", ".join(missing)}')
    labels, samples = zip(*present)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    boxplot(ax, samples, labels)
    ax.set_title(title)
    ax.set_ylim(0, ylim_top(samples))
    save(fig, outfile)


def core_group_plot():
    loaded = [(lab, load(name), alpha) for lab, name, alpha in CORE_ARMS]
    loaded = [(lab, df, alpha) for lab, df, alpha in loaded if df is not None]

    # levels: one box per arm, grouped by alpha
    samples = [at_year(df) for _, df, _ in loaded]
    labels = [f'{lab}\nα={alpha}' for lab, _, alpha in loaded]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    boxplot(ax, samples, labels)
    n205 = sum(alpha == '2.05' for *_, alpha in loaded)
    if 0 < n205 < len(loaded):
        ax.axvline(n205 + 0.5, color='grey', linestyle='--', linewidth=1)
    ax.set_title('Cancers per 100,000 Women - Power Law, Core-Group Vaccination')
    ax.set_ylim(0, ylim_top(samples))
    save(fig, 'graph3_core_group.png')

    # paired: per-seed % change of each core arm vs the baseline at the same alpha. The between-run
    # spread is far bigger than the effect, so the levels plot above cannot show it; differencing
    # within a seed (same network, same demography) removes that spread.
    # Mean incidence over 2035-2055 rather than 2040 alone, as in plot_core_vacc_compare.py:
    # cancer is a rare annual endpoint and a single year is too noisy to show a ~2% shift.
    base = {alpha: df for lab, df, alpha in loaded if lab == 'Baseline'}
    diffs, dlabels = [], []
    for lab, df, alpha in loaded:
        if lab == 'Baseline' or alpha not in base:
            continue
        b = base[alpha]
        pb = b[b['year'].between(2035, 2055)].groupby('Seed')[COL].mean()
        pc = df[df['year'].between(2035, 2055)].groupby('Seed')[COL].mean()
        idx = pb.index.intersection(pc.index)
        diffs.append(((pc[idx] - pb[idx]) / pb[idx] * 100).to_numpy())
        dlabels.append(f'{lab}\nα={alpha}\n(n={len(idx)})')
    if diffs:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.boxplot(diffs, tick_labels=dlabels, showmeans=True, showfliers=False)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_ylabel('% change in cancer incidence vs baseline\n(mean 2035-2055, paired by seed)')
        ax.set_title('Core-Group Vaccination vs Baseline - Power Law')
        save(fig, 'graph3b_core_group_paired.png')


def ethnicity_plots():
    loaded = [(arm, load(name), title, out) for arm, name, title, out in ETHNICITY_ARMS]
    loaded = [x for x in loaded if x[1] is not None]
    per_arm = {arm: [at_year(df, f'{COL}_by_community_{e}') for e in ETHNICITIES]
               for arm, df, *_ in loaded}
    # shared y-axis so the two figures can be compared side by side
    top = ylim_top([s for samples in per_arm.values() for s in samples])
    for arm, df, title, outfile in loaded:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        boxplot(ax, per_arm[arm], ETHNICITIES)
        ax.set_title(f'Cancers per 100,000 Women by Ethnicity\n{title}')
        ax.set_ylim(0, top)
        save(fig, outfile)


GAMMA_SWEEP = [
    ('0.05', 'gammashape_0p05.csv'),
    ('0.25', 'gammashape_0p25.csv'),
    ('1',    'gammashape_1p0.csv'),
    ('2.5',  'gammashape_2p5.csv'),
    ('5',    'gammashape_5p0.csv'),
]


def gamma_sweep_plot():
    """All five sweep shapes, plus the older shape=3 run set drawn apart so the mismatch shows."""
    samples = [at_year(load(name)) for _, name in GAMMA_SWEEP]
    labels = [f'shape={s}' for s, _ in GAMMA_SWEEP]
    s3 = at_year(load('gammashape3v1inter.csv'))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    boxplot(ax, samples + [s3], labels + ['shape=3\n(29 Jul run)'])
    ax.axvline(len(samples) + 0.5, color='grey', linestyle='--', linewidth=1)
    ax.set_xlabel('Gamma shape  (lower = more heterogeneous partner propensity, CV = 1/√shape)')
    ax.set_title('Cancers per 100,000 Women - Gamma Shape Sweep, Usual Interventions')
    ax.set_ylim(0, ylim_top(samples + [s3]))
    save(fig, 'graph5_gamma_sweep.png')


def timeseries_plot():
    """Median and IQR of cancer incidence over time, one line per network, usual interventions."""
    colours = ['tab:grey', 'tab:orange', 'tab:green', 'tab:purple', 'tab:blue']
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (label, name), colour in zip(NETWORKS_INTER, colours):
        df = load(name)
        df[COL] = pd.to_numeric(df[COL], errors='coerce')
        q = df.groupby('year')[COL].quantile([0.25, 0.5, 0.75]).unstack()
        ax.fill_between(q.index, q[0.25], q[0.75], color=colour, alpha=0.15)
        ax.plot(q.index, q[0.5], color=colour, linewidth=2, label=label.replace('\n', ' '))
    ax.axhline(TARGET, color='r', linestyle='-', label='Target (4)')
    ax.axvline(YEAR_TARGET, color='black', linestyle=':', linewidth=1)
    ax.set_xlim(1980, 2055)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('year')
    ax.set_ylabel('Cancer incidence per 100,000 women')
    ax.set_title('Cancer Incidence Over Time by Network - Usual Interventions\n(median, IQR band)')
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(fontsize=8)
    save(fig, 'graph6_networks_timeseries.png')


def ethnicity_effect_plot():
    """
    Observed vs equal uptake, % change with 95% CI, per community. Read straight from
    analyse_ethnicity_uptake.py's output (Welch test, BH-corrected q) rather than recomputed.
    Filled marker = significant after FDR correction (q < 0.05).
    """
    path = CSV_DIR / 'ethnicity_significance_census.csv'
    if not path.exists():
        print(f'WARNING: {path.name} not found, skipping')
        return
    sig = pd.read_csv(path)
    metrics = [('n_vaccinated', 'Vaccinated (2070 stock)'),
               ('hpv_prevalence', 'HPV prevalence (2070)'),
               ('cancers_since_2035', 'Cancers 2035-2070'),
               ('cancer_deaths_since_2035', 'Cancer deaths 2035-2070')]
    groups = ETHNICITIES + ['Overall']
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 3.8), sharey=True)
    for ax, (metric, title) in zip(axes, metrics):
        rows = sig[sig['metric'] == metric].set_index('community').loc[groups]
        y = np.arange(len(groups))[::-1]
        ci = 1.96 * rows['se_pct']
        ax.errorbar(rows['change_pct'], y, xerr=ci, fmt='none', ecolor='tab:blue', capsize=3)
        filled = rows['q'] < 0.05
        ax.scatter(rows['change_pct'][filled], y[filled.to_numpy()], color='tab:blue', zorder=3)
        ax.scatter(rows['change_pct'][~filled], y[~filled.to_numpy()], facecolor='white',
                   edgecolor='tab:blue', zorder=3)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('% change, observed vs equal')
        ax.grid(True, axis='x', linewidth=0.5, alpha=0.4)
    axes[0].set_yticks(np.arange(len(groups))[::-1], groups)
    fig.suptitle('Ethnicity-based uptake vs national uptake (50 runs per arm, 95% CI; '
                 'filled = significant after FDR correction)', fontsize=10)
    save(fig, 'graph7_ethnicity_effect.png')


def print_table():
    """The numbers quoted in the summary: 2040 median [IQR] for every scenario."""
    rows = []
    for tag, scenarios in (('inter', NETWORKS_INTER), ('none', NETWORKS_NO_INTER),
                           ('sweep', [(f'gamma {s}', n) for s, n in GAMMA_SWEEP])):
        for label, name in scenarios:
            df = load(name)
            if df is None:
                continue
            ci, hp = at_year(df), at_year(df, 'hpv_prevalence')
            rows.append(dict(set=tag, scenario=label.replace('\n', ' '), n=len(ci),
                             ci_med=np.median(ci), ci_q1=np.percentile(ci, 25),
                             ci_q3=np.percentile(ci, 75), hpv_prev_med=np.median(hp)))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f'{v:.3f}'))


def main():
    network_plot(NETWORKS_INTER,
                 'Cancers per 100,000 Women - Usual Interventions',
                 'graph1_networks_interventions.png')
    network_plot(NETWORKS_NO_INTER,
                 'Cancers per 100,000 Women - No Interventions',
                 'graph2_networks_no_interventions.png')
    core_group_plot()
    ethnicity_plots()
    gamma_sweep_plot()
    timeseries_plot()
    ethnicity_effect_plot()
    print_table()


if __name__ == '__main__':
    main()

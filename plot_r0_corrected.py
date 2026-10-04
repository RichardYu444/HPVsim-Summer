"""
plot_r0_corrected.py
====================

Every outcome figure for a run_r0_corrected_1900.py run, into figs/R0 corrected/<network>/:

    NETWORK=gamma2      python plot_r0_corrected.py      (the default)
    NETWORK=powerlaw3p5 python plot_r0_corrected.py
    NETWORK=default     python plot_r0_corrected.py

  * plot_IQR.py's standard outcomes (hpv_prevalence, infections, cancer_incidence, n_vaccinated,
    n_cancer_treated) -- the same code as its command line, called in-process;
  * plot_powerlaw_nogate.py's prevalence*/infections* and by-genotype figures, with its code
    (as plot_default_meandeg1p4.py does), narrowed to the tracked genotypes and given more outcomes;
  * the prevalence timelines:
        <TAG>_hpv_prevalence_<g>.png          plot_IQR style, one per genotype
        <TAG>_prevalence_timelines.png        all types + each genotype, median / IQR / seed range,
                                              screening and vaccination marked
        <TAG>_prevalence_indexed_1960_79.png  analyse_equilibrium_1900.py's view: each series over
                                              its own 1960-79 (pre-screening) level

It reads TAG, FIG_DIR, the genotypes and the years from run_r0_corrected_1900 (same NETWORK), so the
two stay in step. The run's CSV must exist.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

import plot_IQR
import plot_powerlaw_nogate as figs  # sets plot_IQR.XLIM_* on import; reset in main()
import run_r0_corrected_1900 as run

CSV = run.CSV_DIR / f'{run.TAG}.csv'
BY_GENOTYPE = ['hpv_prevalence', 'infections', 'hpv_incidence', 'cin_prevalence', 'cancers',
               'cancer_incidence']
SCREENING, VACCINATION = 1980, 2008
BASE = (1960, 1979)
ALL_TYPES = figs.ALL_TYPES
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def series():
    """ (column, label, colour) for all types then each tracked genotype, colours as elsewhere """
    return [('hpv_prevalence', 'All tracked types', ALL_TYPES)] + [
        (f'hpv_prevalence_{g}', *figs.GENOTYPES[g]) for g in run.VAX_GENOTYPES]


def save(fig, name):
    out = run.FIG_DIR / f'{run.TAG}_{name}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved {out}')


def mark_policy(ax, top_text=True):
    for x, lab in ((SCREENING, 'screening'), (VACCINATION, 'vaccination')):
        ax.axvline(x, color=INK2, lw=0.9)
        if top_text:
            ax.text(x, 1.0, f' {lab}', transform=ax.get_xaxis_transform(), color=INK2, fontsize=8,
                    va='top')


def standard(df):
    """ plot_IQR.py's command-line loop, in-process """
    for v in plot_IQR.values:
        if v not in df.columns:
            continue
        fig, ax = plot_IQR.plot_iqr(df, value=v, time='year', title=v)
        fig.tight_layout()
        save(fig, v)


def per_genotype_iqr(df):
    for col, label, color in series()[1:]:
        fig, ax = plt.subplots()
        plot_IQR.draw_iqr(ax, df, col, 'year', color=color, median_color=INK)
        plot_IQR.style_axis(ax, col, 'year')
        ax.set_title(f'{col}  ({label})')
        ax.legend()
        fig.tight_layout()
        save(fig, col)


def timelines(df, n_runs):
    """ Median, IQR and full seed range, all types + each genotype, one panel each """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, facecolor=SURFACE)
    for ax, (col, label, color) in zip(axes.ravel(), series()):
        q = df.groupby('year')[col].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).unstack()
        ax.set_facecolor(SURFACE)
        ax.fill_between(q.index, q[0.0], q[1.0], color=color, alpha=0.10, lw=0, label='range over runs')
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.30, lw=0, label='IQR')
        ax.plot(q.index, q[0.5], color=color, lw=2, label='median')
        mark_policy(ax)
        pre = q.loc[BASE[0]:BASE[1], 0.5].mean()
        ax.set_title(f'{label}   (1960-79 median {pre:.2%}; 2050 {q[0.5].iloc[-1]:.2%})',
                     color=INK, fontsize=11, loc='left')
        ax.set_xlim(run.START, run.END)
        ax.set_ylim(0, None)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
        ax.grid(True, color=GRID, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2)
    for ax in axes[1]:
        ax.set_xlabel('year', color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel('HPV prevalence (whole population)', color=INK2)
    axes[0, 0].legend(fontsize=8, frameon=False, loc='upper right')
    fig.suptitle(f'{run.NET_LABEL}, mean degree {run.MEAN_DEGREE}, R0-corrected (beta {run.BETA:g}) -- '
                 f'HPV prevalence, {n_runs} runs, {run.N_AGENTS:,} agents', color=INK, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, 'prevalence_timelines')


def indexed(df, n_runs):
    """ analyse_equilibrium_1900.py's figure: each series over its own 1960-79 level """
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(0.95, 1.05, color=GRID, alpha=0.8, lw=0)
    ends = []
    for col, label, color in series():
        piv = df.pivot_table(index='year', columns='Seed', values=col)
        base = piv.loc[BASE[0]:BASE[1]].mean()
        rel = piv / base.replace(0, np.nan)
        q = rel.quantile([0.25, 0.5, 0.75], axis=1).T
        ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=0.18, lw=0)
        ax.plot(q.index, q[0.5], color=color, lw=2, label=label)
        ends.append((label, color, float(q[0.5].iloc[-1])))
    mark_policy(ax)
    ax.axhline(1.0, color=INK2, lw=0.6)
    ax.set_xlim(run.START, run.END)
    ax.set_ylim(0, None)
    ax.set_xlabel('year', color=INK2)
    ax.set_ylabel('prevalence / own 1960-79 mean', color=INK2)
    ax.grid(True, color=GRID, lw=0.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2)
    ax.legend(fontsize=9, frameon=False, loc='upper right')
    figs.end_labels(ax, ends)
    ax.set_title(f'{run.NET_LABEL}, R0-corrected: prevalence from a 1900 start, indexed to its 1960-79 '
                 f'pre-screening level\n(median over {n_runs} runs; ribbons = IQR; grey band = +-5%)',
                 color=INK, fontsize=11, loc='left')
    fig.tight_layout()
    save(fig, 'prevalence_indexed_1960_79')


def main():
    run.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CSV)  # 'year' stays a column, which is what plot_IQR / iqr_bands want
    n_runs = df['Seed'].nunique()
    print(f'{CSV.name}: {n_runs} runs, {df["year"].min():.0f}-{df["year"].max():.0f} -> {run.FIG_DIR}')
    plot_IQR.XLIM_LEFT, plot_IQR.XLIM_RIGHT = run.START, run.END

    standard(df)

    figs.TAG, figs.CSV, figs.FIG_DIR = run.TAG, CSV, run.FIG_DIR
    figs.GENOTYPES = {g: figs.GENOTYPES[g] for g in run.VAX_GENOTYPES}
    figs.BY_GENOTYPE = BY_GENOTYPE
    figs.main()

    per_genotype_iqr(df)
    timelines(df, n_runs)
    indexed(df, n_runs)


if __name__ == '__main__':
    main()

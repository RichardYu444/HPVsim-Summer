"""
plot_powerlaw_nogate.py
=======================

The figures for run_powerlaw_nogate.py's 50 runs that plot_IQR.py does not draw: the two columns
that run added (prevalence*, infections*) and the per-genotype results. The standard outcomes come
from plot_IQR.py itself, into the same folder:

    python plot_IQR.py csvs/powerlaw_alpha3p5_nogate_100k_1950_2070_50runs.csv powerlaw_alpha3p5_nogate_100k_1950_2070_50runs "figs/Power Law/alpha3p5_nogate" 2070 1950
    python plot_powerlaw_nogate.py

As in plot_IQR.py, every band is the IQR across runs and every line the median; its iqr_bands,
draw_iqr and style_axis are reused so these figures match the ones beside them.

Writes figs/Power Law/alpha3p5_nogate/<csv stem>_<name>.png for each name below:

    prevalence_star                prevalence* over hpv_prevalence, and the gap between them
    infections_star                infections* -- people currently infected, NOT new infections
    <value>_by_genotype            hpv_prevalence, infections, cancer_incidence, one line per genotype
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

import plot_IQR
from plot_IQR import draw_iqr, iqr_bands, plot_iqr, style_axis

REPO = pathlib.Path(__file__).resolve().parent
TAG = 'powerlaw_alpha3p5_nogate_100k_1950_2070_50runs'
CSV = REPO / 'csvs' / f'{TAG}.csv'
FIG_DIR = REPO / 'figs' / 'Power Law' / 'alpha3p5_nogate'

# style_axis reads these module globals; show the whole run, burn-in included
plot_IQR.XLIM_LEFT, plot_IQR.XLIM_RIGHT = 1950, 2070

# Reference palette (dataviz skill, light mode) in slot order. Slot 1 blue is all types and slot 2
# orange is HPV16, as in figs/equilibrium_1900, so a genotype keeps its colour from figure to figure.
ALL_TYPES = '#2a78d6'
GENOTYPES = {  # CSV column suffix: (label, colour)
    'hpv16': ('HPV16', '#eb6834'),
    'hpv18': ('HPV18', '#1baf7a'),
    'hi5':   ('hi5',   '#eda100'),
    'ohr':   ('ohr',   '#e87ba4'),
}
INK, INK2 = '#0b0b0b', '#52514e'

BY_GENOTYPE = ['hpv_prevalence', 'infections', 'cancer_incidence']


def fit_decimals(ax, comma):
    """
    Fewest decimals that print every y tick exactly. style_axis gives prevalences 2 and everything
    else 0, which would print the prevalence gap (~0.0004) as a column of 0.00s.
    """
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo - 1e-12 <= t <= hi + 1e-12]
    d = next(d for d in range(8) if all(abs(round(t, d) - t) < 1e-9 * max(1, hi) for t in ticks))
    ax.yaxis.set_major_formatter(StrMethodFormatter(f'{{x:{"," if comma else ""}.{d}f}}'))


def end_labels(ax, ends, min_gap=0.055):
    """
    Name each series at the right-hand end of the axis, in ink beside a swatch of its colour -- three
    of the genotype colours are below 3:1 contrast on white, so colour alone would not carry them.
    ends is [(label, colour, final median)]. Labels that would collide are pushed apart, so one can
    sit a little off the end of its line.
    """
    lo, hi = ax.get_ylim()
    items = sorted((min(max((y - lo) / (hi - lo), 0), 1), label, colour) for label, colour, y in ends)
    pos = []
    for frac, _, _ in items:
        pos.append(max(frac, pos[-1] + min_gap) if pos else frac)
    overflow = max(0, pos[-1] - 1)
    for p, (_, label, colour) in zip(pos, items):
        ax.plot([1.01, 1.035], [p - overflow] * 2, transform=ax.transAxes, color=colour, linewidth=2,
                solid_capstyle='round', clip_on=False)
        ax.text(1.045, p - overflow, label, transform=ax.transAxes, va='center', ha='left', fontsize=9,
                color=INK)


def save(fig, name):
    fig.tight_layout()
    out = FIG_DIR / f'{TAG}_{name}.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f'Saved {out}')


def prevalence_star(df):
    """
    prevalence* counts people who are infectious OR inactive (People.infected); hpv_prevalence counts
    the infectious only, over the same denominator. The gap between them is therefore the inactive-
    only share: people with cancer, whose infections go inactive, and latent infections. At ~0.5% of
    prevalence* it is far too thin to see between the two lines, so it gets its own panel and scale.
    """
    d = df.assign(gap=df['prevalence_star'] - df['hpv_prevalence'])  # within each run, then summarised
    fig, (top, bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 2]))

    draw_iqr(top, d, 'prevalence_star', 'year', label_prefix='prevalence* ')
    x, _, med, _ = iqr_bands(d, 'hpv_prevalence', 'year')
    top.plot(x, med, color=ALL_TYPES, linewidth=1.2, label='hpv_prevalence Median')
    style_axis(top, 'prevalence_star', 'year', ylabel='prevalence')
    fit_decimals(top, comma=False)
    top.set_xlabel('')
    top.set_title('prevalence_star vs hpv_prevalence')
    top.legend(fontsize=8)

    draw_iqr(bot, d, 'gap', 'year')
    style_axis(bot, 'gap', 'year', ylabel='prevalence* − hpv_prevalence')
    fit_decimals(bot, comma=False)
    bot.set_title('the gap: inactive-only infections (people with cancer, latent infections)',
                  fontsize=10)
    bot.legend(fontsize=8)
    save(fig, 'prevalence_star')


def infections_star(df):
    fig, ax = plot_iqr(df, 'infections_star', 'year', title='infections_star')
    # A stock, not a flow: set it apart from 'infections', which is new infections per year
    ax.set_ylabel('infections_star  (people currently infected)')
    save(fig, 'infections_star')


def by_genotype(df, value, n_runs):
    """ One axis, every genotype's IQR band and median drawn on it """
    fig, ax = plt.subplots(figsize=(8, 5))
    bands = {g: iqr_bands(df, f'{value}_{g}', 'year') for g in GENOTYPES}
    # All bands before any line, so no band tints another genotype's median
    for g, (x, q25, _, q75) in bands.items():
        ax.fill_between(x, q25, q75, color=GENOTYPES[g][1], alpha=0.2, linewidth=0)
    for g, (x, _, med, _) in bands.items():
        ax.plot(x, med, color=GENOTYPES[g][1], linewidth=2)
    style_axis(ax, value, 'year')
    fit_decimals(ax, comma=not value.endswith('prevalence'))
    ax.set_title(f'{value} by genotype')
    handles = [Line2D([], [], color=c, linewidth=2, label=f'{label} Median')
               for label, c in GENOTYPES.values()]
    handles.append(Patch(facecolor=INK2, alpha=0.2, label=f'IQR across {n_runs} runs'))
    ax.legend(handles=handles, fontsize=8, loc='best')
    end_labels(ax, [(GENOTYPES[g][0], GENOTYPES[g][1], med[-1]) for g, (_, _, med, _) in bands.items()])
    save(fig, f'{value}_by_genotype')


def main():
    df = pd.read_csv(CSV)  # 'year' stays a column, which is what iqr_bands wants as its time axis
    n_runs = df['Seed'].nunique()
    print(f'{CSV.name}: {n_runs} runs, {df["year"].min():.0f}-{df["year"].max():.0f}')
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    prevalence_star(df)
    infections_star(df)
    for v in BY_GENOTYPE:
        by_genotype(df, v, n_runs)


if __name__ == '__main__':
    main()

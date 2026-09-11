"""
plot_ethnicity_uptake.py
========================

Plots for run_ethnicity_uptake.py's two arms.

plot_IQR.py already draws one CSV at a time -- median plus interquartile band across runs, and a
2x2 grid of the same by community. That is the right picture for a single run set, but the
question here is a COMPARISON, so the arms have to share an axis. This reuses plot_IQR's
machinery (iqr_bands, style_axis, is_rate) so the styling matches, and adds:

    <prefix>_<value>_compare.png        both arms overlaid, population-wide
    <prefix>_<value>_by_community.png   2x2, both arms overlaid in each community's panel
    <prefix>_effect_by_community.png    the difference between arms, as a percentage of the
                                        control, with a 95% CI band -- the plot that actually
                                        answers "is anything happening"

The third is the one to look at first. A difference plot whose CI band contains zero across the
whole horizon is a null result, and reading that off one chart is much easier than eyeballing two
near-identical curves in the first two. The band is a normal approximation from the across-run
standard errors of the two arms (unpaired, since the interventions perturb the random stream and
seed 3 in one arm is not seed 3's realisation in the other).

    python plot_ethnicity_uptake.py
    python plot_ethnicity_uptake.py --tag partnership --values cancers hpv_prevalence

Figures go to figs/Ethnicity/ by default.
"""

import argparse
import math
import pathlib

import matplotlib
matplotlib.use('Agg')  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_IQR import iqr_bands, is_rate, style_axis

CSV_DIR = pathlib.Path(__file__).with_name('csvs')
FIG_DIR = pathlib.Path(__file__).with_name('figs') / 'Ethnicity'
SUFFIX = '_by_community_'
ARMS = ('equal', 'observed')
ARM_COLOURS = {'equal': ('tab:blue', '#4c72b0'), 'observed': ('tab:red', '#c44e52')}
ARM_LABELS = {'equal': 'equal uptake (control)', 'observed': 'observed uptake by ethnicity'}

DEFAULT_VALUES = ['hpv_prevalence', 'infections', 'n_infectious', 'n_vaccinated',
                  'cancers', 'n_cancerous', 'cancer_incidence', 'cancer_deaths']

# Effect plots drop the leading years where the control mean is below this fraction of its
# maximum -- the burn-in, where a percentage change is a ratio against ~0. See effect_plot().
BURN_IN_FRAC = 0.05


def find_csv(arm, tag):
    matches = sorted(CSV_DIR.glob(f'ethnicity_{arm}_{tag}_*.csv'))
    if not matches:
        raise FileNotFoundError(f'No CSV for arm {arm!r} with tag {tag!r} in {CSV_DIR}.')
    return matches[-1]


def load(tag):
    frames = {}
    for arm in ARMS:
        path = find_csv(arm, tag)
        df = pd.read_csv(path, index_col=0)
        df = df.reset_index().rename(columns={df.index.name or 'index': 'year'})
        if 'year' not in df.columns:
            df = df.rename(columns={df.columns[0]: 'year'})
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        frames[arm] = df
        print(f'{arm:<9} {path.name}  ({df["Seed"].nunique()} runs)')
    return frames


def communities(df):
    out = []
    for col in df.columns:
        if SUFFIX in col:
            label = col.split(SUFFIX, 1)[1]
            if label not in out:
                out.append(label)
    return out


def draw_arm(ax, df, col, arm, time='year'):
    ''' One arm's median and IQR band, in that arm's colour. '''
    line, fill = ARM_COLOURS[arm]
    x, q25, med, q75 = iqr_bands(df, col, time)
    ax.fill_between(x, q25, q75, color=fill, alpha=0.20, linewidth=0)
    ax.plot(x, med, linewidth=2, color=line, label=ARM_LABELS[arm])
    return x, med


def compare_plot(frames, value, out, time='year'):
    ''' Population-wide, both arms overlaid. '''
    if value not in frames['equal'].columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arm in ARMS:
        draw_arm(ax, frames[arm], value, arm, time)
    style_axis(ax, value, time)
    ax.set_xlim(right=frames['equal'][time].max())
    ax.set_title(f'{value} -- whole population')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return out


def community_plot(frames, value, out, time='year'):
    ''' 2x2 grid, both arms overlaid in each community panel. '''
    labels = communities(frames['equal'])
    cols = {lab: f'{value}{SUFFIX}{lab}' for lab in labels}
    cols = {k: v for k, v in cols.items() if v in frames['equal'].columns}
    if not cols:
        return None
    n = len(cols)
    ncols = 2 if n <= 4 else math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, (label, col) in zip(flat, cols.items()):
        for arm in ARMS:
            draw_arm(ax, frames[arm], col, arm, time)
        style_axis(ax, value, time, ylabel=value)
        ax.set_xlim(right=frames['equal'][time].max())
        ax.set_title(label)
        ax.legend(fontsize=8)
    for ax in flat[n:]:
        ax.axis('off')
    # Counts scale with community size (White is 86% of the population, the fourth 0.7%), so a
    # shared axis would flatten every minority panel; rates are directly comparable, so they get
    # one. Same rule plot_IQR.py uses.
    if is_rate(value):
        top = max(ax.get_ylim()[1] for ax in flat[:n])
        for ax in flat[:n]:
            ax.set_ylim(0, top)
        note = 'shared y axis'
    else:
        note = 'independent y axes -- note the differing scales'
    fig.suptitle(f'{value} by community, both arms  ({note})', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return out


def mean_se(df, col, time='year'):
    ''' Across-run mean and standard error of the mean at each timepoint. '''
    g = df.groupby(time)[col]
    return g.mean(), g.std(ddof=1) / np.sqrt(g.count())


def effect_plot(frames, value, out, time='year', smooth=5):
    '''
    The difference between arms as a percentage of the control, with a 95% CI band.

    This is the plot that answers the question. Two near-identical curves on a shared axis look
    the same whether the difference is 0.1% or 5%; here the difference is the line, and if its
    band straddles zero the run cannot distinguish the arms.

    ``smooth`` is a centred rolling mean in years, applied because annual flows (cancers,
    cancer_deaths) arrive in multiples of the multiscale scale factor and jump around far more
    than the trend does. The BAND IS SMOOTHED THE SAME WAY AND NOT NARROWED: averaging several
    years of a correlated series does reduce the error somewhat, but by an amount that depends on
    the autocorrelation, so the band is left at its per-year width rather than claiming a
    precision this does not establish. Pass smooth=1 for the raw series.
    '''
    labels = communities(frames['equal'])
    panels = [(lab, f'{value}{SUFFIX}{lab}') for lab in labels
              if f'{value}{SUFFIX}{lab}' in frames['equal'].columns]
    if value in frames['equal'].columns:
        panels.append(('Whole population', value))
    if not panels:
        return None

    n = len(panels)
    ncols = 2 if n <= 4 else 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.6 * nrows), squeeze=False)
    flat = axes.ravel()
    for ax, (label, col) in zip(flat, panels):
        ma, sa = mean_se(frames['equal'], col, time)
        mb, sb = mean_se(frames['observed'], col, time)
        idx = ma.index.intersection(mb.index)
        ma, sa, mb, sb = ma[idx], sa[idx], mb[idx], sb[idx]
        # Drop the burn-in, where the control mean is still near zero: a percentage difference
        # against a denominator of ~0 is meaningless, and left in it swamps the y axis (cancers
        # in 1980 gave a +150% spike that flattened the entire informative range to a line).
        if (ma > 0).any():
            keep = ma > BURN_IN_FRAC * ma.max()
            idx = idx[keep.values]
            ma, sa, mb, sb = ma[keep], sa[keep], mb[keep], sb[keep]
        with np.errstate(divide='ignore', invalid='ignore'):
            pct = 100 * (mb - ma) / ma
            # SE of (b-a)/a, treating a's own error as the dominant term via the ratio's
            # first-order expansion; good enough for a band whose job is to show whether the
            # difference is distinguishable from zero.
            se_pct = 100 * np.sqrt(sa ** 2 + sb ** 2) / ma.abs()
        if smooth and smooth > 1:
            pct = pct.rolling(smooth, center=True, min_periods=1).mean()
            se_pct = se_pct.rolling(smooth, center=True, min_periods=1).mean()
        ax.axhline(0, color='0.4', linewidth=1)
        ax.fill_between(idx, pct - 1.96 * se_pct, pct + 1.96 * se_pct,
                        color='tab:purple', alpha=0.20, linewidth=0, label='95% CI')
        ax.plot(idx, pct, color='tab:purple', linewidth=2, label='observed - equal')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'% change in {value}')
        ax.set_title(label)
        ax.grid(True, linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=8)
        finite = pct[np.isfinite(pct)]
        if len(finite):
            span = max(5.0, float(np.nanmax(np.abs(finite))) * 1.3)
            ax.set_ylim(-span, span)
    for ax in flat[n:]:
        ax.axis('off')
    fig.suptitle(f'{value}: observed uptake vs equal uptake, % difference with 95% CI', fontsize=13)
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', dpi=120)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='census')
    ap.add_argument('--values', nargs='*', default=DEFAULT_VALUES)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--smooth', type=int, default=5,
                    help='centred rolling mean, in years, for the effect plot (1 = off)')
    args = ap.parse_args()

    out_dir = pathlib.Path(args.outdir) if args.outdir else FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load(args.tag)
    prefix = f'ethnicity_{args.tag}'

    made = []
    for value in args.values:
        for fn, suffix in ((compare_plot, 'compare'),
                           (community_plot, 'by_community'),
                           (effect_plot, 'effect')):
            path = out_dir / f'{prefix}_{value}_{suffix}.png'
            kw = dict(smooth=args.smooth) if fn is effect_plot else {}
            if fn(frames, value, path, **kw) is not None:
                made.append(path)
                print(f'  saved {path.name}')
            else:
                print(f'  skipped {path.name} (no such column)')
    print(f'\n{len(made)} figures in {out_dir}')
    return made


if __name__ == '__main__':
    main()

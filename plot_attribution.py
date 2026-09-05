"""
plot_attribution.py
===================

Figures for the core-group attribution runs produced by run_powerlaw_attribution.py. Sibling of
plot_IQR.py -- same CLI shape, same median + IQR convention, same figs/ output layout -- but the
x axis here is a percentile of the population rather than time.

    python plot_attribution.py <tag> [<tag> ...] [--out <subdir>]
    python plot_attribution.py powerlaw_alpha3_50runs
    python plot_attribution.py powerlaw_alpha2p5 powerlaw_alpha3 powerlaw_alpha5 --out alpha_sweep

reads csvs/<tag>_curves.npz and csvs/<tag>_attribution.csv and writes into figs/[out_dir]/:

    <tag>_attribution_curve.png   share of infections and of cancers transmitted by the top X%
                                  of the population, X on a log axis, median + IQR across runs.
                                  The headline figure: read the 5% and 20% annotations off it.
    <tag>_lorenz.png              Lorenz curve of infections ACQUIRED, ranked by activity, with
                                  the Gini in the legend. Comparable in FORM to Gsteiger et al.,
                                  PeerJ 2020 (Gini 0.38 for HPV-18), but not in level: they rank
                                  by partners in the last year and accumulate PREVALENT infection,
                                  whereas this accumulates every infection over a lifetime and
                                  ranks by latent propensity. Lifetime totals average over each
                                  agent's whole history, so this Gini is systematically the
                                  flatter of the two -- do not quote the two side by side as if
                                  they measured the same thing.
    <tag>_downstream.png          direct vs downstream attribution at each tracked core
                                  quantile, as grouped bars over generation depth.

Multiple tags can be passed to overlay them (e.g. several Pareto alphas), in which case a
'<n>-way compare_' figure is written instead, matching plot_IQR.py's naming.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

CSV_DIR = pathlib.Path(__file__).parent / 'csvs'
FIG_DIR = pathlib.Path(__file__).parent / 'figs'

ANNOTATE_AT = (0.05, 0.20)  # percentiles called out on the attribution curve
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#8c564b']

pct = FuncFormatter(lambda x, _: f'{x:.0%}')
# Log axes reach down to 0.1%, where a 0-decimal percentage would just read "0%"
pct_log = FuncFormatter(lambda x, _: (f'{x * 100:g}%' if x < 0.01 else f'{x:.0%}'))


def load(tag):
    """(curves npz as a dict, tidy dataframe) for one run set."""
    npz = np.load(CSV_DIR / f'{tag}_curves.npz', allow_pickle=True)
    curves = {k: npz[k] for k in npz.files}
    df = pd.read_csv(CSV_DIR / f'{tag}_attribution.csv')
    return curves, df


def band(ax, x, stack, color, label, alpha=0.25):
    """Median line + IQR band for a (n_runs, n_points) stack, matching plot_IQR.draw_iqr."""
    med = np.median(stack, axis=0)
    lo = np.quantile(stack, 0.25, axis=0)
    hi = np.quantile(stack, 0.75, axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=alpha, linewidth=0)
    ax.plot(x, med, color=color, linewidth=2, label=label)
    return med


def style_pct_axes(ax, xlabel, ylabel):
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return ax


# -------------------------------------------------------------------
# 1. attribution curve -- the headline figure
# -------------------------------------------------------------------

def plot_attribution_curve(sets, out_path):
    """
    The headline figure: cumulative share of transmission against activity percentile, i.e. the
    whole "top X% are responsible for Y%" relationship rather than a single quoted pair.

    Two panels of the same curve. LEFT is linear over the full 0-100% range -- the one to read
    percentiles off, and the one where the gap from the diagonal is the honest visual measure of
    concentration. RIGHT is log-scaled down to 0.1%, which is the only way to see what the extreme
    tail does; on a linear axis the top 1% is squashed against the y axis.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    multi = len(sets) > 1

    for panel, ax in enumerate(axes):
        log = (panel == 1)
        for i, (tag, curves, _) in enumerate(sets):
            x = curves['grid']
            c = PALETTE[i % len(PALETTE)]
            inf = band(ax, x, curves['direct_infections'], c,
                       f'{tag} — infections' if multi else 'Infections')
            if not multi:
                band(ax, x, curves['direct_cancers'], PALETTE[1], 'Cancers')
                if not log:  # annotate the readable panel only
                    for q in ANNOTATE_AT:
                        j = int(np.argmin(np.abs(x - q)))
                        share = inf[j]
                        ax.plot([q], [share], 'o', color=c, markersize=6, zorder=5)
                        ax.annotate(f'top {q:.0%} → {share:.0%} of infections',
                                    xy=(q, share), xytext=(q + 0.10, share - 0.10),
                                    fontsize=9, color=c,
                                    arrowprops=dict(arrowstyle='-', color=c, linewidth=0.8))

        # y = x is the no-heterogeneity null: the top X% causing exactly X% of transmission. It
        # has to be drawn as a dense curve, not two endpoints -- on a log x axis matplotlib would
        # join two points with a straight line in DISPLAY space, which is not y = x and would make
        # the observed concentration look far more extreme than it is.
        null = np.logspace(-3, 0, 400)
        ax.plot(null, null, '--', color='grey', linewidth=1,
                label='No heterogeneity (top X% causes X%)')
        if log:
            ax.set_xscale('log')
            ax.set_xlim(1e-3, 1)
        else:
            ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        style_pct_axes(ax, 'Most active X% of the population (ranked by propensity θ)',
                       'Cumulative share transmitted' if not log else '')
        if log:
            ax.xaxis.set_major_formatter(pct_log)
        ax.set_title('Full range (linear)' if not log else 'Tail detail (log scale)', fontsize=10)

    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Core-group attribution: share of transmission by activity percentile\n'
                 'median and IQR across runs', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'  wrote {out_path}')


# -------------------------------------------------------------------
# 2. Lorenz curve of acquisitions
# -------------------------------------------------------------------

def plot_lorenz(sets, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, (tag, curves, _) in enumerate(sets):
        g = curves['gini']
        label = (f'{tag}  (Gini {np.median(g):.3f})' if len(sets) > 1
                 else f'Infections acquired\nGini {np.median(g):.3f} '
                      f'[IQR {np.quantile(g, 0.25):.3f}–{np.quantile(g, 0.75):.3f}]')
        band(ax, curves['grid'], curves['lorenz_acquired'], PALETTE[i % len(PALETTE)], label)
    ax.plot([0, 1], [0, 1], '--', color='grey', linewidth=1, label='Line of equality')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    style_pct_axes(ax, 'Cumulative share of population\n(least to most active, by θ)',
                   'Cumulative share of infections acquired')
    ax.set_title('Concentration of acquired infection\n'
                 'cf. Gsteiger et al., PeerJ 2020', fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'  wrote {out_path}')


# -------------------------------------------------------------------
# 3. direct vs downstream
# -------------------------------------------------------------------

def plot_downstream(sets, out_path):
    tag, _, df = sets[0]
    down = df[df['measure'].str.startswith('downstream_gen')].copy()
    if down.empty:
        print('  no downstream rows in the CSV; skipping')
        return
    # NB bracket access, not attribute: DataFrame.quantile is a method, so down.quantile would
    # return that method rather than the column of the same name
    down['gen'] = down['measure'].str.replace('downstream_gen', '', regex=False).astype(int)
    quants = sorted(down['quantile'].unique())
    gens = sorted(down['gen'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, kind in zip(axes, ['infections', 'cancers']):
        width = 0.8 / len(gens)
        base = np.arange(len(quants))
        for j, gen in enumerate(int(g) for g in gens):  # int(), else numpy scalars in the f-string
            med, lo, hi = [], [], []
            for q in quants:
                v = down[(down['quantile'] == q) & (down['gen'] == gen)][kind].to_numpy()
                med.append(np.median(v))
                lo.append(np.median(v) - np.quantile(v, 0.25))
                hi.append(np.quantile(v, 0.75) - np.median(v))
            label = 'Direct (transmitter)' if gen == 0 else f'Within {gen} generation{"s" * (gen > 1)}'
            ax.bar(base + j * width, med, width, yerr=[lo, hi], capsize=2,
                   color=PALETTE[j % len(PALETTE)], label=label, error_kw=dict(linewidth=0.8))
        ax.set_xticks(base + width * (len(gens) - 1) / 2)
        ax.set_xticklabels([f'top {q:.0%}' for q in quants])
        ax.yaxis.set_major_formatter(pct)
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', linewidth=0.5, alpha=0.4)
        ax.set_title(kind.capitalize())
    axes[0].set_ylabel('Share attributable to the core group')
    axes[0].legend(frameon=False, fontsize=9, loc='upper left')
    fig.suptitle('Direct vs downstream attribution, by core-group size and generation depth\n'
                 'median and IQR across runs', fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'  wrote {out_path}')


# -------------------------------------------------------------------

def main(argv):
    argv = list(argv[1:])
    out_sub = ''
    if '--out' in argv:
        i = argv.index('--out')
        out_sub = argv[i + 1] if i + 1 < len(argv) else ''
        del argv[i:i + 2]
    tags = list(dict.fromkeys(a for a in argv if not a.startswith('-')))  # de-duped, order kept
    if not tags:
        print(__doc__)
        return 1

    missing = [t for t in tags if not (CSV_DIR / f'{t}_curves.npz').exists()]
    if missing:
        print(f'No <tag>_curves.npz found in {CSV_DIR} for: {", ".join(missing)}')
        return 1
    out_dir = FIG_DIR / out_sub
    out_dir.mkdir(parents=True, exist_ok=True)

    sets = [(t, *load(t)) for t in tags]
    prefix = tags[0] if len(tags) == 1 else f'{len(tags)}-way compare_{tags[0]}'
    print(f'Plotting {len(tags)} run set(s) into {out_dir}')
    plot_attribution_curve(sets, out_dir / f'{prefix}_attribution_curve.png')
    plot_lorenz(sets, out_dir / f'{prefix}_lorenz.png')
    plot_downstream(sets, out_dir / f'{prefix}_downstream.png')
    return 0


if __name__ == '__main__':
    main(sys.argv)

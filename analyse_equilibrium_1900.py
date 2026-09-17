"""
analyse_equilibrium_1900.py
===========================

Reads run_equilibrium_1900.py's outputs (csvs/equilibrium_1900/<network>_seed<seed>.csv) and asks,
for each network, whether HPV prevalence has settled before screening starts in 1980.

For HPV prevalence (all types) and each genotype, per network (mean over seeds):
  * pre-screening level   = mean over 1960-1979
  * trend 1960-1979       = linear-fit change, % of that level per decade
  * settled from          = first year after which the 5-year rolling mean stays within +-5% of the
                            1960-1979 level through 1979 (None if it never does)

Figure: figs/equilibrium_1900/equilibrium_1900_2070.png -- one panel per network, prevalence
indexed to its own 1960-1979 level (1 = pre-screening equilibrium), mean over seeds with the
seed range shaded.
"""
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
IN_DIR = ROOT / 'csvs' / 'equilibrium_1900'
FIG_DIR = ROOT / 'figs' / 'equilibrium_1900'
NETWORKS = ['default', 'gamma_0.05', 'gamma_0.25', 'gamma_1', 'gamma_2', 'gamma_5', 'powerlaw_3']
LABELS = {'default': 'Default (Poisson)', 'gamma_0.05': 'Gamma shape 0.05', 'gamma_0.25': 'Gamma shape 0.25',
          'gamma_1': 'Gamma shape 1', 'gamma_2': 'Gamma shape 2', 'gamma_5': 'Gamma shape 5',
          'powerlaw_3': 'Power law, alpha 3'}
BASE = (1960, 1979)
TOL = 0.05
SCREENING, VACCINATION = 1980, 2008

# Reference palette (dataviz skill, light mode): categorical slots 1-2, surface and text tokens
SURFACE, TEXT, TEXT2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'hpv_prevalence': ('#2a78d6', 'All types'), 'hpv_prevalence_hpv16': ('#eb6834', 'HPV16')}


def load():
    frames = [pd.read_csv(f) for f in sorted(IN_DIR.glob('*_seed*.csv')) if '-' not in f.stem.split('seed')[-1]]
    if not frames:
        raise SystemExit(f'no run outputs in {IN_DIR}')
    df = pd.concat(frames, ignore_index=True)
    df['year'] = df['year'].round().astype(int)
    return df


def settle_stats(years, values):
    base_mask = (years >= BASE[0]) & (years <= BASE[1])
    level = values[base_mask].mean()
    slope = np.polyfit(years[base_mask], values[base_mask], 1)[0]
    rolling = pd.Series(values, index=years).rolling(5, center=True, min_periods=3).mean()
    pre = rolling[rolling.index <= BASE[1]]
    outside = pre[(pre / level - 1).abs() > TOL]
    settled = int(outside.index.max()) + 1 if len(outside) else int(pre.index.min())
    if settled > BASE[1]:
        settled = None
    return level, 100 * slope * 10 / level, settled


def main():
    df = load()
    measures = [c for c in df.columns if c.startswith('hpv_prevalence')]
    rows = []
    for net in NETWORKS:
        d = df[df['network'] == net]
        if d.empty:
            continue
        g = d.groupby('year')[measures].mean()
        years = g.index.to_numpy()
        for m in measures:
            level, trend, settled = settle_stats(years, g[m].to_numpy())
            rows.append(dict(network=net, measure=m.replace('hpv_prevalence', 'all types').replace('all types_', ''),
                             seeds=d['seed'].nunique(), level_1960_79=level, trend_pct_per_decade=trend,
                             settled_from=settled, value_1900=g[m].iloc[0], peak_year=int(g[m].idxmax())))
    table = pd.DataFrame(rows)
    pd.set_option('display.width', 200)
    print(table.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
    table.to_csv(IN_DIR / 'equilibrium_summary.csv', index=False)

    nets = [n for n in NETWORKS if n in set(df['network'])]
    ncol = 4
    nrow = int(np.ceil(len(nets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, net in zip(axes.flat, nets):
        ax.set_visible(True)
        ax.set_facecolor(SURFACE)
        d = df[df['network'] == net]
        for m, (color, label) in SERIES.items():
            piv = d.pivot_table(index='year', columns='seed', values=m)
            base = piv.loc[BASE[0]:BASE[1]].mean().mean()
            idx = piv / base
            ax.fill_between(idx.index, idx.min(axis=1), idx.max(axis=1), color=color, alpha=0.18, linewidth=0)
            ax.plot(idx.index, idx.mean(axis=1), color=color, linewidth=2, solid_capstyle='round', label=label)
        ax.axhspan(1 - TOL, 1 + TOL, color=GRID, alpha=0.6, linewidth=0, zorder=0)
        ax.axhline(1, color=TEXT2, linewidth=0.8, zorder=1)
        for yr, name in ((SCREENING, 'screening'), (VACCINATION, 'vaccination')):
            ax.axvline(yr, color=TEXT2, linewidth=0.8, zorder=1)
            ax.text(yr + 1.5, 0.98, name, transform=ax.get_xaxis_transform(), va='top', ha='left',
                    fontsize=8, color=TEXT2)
        ax.set_title(f"{LABELS[net]}  (n={d['seed'].nunique()})", fontsize=10, color=TEXT, loc='left')
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT2, labelsize=8)
        ax.set_xlim(1900, 2070)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel('Year', color=TEXT2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('Prevalence / 1960-79 mean', color=TEXT2, fontsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle('HPV prevalence from a 1900 start, indexed to the 1960-79 pre-screening level '
                 '(shaded band = +-5%; ribbons = range over seeds)', fontsize=11, color=TEXT, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / 'equilibrium_1900_2070.png'
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

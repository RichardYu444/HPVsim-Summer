"""
plot_equilibrium_hpv16.py
=========================

HPV16 prevalence time series for each run_equilibrium_1900.py output
(csvs/equilibrium_1900/<network>_seed<seed>.csv): one panel per CSV, a row per network and a column
per seed, all on the same axes. Each panel draws its own run in the HPV16 colour over the network's
other seeds in gray, and gives that run's 1960-1979 (pre-screening) mean top right.

    python plot_equilibrium_hpv16.py

Figure: figs/equilibrium_1900/equilibrium_1900_2070_hpv16_per_csv.png. Re-run as more networks finish.
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter

from analyse_equilibrium_1900 import (BASE, FIG_DIR, GRID, NETWORKS, SCREENING, SERIES, SURFACE, TEXT,
                                      TEXT2, VACCINATION, load)

VALUE = 'hpv_prevalence_hpv16'
COLOR = SERIES[VALUE][0]    # same orange as HPV16 in equilibrium_1900_2070.png
CONTEXT = '#c3c2b7'         # de-emphasis gray for the network's other seeds


def main():
    df = load()
    present = set(df['network'])
    nets = [n for n in NETWORKS if n in present] + sorted(present - set(NETWORKS))
    runs = {net: df[df['network'] == net].pivot_table(index='year', columns='seed', values=VALUE)
            for net in nets}
    ncol = max(piv.shape[1] for piv in runs.values())
    height = 2.5 * len(nets) + 0.8
    fig, axes = plt.subplots(len(nets), ncol, figsize=(3.2 * ncol, height), sharex=True, sharey=True,
                             facecolor=SURFACE, squeeze=False)

    print(f"{'csv':<24}{'1900':>8}{'peak':>8}{'in':>6}{'1960-79':>9}{'2070':>8}")
    for row, net in zip(axes, nets):
        piv = runs[net]
        for ax in row[piv.shape[1]:]:
            ax.set_visible(False)
        for ax, seed in zip(row, piv.columns):
            name = f'{net}_seed{seed}.csv'
            s = piv[seed]
            level = s.loc[BASE[0]:BASE[1]].mean()
            print(f'{name:<24}{s.iloc[0]:>8.2%}{s.max():>8.2%}{s.idxmax():>6}{level:>9.2%}{s.iloc[-1]:>8.2%}')

            ax.set_facecolor(SURFACE)
            for other in piv.columns.drop(seed):
                ax.plot(piv.index, piv[other], color=CONTEXT, linewidth=1)
            ax.plot(s.index, s, color=COLOR, linewidth=2, solid_capstyle='round')
            for yr in (SCREENING, VACCINATION):
                ax.axvline(yr, color=TEXT2, linewidth=0.8, zorder=1)
            ax.set_title(name, loc='left', fontsize=9, color=TEXT)
            ax.set_title(f'{BASE[0]}-{str(BASE[1])[2:]} mean {level:.1%}', loc='right', fontsize=8, color=TEXT2)
            ax.grid(axis='y', color=GRID, linewidth=0.8)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=TEXT2, labelsize=8)

    ax = axes[0, 0]
    ax.set_xlim(1900, 2070)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    for yr, label in ((SCREENING, 'screening'), (VACCINATION, 'vaccination')):
        ax.text(yr + 2, 0.97, label, transform=ax.get_xaxis_transform(), rotation=90, va='top', ha='left',
                fontsize=7, color=TEXT2)
    for ax in axes[:, 0]:
        ax.set_ylabel('HPV16 prevalence', color=TEXT2, fontsize=9)
    # Year labels under the lowest panel of each column (a short last row would otherwise hide them)
    for c in range(ncol):
        shown = [ax for ax in axes[:, c] if ax.get_visible()]
        if shown:
            shown[-1].tick_params(labelbottom=True)
            shown[-1].set_xlabel('Year', color=TEXT2, fontsize=9)

    handles = [Line2D([], [], color=COLOR, linewidth=2), Line2D([], [], color=CONTEXT, linewidth=1)]
    fig.legend(handles, ["This CSV's run", 'Other seeds, same network'], loc='upper right', ncol=2,
               frameon=False, fontsize=9, labelcolor=TEXT)
    fig.suptitle('HPV16 prevalence from a 1900 start, one panel per CSV in csvs/equilibrium_1900',
                 fontsize=11, color=TEXT, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.1 / height))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / 'equilibrium_1900_2070_hpv16_per_csv.png'
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()

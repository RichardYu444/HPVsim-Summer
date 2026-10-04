"""
Plots the output of diagnose_community_isolation.py's default runs as a 2x2 figure:

    A. both communities seeded in 1980 (what the real runs do) -- identical curves
    B. Small left uninfected until 2000 -- it stays at exactly 0, then runs its own epidemic
    C. panel B with Small shifted back 20 years -- the late epidemic replays Large's curve
    D. Small seeded at 5% of the usual prevalence -- a delayed peak, then convergence

Every panel plots sim.results['hpv_prevalence_by_community'], i.e. what the run CSVs contain.
Output: figs/community/diagnostics/isolation_diagnostics_v2.png
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
# Same folder and naming as diagnose_community_isolation.output_path() -- not imported from
# there, since that would load all of HPVsim just to build a file name
INPUT_DIR = ROOT / 'csvs' / 'isolation_diagnostics'
OUT = ROOT / 'figs' / 'community' / 'diagnostics' / 'isolation_diagnostics_v2.png'
N_AGENTS, END = 20_000, 2045
BLUE, ORANGE = '#1f77b4', '#ff7f0e'


def load(exp, seed):
    return np.load(INPUT_DIR / f'diag_{exp}_s{seed}_n{N_AGENTS}_{END}.npz')


def stock_series(d):
    ''' Recorded by-community prevalence; stocks are taken on the last step of each year '''
    return d['res_year'] + 0.75, d['res_prev']


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    runs = {}

    # A: as in the real runs
    d = runs['both_t0'] = load('both_t0', 1)
    x, p = stock_series(d)
    ax = axes[0, 0]
    ax.plot(x, p[0], color=BLUE, lw=2, label='Large (80%)')
    ax.plot(x, p[1], color=ORANGE, lw=2, label='Small (20%)')
    ax.set_title('A. Isolated, both seeded 1980 with the same init_hpv_prev\n(what the real runs do)')

    # B: Small seeded 20 years late, and C: the same with Small shifted back 20 years
    for seed, ls in ((1, '-'), (2, '--')):
        d = runs[f'delayed s{seed}'] = load('delayed', seed)
        x, p = stock_series(d)
        axes[0, 1].plot(x, p[0], color=BLUE, lw=2, ls=ls, label=f'Large, seeded 1980 (seed {seed})')
        axes[0, 1].plot(x, p[1], color=ORANGE, lw=2, ls=ls, label=f'Small, seeded 2000 (seed {seed})')
        axes[1, 0].plot(x, p[0], color=BLUE, lw=2, ls=ls, label=f'Large (seed {seed})')
        axes[1, 0].plot(x - 20, p[1], color=ORANGE, lw=2, ls=ls,
                        label=f'Small, shifted back 20 yr (seed {seed})')
    axes[0, 1].axvline(2000, color='grey', lw=0.8, ls=':')
    axes[0, 1].set_title('B. Isolated, Small left uninfected until 2000')
    axes[1, 0].set_title('C. Panel B with Small moved back 20 years:\nthe late epidemic replays the same curve')

    # D: Small seeded at 5% of the usual prevalence
    d = runs['low_small'] = load('low_small', 1)
    x, p = stock_series(d)
    ax = axes[1, 1]
    ax.plot(x, p[0], color=BLUE, lw=2, label='Large, usual seeding')
    ax.plot(x, p[1], color=ORANGE, lw=2, label='Small, seeded at 5% of usual')
    ax.set_title('D. Isolated, both seeded 1980, Small seeded lightly')

    for ax in axes.ravel():
        ax.set_ylabel('HPV prevalence (sim.results, by community)')
        ax.set_xlabel('year')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlim(1980, 2025 if ax is axes[1, 0] else END)
    axes[0, 0].set_ylim(0, 0.65)

    n_cross = sum(int(d['cross_tx'].sum()) for d in runs.values())
    fig.suptitle(f'Two isolated communities, {N_AGENTS // 1000}k agents, no interventions. '
                 f'Cross-community transmissions across all runs: {n_cross}', fontsize=13)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f'saved {OUT}')

    for name, d in runs.items():
        print(f"  {name:12s} cross-community transmissions={int(d['cross_tx'].sum())}  "
              f"within={int(d['within_tx'].sum())}  max cross-community contacts={int(d['cross_edges'].max())}")


if __name__ == '__main__':
    main()

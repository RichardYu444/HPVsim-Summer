"""
plot_reff_verification.py
=========================

Figures and tables for run_reff_verification.py's output.

    python plot_reff_verification.py [TAG]        (default TAG: default_20k_1950_2090)

Writes into figs/Default/reff/:
    <TAG>_reff_by_genotype.png     analytic vs empirical R_eff over time, vaccination vs none
    <TAG>_ohr_cross_immunity.png   OHR: R_eff with and without natural cross-immunity, + prevalence
    <TAG>_estimator_check.png      analytic / no-competition / mean-field against empirical
and prints the R0 table.
"""
import pathlib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).parent
TAG = sys.argv[1] if len(sys.argv) > 1 else 'default_20k_1950_2090'
IN = ROOT / 'csvs' / 'reff_verification'
FIGS = ROOT / 'figs' / 'Default' / 'reff'

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SCEN = {'novacc': (BLUE, 'no vaccination'), 'vacc': (ORANGE, 'NHS vaccination')}
LABEL = {'hpv16': 'HPV16', 'hpv18': 'HPV18', 'hi5': 'hi5', 'ohr': 'OHR'}
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False})


def style(ax, ylabel=None):
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel('year the cases were infected')
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(IN / f'{TAG}_summary.csv')
    mf = pd.read_csv(IN / f'{TAG}_meanfield.csv')
    prev = pd.read_csv(IN / f'{TAG}_prevalence.csv')
    S = summ[summ['count'] == 'all']
    genos = [g for g in ['hpv16', 'hpv18', 'hi5', 'ohr'] if g in set(S.genotype)]

    # ---------------- R0 table ----------------
    # Immune-naive R0 depends only on the network, so both scenarios' rows (where present) are pooled
    R0 = summ[summ.estimator == 'R0'].groupby(['count', 'genotype', 'year'])['rho'].mean()
    RV = summ[summ.estimator == 'R_vx'].groupby(['count', 'genotype', 'year'])['rho'].mean()
    y0, y1 = int(summ.year.min()), int(summ.year.max())
    tab = pd.DataFrame({f'R0 {y0}': R0.xs(y0, level='year'), f'R0 {y1}': R0.xs(y1, level='year')})
    if len(RV):
        tab[f'R_vx {y1}'] = RV.xs(y1, level='year')
    print('R0 (immune-naive) and R_vx (vaccine immunity only), by genotype:')
    print(tab.round(2).to_string())
    r0_all = R0.loc['all'].groupby('genotype').agg(['first', 'last'])
    r0_dist = R0.loc['dist'].groupby('genotype').agg(['first', 'last'])

    # Cohorts with too few cases (e.g. hi5, extinct) say nothing -- kept out of the plots
    MIN_CASES = 100
    big = (S.n_m + S.n_f) >= MIN_CASES

    # ---------------- Fig 1: R_eff by genotype ----------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    top = S[(S.estimator.isin(['analytic', 'empirical']) & big) | (S.estimator == 'R_intro')]
    ymax = max(1.3, float(np.nanmax(np.r_[top.rho.to_numpy(), top['hi'].dropna().to_numpy()])) * 1.05)
    for ax, g in zip(axes.flat, genos):
        for scen, (col, lab) in SCEN.items():
            if g == 'hi5':
                a = S[(S.genotype == g) & (S.scen == scen) & (S.estimator == 'R_intro')].sort_values('year')
                ax.plot(a.year, a.rho, color=col, lw=2, label=f'{lab}: introduced case (analytic)')
                continue
            a = S[(S.genotype == g) & (S.scen == scen) & (S.estimator == 'analytic') & big].sort_values('year')
            e = S[(S.genotype == g) & (S.scen == scen) & (S.estimator == 'empirical') & big].sort_values('year')
            ax.plot(a.year, a.rho, color=col, lw=2, label=f'{lab}: analytic')
            off = -0.6 if scen == 'novacc' else 0.6
            yerr = np.vstack([e.rho - e.lo, e.hi - e.rho]).clip(0)
            ax.errorbar(e.year + off, e.rho, yerr=yerr, fmt='o', ms=4.5, color=col, mfc='white',
                        mew=1.5, elinewidth=1, capsize=0, label=f'{lab}: simulated (95% CI)')
        ax.axhline(1, color=INK2, lw=0.9, ls=':')
        ax.axvline(2008, color=INK2, lw=0.8, ls='--', alpha=0.6)
        extra = ' (extinct: R for an introduced case)' if g == 'hi5' else ''
        ax.set_title(f'{LABEL[g]}{extra}\nR0 {r0_all.loc[g, "first"]:.2f} -> {r0_all.loc[g, "last"]:.2f} '
                     f'({y0}->{y1}); distinct people {r0_dist.loc[g, "first"]:.2f} -> '
                     f'{r0_dist.loc[g, "last"]:.2f}', fontsize=10, color=INK)
        ax.set_ylim(0, ymax)
        style(ax, 'R_eff (spectral radius of K)')
    axes[0, 0].text(2008.8, axes[0, 0].get_ylim()[1] * 0.96, 'vaccination\nstarts', fontsize=8,
                    color=INK2, va='top')
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Effective reproduction number by genotype: analytic next-generation estimate vs '
                 'realised offspring of the same cases', fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out1 = FIGS / f'{TAG}_reff_by_genotype.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------- Fig 2: OHR, cross-immunity ----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for scen, (col, lab) in SCEN.items():
        a = S[(S.genotype == 'ohr') & (S.scen == scen) & (S.estimator == 'analytic')].sort_values('year')
        n = S[(S.genotype == 'ohr') & (S.scen == scen) & (S.estimator == 'analytic_no_cross')].sort_values('year')
        ax1.plot(a.year, a.rho, color=col, lw=2, label=f'{lab}: as modelled')
        ax1.plot(n.year, n.rho, color=col, lw=2, ls='--', label=f'{lab}: natural cross-immunity removed')
        ax1.fill_between(a.year, a.rho, n.rho, color=col, alpha=0.10, lw=0)
    ax1.axhline(1, color=INK2, lw=0.9, ls=':')
    ax1.axvline(2008, color=INK2, lw=0.8, ls='--', alpha=0.6)
    ax1.set_ylim(0, None)
    ax1.set_title('OHR R_eff: the shaded gap is what cross-immunity from\nHPV16/18/hi5 infection takes off',
                  fontsize=10, color=INK)
    style(ax1, 'R_eff (analytic)')
    ax1.legend(frameon=False, fontsize=8.5, loc='lower left')
    pm = prev.groupby(['scen', 'year'])['prev_ohr'].mean().reset_index()
    for scen, (col, lab) in SCEN.items():
        p = pm[pm.scen == scen]
        ax2.plot(p.year, p.prev_ohr, color=col, lw=2, label=lab)
    ax2.axvline(2008, color=INK2, lw=0.8, ls='--', alpha=0.6)
    ax2.set_xlim(1990, 2090)
    ax2.set_ylim(0, None)
    ax2.grid(color=GRID, lw=0.6)
    ax2.set_xlabel('year')
    ax2.set_ylabel('OHR prevalence')
    ax2.set_title('OHR prevalence in the same runs (no screening)', fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.tight_layout()
    out2 = FIGS / f'{TAG}_ohr_cross_immunity.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------- Fig 3: estimator check ----------------
    e = S[S.estimator == 'empirical'][['scen', 'year', 'genotype', 'rho', 'n_m', 'n_f']].rename(columns={'rho': 'emp'})
    e = e[(e.n_m + e.n_f) >= 200]  # cohorts too small to say anything are left out
    rows = []
    for est, lab, col, mk in (('analytic', 'analytic (with competition)', ORANGE, 'o'),
                              ('analytic_no_comp', 'analytic, no competition', BLUE, 's'),
                              ('mean_field', 'mean field R0 x sqrt(sigma_f sigma_m delta_f)', AQUA, '^')):
        if est == 'mean_field':
            a = mf.rename(columns={'R_mf': 'val'})[['scen', 'year', 'genotype', 'val']]
        else:
            a = S[S.estimator == est][['scen', 'year', 'genotype', 'rho']].rename(columns={'rho': 'val'})
        m = e.merge(a, on=['scen', 'year', 'genotype'])
        rows.append((lab, col, mk, m))
    fig, ax = plt.subplots(figsize=(6.4, 6))
    lim = max(max(m.val.max(), m.emp.max()) for _, _, _, m in rows) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, lw=0.9, ls=':')
    for lab, col, mk, m in rows:
        err = np.median(np.abs(m.val / m.emp - 1))
        ax.scatter(m.emp, m.val, s=26, marker=mk, facecolor='white', edgecolor=col, lw=1.4,
                   label=f'{lab}  (median |error| {100 * err:.0f}%)')
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(color=GRID, lw=0.6)
    ax.set_xlabel('simulated R_eff (realised offspring of the cohort)')
    ax.set_ylabel('predicted R_eff')
    ax.set_title('Each point: one genotype x scenario x cohort year', fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    out3 = FIGS / f'{TAG}_estimator_check.png'
    fig.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------- Fig 4: R0 and R_vx ----------------
    fig, axes = plt.subplots(1, len(genos), figsize=(3.4 * len(genos), 3.8), sharey=True)
    for ax, g in zip(np.atleast_1d(axes), genos):
        a = R0.loc['all'].loc[g]
        ax.plot(a.index, a.values, color=BLUE, lw=2, label='R0 (immune-naive)')
        if len(RV):
            v = RV.loc['all'].loc[g]
            ax.plot(v.index, v.values, color=ORANGE, lw=2, label='R_vx (vaccine immunity only)')
        ax.axhline(1, color=INK2, lw=0.9, ls=':')
        ax.axvline(2008, color=INK2, lw=0.8, ls='--', alpha=0.6)
        ax.set_ylim(0, None)
        ax.set_title(LABEL[g], fontsize=10.5, color=INK)
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel('year')
    np.atleast_1d(axes)[0].set_ylabel('reproduction number (counting reinfections)')
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('What the network alone allows (R0), and what vaccination alone leaves (R_vx)',
                 fontsize=11, color=INK)
    fig.tight_layout()
    out4 = FIGS / f'{TAG}_r0_and_rvx.png'
    fig.savefig(out4, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------- printed summary ----------------
    piv = S[S.estimator.isin(['analytic', 'empirical', 'analytic_no_cross'])].pivot_table(
        index=['genotype', 'scen', 'year'], columns='estimator', values='rho')
    with pd.option_context('display.width', 200, 'display.max_rows', 500):
        print(piv.round(3).to_string())
    for lab, col, mk, m in rows:
        print(f'{lab}: median |error| {100 * np.median(np.abs(m.val / m.emp - 1)):.1f}%, '
              f'mean bias {100 * np.mean(m.val / m.emp - 1):+.1f}%  (n={len(m)})')
    print('saved', out1, out2, out3, out4, sep='\n  ')


if __name__ == '__main__':
    main()

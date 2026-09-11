"""
plot_core_vacc_compare.py
=========================

Collect the core-group vaccination results into something readable: one figure per outcome with
every arm overlaid (IQR band + median), plus a tidy summary CSV of the paired per-seed statistics.

The arms
--------
Each arm is a full 200k-agent, 1980-2055 run set produced by run_core_vacc.py:

    baseline   NHS_Vacc as modelled today
    top10      100% of the top 10% most active, general coverage cut to hold total doses fixed
    top20      the same at 20%

ARMS below lists them with the alpha they were run at. alpha is the Pareto tail index of the
partner-formation propensity: 3 is powerlaw.py's uncalibrated default, 2.05 is the value the
Natsal calibration selected (pinned at GAMMA_SHAPE_FLOOR, the finite-variance bound).

Why the comparison is paired
----------------------------
Arms at the same alpha share rand_seed, so a given seed gives the same network and the same
demography in every arm and only the vaccination policy differs. Differencing within a seed
removes most of the between-run variance, which is much larger than the effect. The summary
therefore reports the median PER-SEED percentage change and a two-sided sign test on the number of
seeds favouring the core arm -- the sign test uses only the direction of each paired difference, so
it is unaffected by the heavy-tailed spread in magnitude.

The comparison window matters: HPV cancers arise decades after infection, so a vaccination change
starting in 2008 cannot move cancer incidence until the reallocated cohorts reach cancer-bearing
age. 2035-2055 is the response window; 2008-2034 is reported alongside it as a placebo, and should
show nothing.

Reuses plot_IQR.py's draw_iqr/style_axis so these figures match the existing ones.

Usage
-----
    python plot_core_vacc_compare.py

Writes:
    figs/Power Law/core group/core_vacc_alpha2p05_<value>.png   one per outcome, arms overlaid
    figs/Power Law/core group/core_vacc_alpha_compare_<value>.png   alpha 3 vs 2.05, top10 only
    csvs/core_vacc_summary.csv                                  the paired statistics table
"""
import pathlib
from math import comb

import matplotlib.pyplot as plt
import pandas as pd

from plot_IQR import draw_iqr, style_axis

REPO = pathlib.Path(__file__).resolve().parent
CSV_DIR = REPO / 'csvs'
FIG_DIR = REPO / 'figs' / 'Power Law' / 'core group'

# (label, csv stem, colour). The baseline is the comparator for every core arm at the same alpha.
ARMS_205 = [
    ('baseline',   'powerlaw_alpha2p05_200k_baseline_vacc',   'tab:grey'),
    ('top 10%',    'powerlaw_alpha2p05_200k_top10_core_vacc', 'tab:blue'),
    ('top 20%',    'powerlaw_alpha2p05_200k_top20_core_vacc', 'tab:red'),
]
ARMS_A3 = [
    ('baseline',   'powerlaw_alpha3_200k_50runs_baseline_vacc', 'tab:grey'),
    ('top 10%',    'powerlaw_alpha3_200k_50runs_core_vacc',     'tab:blue'),
]

VALUES = ['cancer_incidence', 'infections', 'hpv_prevalence', 'n_vaccinated', 'cum_doses']

# (label, column, first year, last year, 'mean' or 'sum')
WINDOWS = [
    ('cancer_incidence 2035-2055', 'cancer_incidence', 2035, 2055, 'mean'),
    ('cancers 2035-2055',          'cancers',          2035, 2055, 'sum'),
    ('infections 2035-2055',       'infections',       2035, 2055, 'sum'),
    ('cancer_incidence 2008-2034 (placebo)', 'cancer_incidence', 2008, 2034, 'mean'),
    ('cum_doses (final)',          'cum_doses',        2055, 2055, 'mean'),
]


def load(stem):
    path = CSV_DIR / f'{stem}.csv'
    if not path.exists():
        return None
    return pd.read_csv(path)  # 'year' stays a column, which is what draw_iqr wants as its time axis


def sign_test_p(k, n):
    """Two-sided sign test: probability of k or more successes out of n under p=0.5."""
    if n == 0:
        return float('nan')
    tail = sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def aggregate(df, col, lo, hi, how):
    """One number per seed: the mean or total of `col` over [lo, hi]."""
    w = df[(df['year'] >= lo) & (df['year'] <= hi)].groupby('Seed')[col]
    return w.mean() if how == 'mean' else w.sum()


def paired_stats(base, arm, col, lo, hi, how):
    """Per-seed % change of arm vs baseline, and a sign test on the direction."""
    pb, pc = aggregate(base, col, lo, hi, how), aggregate(arm, col, lo, hi, how)
    idx = pb.index.intersection(pc.index)
    rel = (pc[idx] - pb[idx]) / pb[idx] * 100
    k, n = int((rel < 0).sum()), len(rel)
    return dict(median_pct=rel.median(), iqr_lo=rel.quantile(0.25), iqr_hi=rel.quantile(0.75),
                n_favour_core=k, n_runs=n, sign_p=sign_test_p(k, n))


def overlay(arms, value, title, outfile):
    """One axis, every arm's IQR band and median drawn on it."""
    present = [(lab, df, c) for lab, df, c in arms if df is not None and value in df.columns]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df, colour in present:
        draw_iqr(ax, df, value, time='year', color=colour, median_color=colour,
                 label_prefix=f'{label} ')
    style_axis(ax, value, 'year')
    ax.set_title(title)
    # draw_iqr emits an IQR and a Median entry per arm; keep them but shrink so 6 fit
    ax.legend(fontsize=7, ncol=len(present))
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / outfile, bbox_inches='tight', dpi=150)
    print(f'Saved {FIG_DIR / outfile}')
    plt.close(fig)
    return fig


def difference_plot(base, arms, value, title, outfile, response_from=2035, smooth=5):
    """
    The effect, which the levels plot cannot show. Core-group vaccination moves cancer incidence by
    ~2% on a curve that spans 0-65, so on shared axes the arms sit exactly on top of each other.

    Here each core arm is differenced against the baseline WITHIN each seed and then summarised
    across seeds, so the between-run variance -- which is far larger than the effect -- cancels
    instead of swamping it. Band is the IQR of the per-seed percentage difference; the line is its
    median. Anything below zero favours the core arm.
    """
    present = [(lab, df, c) for lab, df, c in arms if df is not None and value in df.columns]
    if not present or base is None or value not in base.columns:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df, colour in present:
        wide_b = base.pivot_table(index='year', columns='Seed', values=value)
        wide_c = df.pivot_table(index='year', columns='Seed', values=value)
        seeds = wide_b.columns.intersection(wide_c.columns)
        years = wide_b.index.intersection(wide_c.index)
        rel = (wide_c.loc[years, seeds] - wide_b.loc[years, seeds]) / wide_b.loc[years, seeds] * 100
        med, q25, q75 = rel.median(axis=1), rel.quantile(0.25, axis=1), rel.quantile(0.75, axis=1)
        # Cancer is a rare annual endpoint, so the year-to-year series is jagged enough to hide the
        # trend. Smooth the SUMMARY lines only (a centred rolling mean over `smooth` years) -- the
        # per-seed differences and every number in core_vacc_summary.csv are unsmoothed.
        if smooth and smooth > 1:
            roll = dict(window=smooth, center=True, min_periods=1)
            med, q25, q75 = med.rolling(**roll).mean(), q25.rolling(**roll).mean(), q75.rolling(**roll).mean()
        ax.fill_between(years, q25, q75, color=colour, alpha=0.20, label=f'{label} IQR')
        ax.plot(years, med, color=colour, linewidth=2, label=f'{label} median')

    ax.axhline(0, color='black', linewidth=1)
    ax.axvspan(response_from, 2055, color='gold', alpha=0.12,
               label=f'response window {response_from}-2055')
    ax.set_xlim(2008, 2055)  # the programme starts in 2008; before that the arms are identical
    ax.set_xlabel('year')
    ax.set_ylabel(f'{value}: % difference vs baseline'
                  + (f'  ({smooth}-yr smoothed)' if smooth and smooth > 1 else ''))
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / outfile, bbox_inches='tight', dpi=150)
    print(f'Saved {FIG_DIR / outfile}')
    plt.close(fig)
    return fig


def main():
    loaded_205 = [(lab, load(stem), c) for lab, stem, c in ARMS_205]
    loaded_a3 = [(lab, load(stem), c) for lab, stem, c in ARMS_A3]
    for lab, df, _ in loaded_205 + loaded_a3:
        if df is None:
            print(f'WARNING: missing CSV for {lab}, it will be left out')

    n_runs = {lab: df['Seed'].nunique() for lab, df, _ in loaded_205 if df is not None}
    print(f'alpha=2.05 arms: {n_runs}')

    # --- figures: all three alpha=2.05 arms on one axis, one figure per outcome ---
    for v in VALUES:
        overlay(loaded_205, v,
                f'{v} -- dose-neutral core-group vaccination (alpha=2.05, 200k agents)',
                f'core_vacc_alpha2p05_{v}.png')

    # --- figures: the effect itself, differenced within seed (this is the readable one) ---
    base_205 = dict((lab, df) for lab, df, _ in loaded_205)['baseline']
    core_205 = [(lab, df, c) for lab, df, c in loaded_205 if lab != 'baseline']
    for v in ('cancer_incidence', 'infections', 'cum_doses'):
        difference_plot(base_205, core_205, v,
                        f'{v} -- core arms vs baseline, paired by seed (alpha=2.05)',
                        f'core_vacc_alpha2p05_DIFF_{v}.png')

    # --- figures: does more heterogeneity change the picture? top10 at both alphas ---
    a3 = dict((lab, (df, c)) for lab, df, c in loaded_a3)
    a205 = dict((lab, (df, c)) for lab, df, c in loaded_205)
    for v in ('cancer_incidence', 'infections'):
        arms = []
        if a3.get('baseline', (None,))[0] is not None:
            arms += [('alpha=3 baseline', a3['baseline'][0], 'tab:grey'),
                     ('alpha=3 top 10%', a3['top 10%'][0], 'tab:green')]
        arms += [('alpha=2.05 baseline', a205['baseline'][0], 'black'),
                 ('alpha=2.05 top 10%', a205['top 10%'][0], 'tab:blue')]
        overlay(arms, v, f'{v} -- top 10% arm at both propensity heterogeneities',
                f'core_vacc_alpha_compare_{v}.png')

    # --- summary CSV: the paired statistics behind the figures ---
    rows = []
    for alpha, loaded in (('2.05', loaded_205), ('3', loaded_a3)):
        by_label = {lab: df for lab, df, _ in loaded}
        base = by_label.get('baseline')
        if base is None:
            continue
        for label, df in by_label.items():
            if label == 'baseline' or df is None:
                continue
            for wlabel, col, lo, hi, how in WINDOWS:
                if col not in df.columns or col not in base.columns:
                    continue
                rows.append(dict(alpha=alpha, arm=label, metric=wlabel,
                                 **paired_stats(base, df, col, lo, hi, how)))
    summary = pd.DataFrame(rows)
    out = CSV_DIR / 'core_vacc_summary.csv'
    summary.to_csv(out, index=False)

    print(f'\nPaired per-seed change vs the baseline at the same alpha (negative = core arm better)')
    with pd.option_context('display.width', 200, 'display.max_columns', 20):
        print(summary.to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()

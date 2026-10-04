"""
compare_r0_corrected_networks.py
================================

Pairwise statistics between the three R0-corrected runs (run_r0_corrected_1900.py; mean degree 1.4,
R0 1-1.5 per genotype, hpv16/hpv18/hi5 only, 100k agents, 1900-2050, 50 seeds each):

    Default   default_meandeg1p4_vaxgeno_100k_1900_2050_50runs.csv
    Gamma-2   gamma2_meandeg1p4_vaxgeno_100k_1900_2050_50runs.csv
    Power law powerlaw3p5_nogate_meandeg1p4_vaxgeno_100k_1900_2050_50runs.csv

One row per pair (Default vs Gamma-2, Default vs Power law, Gamma-2 vs Power law).

1. Overall prevalence time series (hpv_prevalence = all tracked types), as dtw.py does it: the
   median over runs per year, then
     * Pearson r between the two median series;
     * DTW distance between the z-normalised medians (dtw.py's dtw_normalised -- shape only);
     * DTW distance between the raw medians, in percentage points (shape AND level: z-normalising
       removes the level difference, which between these networks is large).
   No p-value is given for r: consecutive years are strongly autocorrelated, so the usual test's
   n-2 degrees of freedom would overstate the evidence. For scale, the same three measures between
   the medians of two disjoint halves (25 runs each) of ONE network give the noise floor -- what
   "the same network" looks like.

2. Cumulative cancers -- the sum of the yearly `cancers` result (new cervical cancers, scaled to the
   UK population) per run -- over 1900-2050 (overall), 1900-1979 (before screening starts in 1980,
   GlobalScreeningParameters.screening_start_year) and 1980-2050 (after). Welch's two-sample t-test
   on the 50 per-run totals (the networks' runs are independent; same seed numbers do not pair them),
   with Cohen's d and a Mann-Whitney U p-value as a distribution-free check. The cumulative incidence
   RATE (sum of the yearly cancer_incidence, per 100,000 women) is tested the same way and written to
   the CSV.

    python compare_r0_corrected_networks.py

Writes
    csvs/r0_corrected_network_comparison.csv          the pairwise table (one row per pair)
    csvs/r0_corrected_cancer_totals_by_network.csv    the per-network cumulative numbers
    csvs/r0_corrected_network_comparison.md           both, as markdown
    figs/R0 corrected/comparison/dtw_*.png             dtw.py's alignment plot per pair
"""
import itertools
import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy import stats

ROOT = pathlib.Path(__file__).parent
CSV_DIR = ROOT / 'csvs'
FIG_DIR = ROOT / 'figs' / 'R0 corrected' / 'comparison'

RUNS = {
    'Default': 'default_meandeg1p4_vaxgeno_100k_1900_2050_50runs',
    'Gamma-2': 'gamma2_meandeg1p4_vaxgeno_100k_1900_2050_50runs',
    'Power law': 'powerlaw3p5_nogate_meandeg1p4_vaxgeno_100k_1900_2050_50runs',
}
SEED_COL, YEAR_COL = 'Seed', 'year'
PREV_COL = 'hpv_prevalence'
SCREENING_START = 1980  # GlobalScreeningParameters.screening_start_year
PERIODS = {
    'overall': (1900, 2050),
    'pre-screening': (1900, SCREENING_START - 1),
    'post-screening': (SCREENING_START, 2050),
}
CANCER_MEASURES = {
    'cancers': 'cumulative cancers (UK-scaled count)',
    'cancer_incidence': 'cumulative incidence rate (sum of annual rates per 100,000 women)',
}
NOISE_SPLITS = 20  # random half-splits per network for the noise floor


# -------------------------------------------------------------------
# time series (as dtw.py)
# -------------------------------------------------------------------

def z_normalise(series):
    std = np.std(series)
    if std == 0:
        return series - np.mean(series)
    return (series - np.mean(series)) / std


def median_series(df, col=PREV_COL):
    return df.groupby(YEAR_COL)[col].median().sort_index()


def series_stats(x, y):
    """ Pearson r, z-normalised DTW (dtw.py), raw DTW in percentage points, on aligned years """
    x, y = np.asarray(x, float), np.asarray(y, float)
    return dict(pearson_r=float(stats.pearsonr(x, y)[0]),
                dtw_znorm=float(dtw.distance(z_normalise(x), z_normalise(y))),
                dtw_raw_pp=float(dtw.distance(x * 100, y * 100)))


def noise_floor(df, rng):
    """ series_stats between the medians of two disjoint 25-run halves, averaged over random splits """
    seeds = np.array(sorted(df[SEED_COL].unique()))
    rows = []
    for _ in range(NOISE_SPLITS):
        a = set(rng.choice(seeds, len(seeds) // 2, replace=False))
        m1 = median_series(df[df[SEED_COL].isin(a)])
        m2 = median_series(df[~df[SEED_COL].isin(a)])
        rows.append(series_stats(m1.to_numpy(), m2.reindex(m1.index).to_numpy()))
    return pd.DataFrame(rows).mean().to_dict()


def plot_dtw_alignment(x, y, years, label1, label2, out_png):
    """ dtw.py's alignment plot, on the raw series (percent), years on the x axis """
    x, y = np.asarray(x) * 100, np.asarray(y) * 100
    path = dtw.warping_path(x, y)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, j in path:
        ax.plot([years[i], years[j]], [x[i], y[j]], 'k-', alpha=0.15, lw=0.6)
    ax.plot(years, x, lw=2, label=label1)
    ax.plot(years, y, lw=2, label=label2)
    ax.axvline(SCREENING_START, color='0.4', lw=0.8, ls=':')
    ax.set_xlabel('year')
    ax.set_ylabel('HPV prevalence, all tracked types (%), median of 50 runs')
    ax.set_title(f'DTW alignment: {label1} vs {label2}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------
# cumulative cancers
# -------------------------------------------------------------------

def cumulative(df, col, period):
    lo, hi = PERIODS[period]
    d = df[(df[YEAR_COL] >= lo) & (df[YEAR_COL] <= hi)]
    return d.groupby(SEED_COL)[col].sum()


def describe_totals(v):
    v = np.asarray(v, float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return dict(n_runs=len(v), mean=v.mean(), sd=v.std(ddof=1), ci95_lo=v.mean() - half,
                ci95_hi=v.mean() + half, median=np.median(v), min=v.min(), max=v.max())


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return dict(mean_A=a.mean(), mean_B=b.mean(), diff=a.mean() - b.mean(),
                pct_diff=100 * (a.mean() - b.mean()) / b.mean(),
                t=float(t.statistic), df=float(dof), p=float(t.pvalue),
                cohens_d=float((a.mean() - b.mean()) / pooled),
                mannwhitney_p=float(stats.mannwhitneyu(a, b, alternative='two-sided').pvalue))


# -------------------------------------------------------------------
# main
# -------------------------------------------------------------------

def fmt_p(p):
    return f'{p:.2e}' if p < 1e-3 else f'{p:.3f}'


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for lab, tag in RUNS.items():
        df = pd.read_csv(CSV_DIR / f'{tag}.csv')
        if df.duplicated([SEED_COL, YEAR_COL]).any():
            raise ValueError(f'{tag}: duplicate (Seed, year) rows')
        data[lab] = df
        print(f'{lab:<10} {df[SEED_COL].nunique()} runs, {df[YEAR_COL].min():.0f}-{df[YEAR_COL].max():.0f}')

    med = {lab: median_series(df) for lab, df in data.items()}
    rng = np.random.default_rng(0)
    noise = {lab: noise_floor(df, rng) for lab, df in data.items()}

    # per-network cumulative numbers
    tot_rows = []
    for lab, df in data.items():
        for col in CANCER_MEASURES:
            for period in PERIODS:
                tot_rows.append(dict(network=lab, measure=col, period=period,
                                     years=f'{PERIODS[period][0]}-{PERIODS[period][1]}',
                                     **describe_totals(cumulative(df, col, period))))
    totals = pd.DataFrame(tot_rows)

    # pairwise table
    rows, long_rows = [], []
    for a, b in itertools.combinations(RUNS, 2):
        years = med[a].index.intersection(med[b].index)
        x, y = med[a].loc[years].to_numpy(), med[b].loc[years].to_numpy()
        row = dict(comparison=f'{a} vs {b}', n_years=len(years), **series_stats(x, y))
        plot_dtw_alignment(x, y, years.to_numpy(), a, b,
                           FIG_DIR / f"dtw_{a}_vs_{b}.png".replace(' ', '_').replace('-', ''))
        for col in CANCER_MEASURES:
            for period in PERIODS:
                w = welch(cumulative(data[a], col, period), cumulative(data[b], col, period))
                long_rows.append(dict(comparison=row['comparison'], measure=col, period=period, **w))
                if col == 'cancers':
                    key = period.replace('-', '_')
                    row.update({f'{key}_mean_A': w['mean_A'], f'{key}_mean_B': w['mean_B'],
                                f'{key}_pct_diff': w['pct_diff'], f'{key}_t': w['t'],
                                f'{key}_df': w['df'], f'{key}_p': w['p'], f'{key}_cohens_d': w['cohens_d']})
        rows.append(row)
    table = pd.DataFrame(rows)
    tests = pd.DataFrame(long_rows)

    table.to_csv(CSV_DIR / 'r0_corrected_network_comparison.csv', index=False)
    tests.to_csv(CSV_DIR / 'r0_corrected_network_comparison_tests_long.csv', index=False)
    totals.to_csv(CSV_DIR / 'r0_corrected_cancer_totals_by_network.csv', index=False)

    # markdown
    L = ['# R0-corrected networks: pairwise comparison', '',
         'Three runs of run_r0_corrected_1900.py (mean degree 1.4, R0 1-1.5 per genotype, hpv16/hpv18/hi5,',
         '100k agents, 1900-2050, 50 runs each). A = first network named in the row, B = second.', '',
         '## Overall HPV prevalence time series (median of 50 runs per year, 151 years)', '',
         '| Comparison | Pearson r | DTW, z-normalised (shape) | DTW, raw (pp; shape + level) |',
         '|---|---|---|---|']
    for _, r in table.iterrows():
        L.append(f"| {r['comparison']} | {r['pearson_r']:.3f} | {r['dtw_znorm']:.2f} | {r['dtw_raw_pp']:.2f} |")
    L += ['', 'Noise floor -- the same measures between two disjoint 25-run halves of one network '
          f'(mean of {NOISE_SPLITS} random splits):', '',
          '| Network | Pearson r | DTW, z-normalised | DTW, raw (pp) |', '|---|---|---|---|']
    for lab, n in noise.items():
        L.append(f"| {lab} | {n['pearson_r']:.3f} | {n['dtw_znorm']:.2f} | {n['dtw_raw_pp']:.2f} |")

    L += ['', '## Cumulative cancers (sum of yearly new cancers, UK-scaled), Welch t-test on 50 vs 50 runs', '']
    for period in PERIODS:
        lo, hi = PERIODS[period]
        L += [f'### {period} ({lo}-{hi})', '',
              '| Comparison | mean A | mean B | A - B | % diff | t | df | p | Cohen d | Mann-Whitney p |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for _, r in tests[(tests['measure'] == 'cancers') & (tests['period'] == period)].iterrows():
            L.append(f"| {r['comparison']} | {r['mean_A']:,.0f} | {r['mean_B']:,.0f} | {r['diff']:,.0f} | "
                     f"{r['pct_diff']:+.1f}% | {r['t']:.2f} | {r['df']:.1f} | {fmt_p(r['p'])} | "
                     f"{r['cohens_d']:.2f} | {fmt_p(r['mannwhitney_p'])} |")
        L.append('')
    L += ['## Per-network cumulative cancers (exact numbers)', '',
          '| Network | Period | Mean | SD | 95% CI of mean | Median | Min | Max |', '|---|---|---|---|---|---|---|---|']
    for _, r in totals[totals['measure'] == 'cancers'].iterrows():
        L.append(f"| {r['network']} | {r['period']} ({r['years']}) | {r['mean']:,.0f} | {r['sd']:,.0f} | "
                 f"{r['ci95_lo']:,.0f} - {r['ci95_hi']:,.0f} | {r['median']:,.0f} | {r['min']:,.0f} | {r['max']:,.0f} |")
    L += ['', '## Same tests on the cumulative incidence RATE (sum of annual cancer_incidence per 100,000 women)', '',
          '| Comparison | Period | mean A | mean B | % diff | t | p | Cohen d |', '|---|---|---|---|---|---|---|---|']
    for _, r in tests[tests['measure'] == 'cancer_incidence'].iterrows():
        L.append(f"| {r['comparison']} | {r['period']} | {r['mean_A']:,.1f} | {r['mean_B']:,.1f} | "
                 f"{r['pct_diff']:+.1f}% | {r['t']:.2f} | {fmt_p(r['p'])} | {r['cohens_d']:.2f} |")
    L += ['', 'Notes: Pearson r carries no p-value because consecutive years are autocorrelated. '
          'Pre-screening includes the 1900-1920 ramp-up from the seeded start (no cancers in 1900-01), '
          'the same for every network. Nine cancer tests per measure (3 pairs x 3 periods): a Bonferroni '
          'threshold is 0.05/9 = 0.0056.']
    md = '\n'.join(L) + '\n'
    (CSV_DIR / 'r0_corrected_network_comparison.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()

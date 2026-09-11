"""
analyse_ethnicity_uptake.py
===========================

Is the difference between run_ethnicity_uptake.py's two arms real, or is it run-to-run noise?

A difference of means with nothing attached says nothing on its own. This reads the per-run rows
that runner wrote and answers the question properly: for every metric and community, the mean in
each arm, the difference, and a Welch t-test on it (unequal variances, unpaired -- the arms share
seed NUMBERS but not realisations, since the interventions perturb the random stream from the
first vaccination onwards, so nothing about seed 3 in one arm pairs with seed 3 in the other).

    python analyse_ethnicity_uptake.py
    python analyse_ethnicity_uptake.py --tag partnership --alpha 0.01
    python analyse_ethnicity_uptake.py --csv

READ THE MDE COLUMN BEFORE ANYTHING ELSE. It is the minimum detectable effect: the smallest
difference this many runs could have distinguished from noise at 80% power. A change smaller than
the MDE is not evidence of no effect -- it is the run being too small to tell. The runs-needed
line at the end converts that into how many more runs would settle it.

Multiplicity: this tests ~11 metrics x 4 communities. At alpha=0.05 that is ~2 false positives
expected by chance alone, so a lone starred cell in a table this size is not a finding. A
Benjamini-Hochberg FDR correction across every test is applied and reported alongside the raw
p-values for that reason.

Input is the house-format CSV (year index, t, 1-D results, <result>_by_community_<label>, Seed),
so this reads exactly what plot_IQR.py reads.
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy import stats

CSV_DIR = pathlib.Path(__file__).with_name('csvs')
ARMS = ('equal', 'observed')
SUFFIX = '_by_community_'

# Stocks are read at the final year; flows are summed over the run and over the window.
STOCKS = ['hpv_prevalence', 'n_infectious', 'n_vaccinated', 'n_cancerous']
FLOWS = ['infections', 'cancers', 'cancer_deaths']
WINDOW_START = 2035


def find_csv(arm, tag, n_agents):
    matches = sorted(CSV_DIR.glob(f'ethnicity_{arm}_{tag}_{n_agents}.csv'))
    if not matches:
        matches = sorted(CSV_DIR.glob(f'ethnicity_{arm}_{tag}_*.csv'))
    if not matches:
        raise FileNotFoundError(
            f'No CSV for arm {arm!r} with tag {tag!r} in {CSV_DIR}. '
            f'Run run_ethnicity_uptake.py first, or pass --tag.')
    return matches[-1]


def communities(df):
    ''' Community labels present in the file, from the by-community column names. '''
    out = []
    for col in df.columns:
        if SUFFIX in col:
            label = col.split(SUFFIX, 1)[1]
            if label not in out:
                out.append(label)
    return out


def per_seed_metrics(df, window_start=WINDOW_START):
    '''
    One row per (community, seed), one column per metric, from the wide house-format frame.
    'Overall' is the population-wide (unstratified) column of the same name, carried alongside
    the communities so the national effect can be read in the same table.
    '''
    year = df['year'] if 'year' in df.columns else df.index.to_series()
    df = df.assign(_year=pd.to_numeric(year, errors='coerce'))
    rows = []
    for label in communities(df) + ['Overall']:
        suffix = '' if label == 'Overall' else f'{SUFFIX}{label}'
        for seed, g in df.groupby('Seed'):
            g = g.sort_values('_year')
            r = {'community': label, 'seed': int(seed)}
            for k in STOCKS:
                col = f'{k}{suffix}'
                if col in g:
                    r[k] = float(g[col].iloc[-1])
            for k in FLOWS:
                col = f'{k}{suffix}'
                if col in g:
                    r[k] = float(g[col].sum())
                    r[f'{k}_since_{window_start}'] = float(g.loc[g['_year'] >= window_start, col].sum())
            fem = f'n_females_alive{suffix}' if label != 'Overall' else 'n_females_alive'
            if fem in g and 'cancers' in r:
                women = float(g[fem].iloc[-1])
                r['cancers_per_100k_women'] = 1e5 * r['cancers'] / women if women else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def compare(a, b, alpha=0.05):
    '''
    Welch t-test on each (community, metric), plus the minimum detectable effect at 80% power.

    MDE uses the usual two-sample normal approximation, 2.8 = z(0.975) + z(0.80):
        MDE = 2.8 * sqrt(sd_a^2/n_a + sd_b^2/n_b)
    expressed as a percentage of the control mean. It answers the question a non-significant
    result actually raises -- how big would the effect have had to be for this many runs to see
    it -- and, rearranged, how many runs a given effect would need.
    '''
    metrics = [c for c in a.columns if c not in ('community', 'seed')]
    out = []
    for community in a['community'].unique():
        ga = a[a['community'] == community]
        gb = b[b['community'] == community]
        for m in metrics:
            xa, xb = ga[m].dropna().values, gb[m].dropna().values
            if len(xa) < 2 or len(xb) < 2:
                continue
            ma, mb = xa.mean(), xb.mean()
            se = np.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
            t, p = stats.ttest_ind(xb, xa, equal_var=False)
            change = 100 * (mb - ma) / ma if ma else np.nan
            mde = 100 * 2.8 * se / ma if ma else np.nan
            # Runs needed for THIS observed effect to clear the bar, at 80% power: MDE scales as
            # 1/sqrt(n), so n_needed = n * (mde/effect)^2
            n = min(len(xa), len(xb))
            needed = n * (mde / abs(change)) ** 2 if change and np.isfinite(change) and change else np.nan
            out.append(dict(community=community, metric=m, n=n, equal=ma, observed=mb,
                            change_pct=change, se_pct=100 * se / ma if ma else np.nan,
                            mde_pct=mde, runs_needed=needed, p=p))
    df = pd.DataFrame(out)
    order = np.argsort(df['p'].values)                      # Benjamini-Hochberg across the table
    ranks = np.empty(len(df), dtype=int)
    ranks[order] = np.arange(1, len(df) + 1)
    raw_q = np.minimum(1.0, df['p'].values * len(df) / ranks)
    q_sorted = np.minimum.accumulate(raw_q[order][::-1])[::-1]
    q = np.empty(len(df))
    q[order] = q_sorted
    df['q'] = q
    df['sig'] = np.where(df['q'] < alpha, '**', np.where(df['p'] < alpha, '*', ''))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', default='census', help="which run to read: 'census' or 'partnership'")
    ap.add_argument('--agents', default='*', help="e.g. 200k")
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--csv', action='store_true', help='write csvs/ethnicity_significance_<tag>.csv')
    args = ap.parse_args()

    frames = {}
    for arm in ARMS:
        path = find_csv(arm, args.tag, args.agents)
        print(f'{arm:<9} {path.name}')
        frames[arm] = per_seed_metrics(pd.read_csv(path, index_col=0))

    n_seeds = {arm: frames[arm]['seed'].nunique() for arm in ARMS}
    common = min(n_seeds.values())
    print(f'\nRuns per arm: {n_seeds}')
    print(f'Welch t-test, alpha={args.alpha}. "*" = raw p < alpha, "**" = survives '
          f'Benjamini-Hochberg across the whole table.')
    print('MDE = smallest change these runs could have detected at 80% power; "need" = runs per '
          'arm that\nwould be required for the observed change to clear that bar.\n')

    res = compare(frames['equal'], frames['observed'], alpha=args.alpha)
    for community in res['community'].unique():
        g = res[res['community'] == community]
        print(f'--- {community} ---')
        print(f"  {'metric':<28}{'equal':>15}{'observed':>15}{'change':>9}{'+-':>8}"
              f"{'MDE':>8}{'need':>7}{'p':>9}")
        for _, r in g.iterrows():
            fmt = ',.4f' if 'prevalence' in r['metric'] else ',.0f'
            need = f"{r['runs_needed']:>7.0f}" if np.isfinite(r['runs_needed']) else f"{'-':>7}"
            print(f"  {r['metric']:<28}{r['equal']:>15{fmt}}{r['observed']:>15{fmt}}"
                  f"{r['change_pct']:>8.1f}%{r['se_pct']:>7.1f}%{r['mde_pct']:>7.1f}%"
                  f"{need}{r['p']:>9.3f} {r['sig']}")
        print()

    hits = res[res['sig'] != '']
    print('=' * 78)
    if len(hits):
        print(f'{len(hits)} of {len(res)} tests reach p < {args.alpha} '
              f'({(res["sig"] == "**").sum()} survive FDR correction):')
        for _, r in hits.sort_values('p').iterrows():
            print(f"  {r['community']:<9} {r['metric']:<28} {r['change_pct']:>7.1f}%  "
                  f"p={r['p']:.4f}  q={r['q']:.3f} {r['sig']}")
    else:
        print(f'Nothing reaches p < {args.alpha}.')
    print(f'\nExpected false positives at alpha={args.alpha} with {len(res)} tests: '
          f'{args.alpha * len(res):.1f}   (runs per arm: {common})')

    if args.csv:
        out = CSV_DIR / f'ethnicity_significance_{args.tag}.csv'
        res.to_csv(out, index=False)
        print(f'Wrote {out}')
    return res


if __name__ == '__main__':
    main()

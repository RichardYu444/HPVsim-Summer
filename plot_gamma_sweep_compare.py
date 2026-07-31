import pathlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Overlay the gamma_shape sweep results: one figure per outcome, one
# median line (+ light IQR band) per shape value, to see the effect of
# varying partner-propensity heterogeneity at a glance.

OUT_DIR = pathlib.Path('figs/GammaSweep')

SHAPES = {
    0.05: 'gammashape_0p05.csv',
    0.25: 'gammashape_0p25.csv',
    1.0:  'gammashape_1p0.csv',
    2.5:  'gammashape_2p5.csv',
    5.0:  'gammashape_5p0.csv',
}

VALUES = [
    'hpv_prevalence', 'infections', 'cancer_incidence', 'n_vaccinated', 'n_cancer_treated'
]

COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']  # one per shape, low->high


def quantiles(df, value, time='year'):
    return (
        df.groupby(time)[value]
          .quantile([0.25, 0.5, 0.75])
          .unstack()
          .rename(columns={0.25: 'q25', 0.5: 'median', 0.75: 'q75'})
          .sort_index()
          .reset_index()
    )


def plot_compare(dfs_by_shape, value, time='year', title=None):
    fig, ax = plt.subplots()
    for (shape, df), color in zip(dfs_by_shape.items(), COLORS):
        q = quantiles(df, value, time)
        x = q[time].to_numpy()
        ax.fill_between(x, q['q25'], q['q75'], color=color, alpha=0.12)
        ax.plot(x, q['median'], linewidth=2, color=color, label=f'shape={shape}')

    ax.ticklabel_format(style='plain', axis='y')
    ax.set_xlim(left=1980, right=2050)
    if value.endswith('prevalence'):
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('Years Past' if time == 't' else time)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(value)
    ax.set_title(title or f'{value} by gamma_shape')
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.legend(title='gamma_shape')
    return fig, ax


if __name__ == '__main__':
    dfs_by_shape = {shape: pd.read_csv(path) for shape, path in SHAPES.items()}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for v in VALUES:
        fig, ax = plot_compare(dfs_by_shape, v, time='year')
        out_path = OUT_DIR / f'compare_{v}.png'
        fig.savefig(out_path)
        print(f'Saved {out_path}')

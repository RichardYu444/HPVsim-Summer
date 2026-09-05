import sys
import pathlib
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
#This is the code that I will use to transform my CSVs into plots

#these are the values we care about plotting
values = [
  'hpv_prevalence', 'infections', 'cancer_incidence', 'n_vaccinated', 'n_cancer_treated',
  # These exist only by community, so they produce a comparison grid but no collated plot
  'mean_degree', 'within_edge_frac',
]

SUFFIX = '_by_community'


def iqr_bands(df, value: str, time: str = 't'):
  '''Median and quartiles of `value` across runs at each timepoint.'''
  q = (
      df.groupby(time)[value]
        .quantile([0.25, 0.5, 0.75])
        .unstack()
        .rename(columns={0.25: 'q25', 0.5: 'median', 0.75: 'q75'})
        .sort_index()
        .reset_index()
  )
  return (q[time].to_numpy(), q['q25'].to_numpy(), q['median'].to_numpy(), q['q75'].to_numpy())


def is_rate(value: str):
  '''
  Whether `value` is a rate (per head of the community) rather than a count.

  This decides how the comparison grids are scaled. Rates are directly comparable between
  communities, so their panels share a y axis. Counts scale with community size -- here White is
  ~91% of the population and Chinese ~0.7% -- so a shared axis would flatten every minority panel
  into a line along the bottom, and each panel gets its own scale instead.
  '''
  return (value.endswith(('prevalence', 'incidence', 'mortality', 'frac'))
          or value.startswith('mean_degree')
          or value in ('cdr', 'cbr'))


def style_axis(ax, value: str, time: str, ylabel: str = None):
  '''Shared axis styling so the single plots and the comparison panels match.'''
  ax.ticklabel_format(style="plain", axis="y")
  ax.set_xlim(left=1980, right=2050)
  if value.endswith("prevalence") or value.endswith("frac") or value.startswith("mean_degree"):
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))
  else:
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
  #x axis label should be Years Past most of the time, but again incase we want the year instead
  if time == 't':
    ax.set_xlabel('Years Past')
  else:
    ax.set_xlabel(time)
  # Anchor at zero, but keep headroom above the data: set_ylim(bottom=0) on its own keeps
  # matplotlib's autoscaled top, which for a near-flat high series (e.g. White's
  # within_edge_frac, 0.972-0.976) leaves the line drawn along the top spine and effectively
  # invisible
  ax.set_ylim(0, ax.get_ylim()[1]*1.05)
  ax.set_ylabel(value if ylabel is None else ylabel)
  ax.grid(True, linewidth = 0.5, alpha = 0.4)
  return ax


def draw_iqr(ax, df, value: str, time: str = 't', color = 'lightblue', median_color = 'black',
             label_prefix: str = ''):
  '''Draw one IQR band + median line onto an existing axis.'''
  x, q25, med, q75 = iqr_bands(df, value, time)
  ax.fill_between(x, q25, q75, color = color, alpha = 0.3, label = f'{label_prefix}IQR')
  ax.plot(x, q25, linewidth = 1, color = color)
  ax.plot(x, med, linewidth = 2, label = f'{label_prefix}Median', color = median_color)  # prominent
  ax.plot(x, q75, linewidth = 1, color = color)
  return x, med


def plot_iqr(df, value: str, time: str = 't', title = 'IQR timeseries'): #time_col can be 'year' instead
  # Compute IQR + median across runs at each time t, we will use time steps instead of year for now
  fig, ax = plt.subplots()
  draw_iqr(ax, df, value, time)
  #plt.axhline(y= 4, color='r', linestyle='-', label = '2040 Target')
  style_axis(ax, value, time)
  ax.set_title(title)
  ax.legend()
  return fig, ax


def community_columns(df, value: str):
  '''
  The per-community columns for `value`, as {community label: column name}, in the order they
  appear in the CSV (which is the order the communities are indexed in the sim). Returns {} if
  this value was not exported by community.
  '''
  prefix = f'{value}{SUFFIX}_'
  return {col[len(prefix):]: col for col in df.columns if col.startswith(prefix)}


def plot_community_grid(df, value: str, time: str = 't', title = None, share_y = 'auto',
                        show_overall = 'auto'):
  '''
  One panel per community in a grid (2x2 for the usual four), each with its own IQR band and
  median, so the communities can be compared side by side.

  Args:
      share_y      : give every panel the same y limits, so panel heights are comparable.
                     'auto' (default) shares them for rates but not for counts -- see is_rate()
      show_overall : also draw the population-wide median as a dashed reference in each panel.
                     'auto' shows it for rates only: on a count panel with its own scale the
                     population total is far off the top and just squashes the community's line
  '''
  cols = community_columns(df, value)
  if not cols:
    return None
  rate = is_rate(value)
  if share_y == 'auto':
    share_y = rate
  if show_overall == 'auto':
    show_overall = rate
  n = len(cols)
  ncols = 2 if n <= 4 else math.ceil(math.sqrt(n))
  nrows = math.ceil(n / ncols)
  fig, axes = plt.subplots(nrows, ncols, figsize = (5.5*ncols, 4*nrows), squeeze = False)
  flat = axes.ravel()

  draw_overall = show_overall and value in df.columns
  if draw_overall:
    overall_x, _, overall_med, _ = iqr_bands(df, value, time)

  for ax, (label, col) in zip(flat, cols.items()):
    draw_iqr(ax, df, col, time)
    if draw_overall:
      ax.plot(overall_x, overall_med, linewidth = 1.2, linestyle = '--', color = 'firebrick',
              label = 'Overall median')
    style_axis(ax, value, time, ylabel = value)
    ax.set_title(label)
    ax.legend(fontsize = 8)

  # Blank out any unused panels (e.g. 3 communities in a 2x2 grid)
  for ax in flat[n:]:
    ax.axis('off')

  if share_y:
    top = max(ax.get_ylim()[1] for ax in flat[:n])
    for ax in flat[:n]:
      ax.set_ylim(0, top)

  scale_note = 'shared y axis' if share_y else 'independent y axes -- note the differing scales'
  fig.suptitle(f'{title or f"{value} by community"}  ({scale_note})', fontsize = 13)
  return fig


if __name__ == '__main__':
  # Always saves figs/<out_dir>/<out_prefix>_<value>.png per value, plus
  # figs/<out_dir>/'4-way compare_<out_prefix>_<value>.png' for any value that was also
  # exported by community.
  # With args: python plot_IQR.py <csv_path> <out_prefix> [out_dir]
  if len(sys.argv) >= 3:
    csv_path, out_prefix = sys.argv[1], sys.argv[2]
    out_dir = pathlib.Path(sys.argv[3]) if len(sys.argv) >= 4 else pathlib.Path('figs/GammaSweep')
  else:
    csv_path=r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\csvs\community_gamma2_50runs.csv'
    out_prefix = 'community_gamma2_50runs'
    out_dir = pathlib.Path(r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer\figs')

  df = pd.read_csv(csv_path)
  print(df)

  out_dir.mkdir(parents=True, exist_ok=True)

  figs = []
  for v in values:
    ccols = community_columns(df, v)
    if v not in df.columns and not ccols:
      print(f'Skipping {v}: neither {v} nor {v}{SUFFIX}_* is a column in {csv_path}')
      continue

    # The usual collated plot, for anything with a population-wide column. Some values exist
    # only by community (mean_degree, within_edge_frac, single_frac): those get the grid only
    if v in df.columns:
      fig, ax = plot_iqr(df, value = v, time = 'year', title = f"{v}")
      figs.append(fig)
      # tight_layout per-figure (plt.tight_layout only touches the current one)
      # and bbox_inches='tight' so wide y tick labels aren't clipped.
      fig.tight_layout()
      fig.savefig(out_dir / f'{out_prefix}_{v}.png', bbox_inches='tight')
      print(f'Saved {out_dir / f"{out_prefix}_{v}.png"}')
    else:
      print(f'{v}: community-only, no collated plot')

    # Community comparison grid, if this value was exported by community
    cgrid = plot_community_grid(df, value = v, time = 'year')
    if cgrid is None:
      print(f'  (no {v}{SUFFIX}_* columns, skipping the comparison grid)')
      continue
    n = len(ccols)
    figs.append(cgrid)
    cgrid.tight_layout()
    cname = f'{n}-way compare_{out_prefix}_{v}.png'
    cgrid.savefig(out_dir / cname, bbox_inches='tight')
    print(f'Saved {out_dir / cname}')

#!/usr/bin/env python3
"""
plot_powerlaw_network_stats
=============================

Plots the network diagnostics saved by run_powerlaw_validation.py
(network_stats_community_powerlaw.npz) in the same visual style as
default_network_testing.py's multi-panel PMF/duration figure -- log-scale PMF lines for
degree distributions, step histograms with target reference lines for durations, 'c0'/'c3'
short-vs-long colouring (matching that file's casual/marital colours, since 's'<->'c' and
'l'<->'m' are already the established layer correspondence in this project).

Only covers what run_powerlaw_validation.py actually recorded -- a single run, so there's
no early/late-window split like default_network_testing.py's two-seed-point comparison, and
no by-type instantaneous/annual degree or standing-fraction time series (those weren't
saved). Four panels instead of that file's eight:
    1. Annual degree distribution (final year, all activity levels)
    2. Instantaneous degree distribution (final timestep)
    3. Completed partnership durations by type, with the calibrated D_mean_short/D_mean_long
       target lines (see basePars_community_powerlaw.py's community_pars)
    4. Mean annual degree over the whole run, with the calibrated mean_partners_per_year
       target line

This script only loads the saved .npz -- it does not re-run the simulation.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from basePars_community_powerlaw import community_pars

NETWORK_STATS_PATH = 'network_stats_community_powerlaw.npz'
OUT_PNG = 'community_powerlaw_network_stats.png'

TYPE_LABEL = {'s': 'short', 'l': 'long'}
TYPE_COLOR = {'s': 'C0', 'l': 'C3'}


def pmf_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    return counts / total if total else counts


def plot_degree_pmf(ax, counts, max_degree_label, title):
    x = np.arange(counts.size)
    labels = [str(k) for k in x[:-1]] + [f'{max_degree_label}+']
    p = pmf_from_counts(counts)
    ax.plot(x, p, marker='o', markersize=4, color='C2')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_title(title)
    ax.set_xlabel('distinct partners')
    ax.set_ylabel('fraction of people')
    ax.set_yscale('log')


def main():
    d = np.load(NETWORK_STATS_PATH)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    max_degree_label = d['annual_degree_dist_end'].size - 1  # last bin is this value's overflow
    plot_degree_pmf(
        axes[0, 0], d['annual_degree_dist_end'], max_degree_label,
        '1. Annual degree distribution\n(distinct partners in the final year)',
    )
    plot_degree_pmf(
        axes[0, 1], d['instant_degree_dist_end'], max_degree_label,
        '2. Instantaneous degree distribution\n(current partners, final timestep)',
    )

    ax = axes[1, 0]
    dur_short = d['duration_short_months']
    dur_long = d['duration_long_months']
    target_short = community_pars['D_mean_short']
    target_long = community_pars['D_mean_long']
    if dur_short.size or dur_long.size:
        maximum = int(max(dur_short.max() if dur_short.size else 0,
                           dur_long.max() if dur_long.size else 0, target_long))
        bins = np.linspace(0, maximum, 60)
        if dur_short.size:
            ax.hist(dur_short, bins=bins, density=True, histtype='step', linewidth=2,
                    color=TYPE_COLOR['s'], label=f"{TYPE_LABEL['s']} (n={dur_short.size})")
        if dur_long.size:
            ax.hist(dur_long, bins=bins, density=True, histtype='step', linewidth=2,
                    color=TYPE_COLOR['l'], label=f"{TYPE_LABEL['l']} (n={dur_long.size})")
    ax.axvline(target_short, linestyle='--', color=TYPE_COLOR['s'], alpha=0.7,
               label=f'short target ({target_short:.0f} mo)')
    ax.axvline(target_long, linestyle='--', color=TYPE_COLOR['l'], alpha=0.7,
               label=f'long target ({target_long:.0f} mo)')
    ax.set_title('3. Completed partnership durations, by type\n'
                  '(left-censored initial-snapshot and right-censored still-active\n'
                  'partnerships excluded -- long-type mean is understated, see caption)')
    ax.set_xlabel('duration in months')
    ax.set_ylabel('fraction of completed partnerships')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    years = d['annual_degree_years']
    values = d['annual_degree_values']
    target_annual = community_pars['mean_partners_per_year']
    ax.plot(years, values, 'o-', color='C2', markersize=3, label='realised')
    ax.axhline(target_annual, linestyle='--', color='k', alpha=0.6,
               label=f'calibration target ({target_annual:.3f})')
    ax.set_title('4. Mean annual degree over time\n(excluding singles)')
    ax.set_xlabel('year')
    ax.set_ylabel('mean distinct partners')
    ax.legend(fontsize=8)

    n_completed = int(d['long_duration_n_completed'])
    n_censored = int(d['long_duration_n_censored'])
    fig.suptitle(
        'Community network (power-law) -- calibrated run diagnostics\n'
        f"long-type partnerships: {n_completed} completed (mean {float(d['long_duration_mean_months']):.0f} mo, "
        f"observed only -- censored, true target {target_long:.0f} mo), "
        f'{n_censored} still active/censored at sim end',
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_PNG, dpi=120)
    print(f'saved figure -> {OUT_PNG}')


if __name__ == '__main__':
    main()

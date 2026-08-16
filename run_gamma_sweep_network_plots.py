import pathlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import hpvsim_working as hpv
from basePars_community import base_pars_geno

# Reuse community_testing.py's analyzers/helpers rather than duplicating them -- same
# pattern run_community_validation.py already uses (`from community_testing import
# _MixingTracker`). Nothing here monkeypatches the theta sampler, so it's safe to import
# directly (unlike the powerlaw.py modules -- see run_community_validation.py's note).
from community_testing import (
    _ActiveTracker, _MixingTracker, degree_from_edges, added_edges_dict, plot_pmf,
    compute_duration_stats, safe_mean, LKEY_SHORT, LKEY_LONG,
    TYPE_LABEL, TYPE_COLOR, WINDOW_STYLE, WINDOW_COLOR,
)

# -------------------------------------------------------------------
# Network-only diagnostics (community_testing.py's Figure 1 -- the
# "*_distributions.png" degree/duration panel) for each gamma_shape swept in
# run_sim_gamma_sweep.py, run on basePars_community.py's own production pars
# (ethnicity communities, real durations/mixing) rather than community_testing.py's
# toy COMMUNITY_PARS. Epidemic dynamics are switched off (init_hpv_prev=0,
# interventions=[]) since only network structure is being inspected -- disease/
# genotype pars are otherwise left in place because hpvsim_working's Sim needs them
# wired up regardless.
# -------------------------------------------------------------------

OUTPUT_DIR = r'C:\Users\richa\OneDrive - Nexus365\Documents\HPV sim Project\Summer'
SEED = 0

GAMMA_SHAPES = [0.05, 0.25, 1.0, 2.5, 5.0]  # same sweep as run_sim_gamma_sweep.py

EARLY_YEAR = 3
LATE_YEAR = 50

ZERO_INIT_PREV = dict(
    age_brackets=np.array([150.0]),
    m=np.array([0.0]),
    f=np.array([0.0]),
)


def shape_tag(shape):
    return str(shape).replace('.', 'p')


def year_of(sim, t):
    return int(sim.yearvec[t] - sim['start']) + 1


def window_of(year):
    if year == EARLY_YEAR:
        return 'early'
    if year == LATE_YEAR:
        return 'late'
    return None


def band_labels(params):
    edges = params['age_band_edges']
    labels = [f'<{int(edges[0])}']
    labels += [f'{int(lo)}-{int(hi)}' for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f'{int(edges[-1])}+')
    return labels


def build_sim(shape):
    pars = sc.dcp(base_pars_geno)
    pars['community_pars']['gamma_shape'] = shape
    pars['init_hpv_prev'] = sc.dcp(ZERO_INIT_PREV)
    pars['interventions'] = []
    pars['rand_seed'] = SEED
    pars['verbose'] = 0
    pars['analyzers'] = [hpv.network_history(), _ActiveTracker(), _MixingTracker()]
    return hpv.Sim(pars, label=f'community network (network-only, gamma_shape={shape})')


def run_for_shape(shape, outdir):
    tag = shape_tag(shape)
    out_png = outdir / f'network_stats_gamma{tag}_distributions.png'
    print(f'=== gamma_shape = {shape} -> {out_png.name} ===')

    sim = build_sim(shape)
    sim.run()

    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    layer_map = nh.layer_map
    n_uid = len(sim.people)
    years = int(round(sim['end'] - sim['start']))

    be = sim.network_backend
    target_k = be._params['mean_partners_per_year']
    target_f = be._params['frac_long_target']
    p_form_long = be._params['p_form_long']

    instantaneous_samples = {'early': [], 'late': []}
    instantaneous_type_samples = {lkey: {'early': [], 'late': []} for lkey in (LKEY_SHORT, LKEY_LONG)}
    integrated_samples = {'early': [], 'late': []}
    yearly_samples = {'early': None, 'late': None}
    yearly_type_samples = {lkey: {'early': None, 'late': None} for lkey in (LKEY_SHORT, LKEY_LONG)}

    instantaneous_mean_t, integrated_mean_t, yearly_mean_t = [], [], []
    standing_long_t, quarterly_long_fracs = [], []

    running_year = None
    annual_base, annual_added = {}, {}
    annual_obs_f, annual_obs_m = set(), set()

    def finalize_year(window):
        annual_union = {**annual_base, **annual_added}
        deg = degree_from_edges(annual_union, n_uid)
        counts = np.concatenate([deg[sorted(annual_obs_f)], deg[sorted(annual_obs_m)]])
        yearly_mean_t.append(counts.mean() if counts.size else 0.0)
        if window:
            yearly_samples[window] = counts
            for lkey in (LKEY_SHORT, LKEY_LONG):
                deg_t = degree_from_edges(annual_union, n_uid, lkey=lkey)
                yearly_type_samples[lkey][window] = np.concatenate(
                    [deg_t[sorted(annual_obs_f)], deg_t[sorted(annual_obs_m)]])
        return

    for t in range(sim.npts):
        year = year_of(sim, t)
        window = window_of(year)

        if year != running_year:
            if running_year is not None:
                finalize_year(window_of(running_year))
            running_year = year
            annual_base = nh.edges_at(t - 1) if t > 0 else {}
            annual_added = {}
            annual_obs_f, annual_obs_m = set(), set()

        edges_now = nh.edges_at(t)
        active_f, active_m = act.active_female[t], act.active_male[t]

        deg = degree_from_edges(edges_now, n_uid)
        pooled = np.concatenate([deg[active_f], deg[active_m]])
        instantaneous_mean_t.append(pooled.mean() if pooled.size else 0.0)
        if window:
            instantaneous_samples[window].append(pooled)
        for lkey in (LKEY_SHORT, LKEY_LONG):
            deg_t = degree_from_edges(edges_now, n_uid, lkey=lkey)
            if window:
                instantaneous_type_samples[lkey][window].append(
                    np.concatenate([deg_t[active_f], deg_t[active_m]]))

        n_edges_now = len(edges_now)
        n_long_now = sum(1 for _, _, lkey in edges_now.values() if lkey == LKEY_LONG)
        standing_long_t.append(n_long_now / n_edges_now if n_edges_now else 0.0)

        this_delta = nh.deltas.get(t)
        added_this_step = added_edges_dict(this_delta, layer_map)
        prev_live = nh.edges_at(t - 1) if t > 0 else {}
        quarter_union = {**prev_live, **added_this_step}
        deg_q = degree_from_edges(quarter_union, n_uid)
        pooled_q = np.concatenate([deg_q[active_f], deg_q[active_m]])
        integrated_mean_t.append(pooled_q.mean() if pooled_q.size else 0.0)
        if window:
            integrated_samples[window].append(pooled_q)
        n_q = len(quarter_union)
        n_q_long = sum(1 for _, _, lkey in quarter_union.values() if lkey == LKEY_LONG)
        quarterly_long_fracs.append(n_q_long / n_q if n_q else 0.0)

        annual_added.update(added_this_step)
        annual_obs_f.update(active_f.tolist())
        annual_obs_m.update(active_m.tolist())

    finalize_year(window_of(running_year))

    durations = compute_duration_stats(nh)
    short_durations = np.asarray(durations[LKEY_SHORT], dtype=float)
    long_durations = np.asarray(durations[LKEY_LONG], dtype=float)

    instantaneous_pool = {label: np.concatenate(instantaneous_samples[label]) for label in ('early', 'late')}
    instantaneous_type_pool = {
        lkey: {label: np.concatenate(instantaneous_type_samples[lkey][label]) for label in ('early', 'late')}
        for lkey in (LKEY_SHORT, LKEY_LONG)
    }
    integrated_pool = {label: np.concatenate(integrated_samples[label]) for label in ('early', 'late')}

    print(f'mean partners/year: realised {np.mean(yearly_mean_t):.3f}; target {target_k:.3f}')
    print(f'standing long fraction: realised {np.mean(standing_long_t):.3f}; target {target_f:.3f}')
    print(f'completed durations: short {safe_mean(short_durations):.2f} months; '
          f'long {safe_mean(long_durations):.2f} months')
    print(f'mean degree: instantaneous {np.mean(instantaneous_mean_t):.3f}; '
          f'quarterly union {np.mean(integrated_mean_t):.3f}')

    window_label = {'early': f'year {EARLY_YEAR}', 'late': f'year {LATE_YEAR}'}
    cp = sim['community_pars']

    fig, axes = plt.subplots(2, 4, figsize=(21, 10))

    plot_pmf(
        axes[0, 0],
        [instantaneous_pool['early'], instantaneous_pool['late']],
        [window_label['early'], window_label['late']],
        [WINDOW_COLOR['early'], WINDOW_COLOR['late']],
        ['-', '-'], ['o', 'o'],
        'current partners',
        '1. Instantaneous degree\n(HPVsim quarterly steps pooled within year)',
    )

    plot_pmf(
        axes[0, 1],
        [integrated_pool['early'], integrated_pool['late']],
        [window_label['early'], window_label['late']],
        [WINDOW_COLOR['early'], WINDOW_COLOR['late']],
        ['-', '-'], ['s', 's'],
        'partners in the quarter',
        '2. Quarterly degree\n(union over one HPVsim step)',
    )

    ax = axes[0, 2]
    maximum = int(max(yearly_samples['early'].max(), yearly_samples['late'].max()))
    bins = np.arange(maximum + 2) - 0.5
    for label in ('early', 'late'):
        ax.hist(yearly_samples[label], bins=bins, density=True, histtype='step',
                linewidth=2, color=WINDOW_COLOR[label], label=window_label[label])
    ax.axvline(target_k, linestyle='--', color='gray', label='target mean')
    ax.set_title('3. Annual degree\n(distinct partners in 12 months)')
    ax.set_xlabel('distinct partners')
    ax.set_ylabel('fraction of people')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    type_inst_samples, type_inst_labels, type_inst_colors, type_inst_styles, type_inst_markers = [], [], [], [], []
    for lkey in (LKEY_SHORT, LKEY_LONG):
        for label in ('early', 'late'):
            type_inst_samples.append(instantaneous_type_pool[lkey][label])
            type_inst_labels.append(f'{TYPE_LABEL[lkey]}, {window_label[label]}')
            type_inst_colors.append(TYPE_COLOR[lkey])
            type_inst_styles.append(WINDOW_STYLE[label])
            type_inst_markers.append('o' if lkey == LKEY_SHORT else 's')
    plot_pmf(
        axes[0, 3], type_inst_samples, type_inst_labels,
        type_inst_colors, type_inst_styles, type_inst_markers,
        'current partners of the selected type',
        '4. Instantaneous degree by type',
    )

    ax = axes[1, 0]
    if short_durations.size or long_durations.size:
        maximum = int(max(short_durations.max() if short_durations.size else 0,
                           long_durations.max() if long_durations.size else 0))
        bins = np.arange(maximum + 2) - 0.5
        if short_durations.size:
            ax.hist(short_durations, bins=bins, density=True, histtype='step',
                    linewidth=2, color=TYPE_COLOR[LKEY_SHORT], label='short')
        if long_durations.size:
            ax.hist(long_durations, bins=bins, density=True, histtype='step',
                    linewidth=2, color=TYPE_COLOR[LKEY_LONG], label='long')
    ax.axvline(cp['D_mean_short'], linestyle='--', color=TYPE_COLOR[LKEY_SHORT], alpha=0.6)
    ax.axvline(cp['D_mean_long'], linestyle='--', color=TYPE_COLOR[LKEY_LONG], alpha=0.6)
    ax.set_title('5. Completed partnership durations\n(HPVsim-step resolution)')
    ax.set_xlabel('duration in months')
    ax.set_ylabel('fraction of completed partnerships')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    years_axis = np.arange(0, years + 1)
    ax.axhline(target_k, linestyle='--', color='gray', label='target')
    ax.plot(years_axis[:len(yearly_mean_t)], yearly_mean_t, 'o-', color='C2', label='realised')
    ax.set_ylim(0, max(max(yearly_mean_t), target_k) * 1.3)
    ax.set_title('6. Mean annual degree')
    ax.set_xlabel('measurement year')
    ax.set_ylabel('mean distinct partners')
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    steps = np.arange(sim.npts)
    ax.plot(steps, standing_long_t, color=TYPE_COLOR[LKEY_LONG], linewidth=1.2, label='standing fraction')
    ax.axhline(target_f, linestyle='--', color='gray', label='target')
    ax.axhline(p_form_long, linestyle=':', color=TYPE_COLOR[LKEY_SHORT],
               label=f'formation probability = {p_form_long:.2f}')
    ax.set_ylim(0, 1)
    ax.set_title('7. Standing long-partnership fraction')
    ax.set_xlabel('HPVsim timestep')
    ax.set_ylabel('fraction of active edges')
    ax.legend(loc='center right', fontsize=8)

    annual_type_samples, annual_type_labels, annual_type_colors = [], [], []
    annual_type_styles, annual_type_markers = [], []
    for lkey in (LKEY_SHORT, LKEY_LONG):
        for label in ('early', 'late'):
            annual_type_samples.append(yearly_type_samples[lkey][label])
            annual_type_labels.append(f'{TYPE_LABEL[lkey]}, {window_label[label]}')
            annual_type_colors.append(TYPE_COLOR[lkey])
            annual_type_styles.append(WINDOW_STYLE[label])
            annual_type_markers.append('o' if lkey == LKEY_SHORT else 's')
    plot_pmf(
        axes[1, 3], annual_type_samples, annual_type_labels,
        annual_type_colors, annual_type_styles, annual_type_markers,
        'distinct partners of the selected type',
        '8. Annual degree by type\n(12-month union)',
    )

    fig.suptitle(
        f'Community network (basePars_community, network-only) -- '
        f"n_agents={sim['n_agents']}, partners/year={target_k}, gamma_shape={shape}, "
        f"durations={cp['D_mean_short']:.1f}/{cp['D_mean_long']:.1f} months, "
        f'standing long fraction={target_f}',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f'saved figure -> {out_png}')


def main():
    outdir = pathlib.Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    for shape in GAMMA_SHAPES:
        run_for_shape(shape, outdir)
    print('All gamma_shape network-stats plots done.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
fran_testing
============

Mirrors the diagnostics in "Francesco simple codes/bipartite_network_model_bundle/
demo_bipartite_distributions.py", but runs the bipartite model *inside* HPVsim (via
pars['network'] == 'francesco' / FrancescoNetworkBackend, see francesco_network.py)
instead of driving bipartite_network_model.py directly. All network statistics below
are reconstructed purely from the network_history analyzer's deltas
(NetworkDelta.added_edges/removed_edges, replayed via edges_at()/nodes_at()) -- nothing
here reaches into FrancescoNetworkBackend's internal bundle state.

Two things necessarily differ from the standalone demo, both flagged inline as they
come up:

1. Resolution. The bundle's own internal clock is monthly; HPVsim's dt (0.25 years =
   3 months here) is the finest resolution anything outside FrancescoNetworkBackend can
   observe. So "instantaneous degree" below is sampled once per HPVsim step (quarterly)
   rather than once per bundle-month, and "quarterly degree" collapses onto exactly one
   HPVsim step instead of being a separate 3-month aggregation.

2. Network-node eligibility. NetworkDelta's node bookkeeping tracks literal HPVsim
   birth/death (see network.py's ENTRY_*/NODE_DEATH_* codes) -- it has no concept of
   sexual debut, so it can't say who currently holds a bipartite U/V node. A small
   companion analyzer (_ActiveTracker) fills that one gap by recording
   sim.people.is_active each step; everything about *edges* still comes from
   network_history.

Durations (panel 5) only count partnerships whose formation was actually observed
within the run: edges present in the t=0 initial snapshot have an unknown true
formation time (they may have formed at any point during FrancescoNetworkBackend's own
pre-t=0 burn-in) and are excluded as left-censored, mirroring standard survival-analysis
practice. Partnerships still active at the end of the run are right-censored and
excluded the same way the demo excludes them.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import hpvsim_working as hpv

# Same USER_PARAMS as demo_bipartite_distributions.py's USER_PARAMS, expressed via the keys
# pars['francesco_pars'] / bipartite_network_model.interpretable_to_params expect ('frac_long'
# is the same quantity the demo calls 'pi_long').
USER_PARAMS = dict(
    mean_partners_per_year=3.0,
    gamma_shape=3,
    D_mean_short=2.0,
    D_mean_long=36.0,
    frac_long=0.5,
    tau=360.0,
)

N_AGENTS = 200_000
START = 2015
YEARS = 12
DT = 0.25  
EARLY_YEAR = 3
LATE_YEAR = 10
SEED = 1
OUT_PNG = 'fran_testing_distributions2.png'

LKEY_SHORT, LKEY_LONG = 's', 'l'
TYPE_LABEL = {LKEY_SHORT: 'short', LKEY_LONG: 'long'}
TYPE_COLOR = {LKEY_SHORT: 'C0', LKEY_LONG: 'C3'}
WINDOW_STYLE = {'early': '-', 'late': '--'}
WINDOW_COLOR = {'early': 'C0', 'late': 'C3'}


class _ActiveTracker(hpv.Analyzer):
    '''
    Records which people are sexually active (is_active), split by sex, at every timestep.

    network_history's NetworkDelta only tracks literal HPVsim birth/death -- it has no debut
    field -- so it can't say who currently holds a bipartite U/V node. This is the one piece of
    "who's eligible" bookkeeping that has to come from sim.people directly rather than from
    network deltas; every *edge* statistic below still comes from network_history.
    '''

    def __init__(self, label='active_tracker'):
        super().__init__(label=label)
        self.active_female = {}  # t -> uid array
        self.active_male = {}
        return

    def initialize(self, sim):
        super().initialize(sim)
        self._record(sim, t=0)
        return

    def apply(self, sim):
        self._record(sim, t=sim.t)
        return

    def _record(self, sim, t):
        ppl = sim.people
        act = ppl.is_active
        self.active_female[t] = np.where(act & ppl.is_female)[0].astype(np.int64)
        self.active_male[t] = np.where(act & ppl.is_male)[0].astype(np.int64)
        return


def added_edges_dict(delta, layer_map):
    ''' {eid: (f, m, lkey)} for one delta's added_edges (or {} if delta is None) '''
    out = {}
    if delta is None:
        return out
    for eid, f, m, layer in zip(delta.added_edges.eid, delta.added_edges.f,
                                 delta.added_edges.m, delta.added_edges.layer):
        out[int(eid)] = (int(f), int(m), layer_map[layer])
    return out


def degree_from_edges(edges, n, lkey=None):
    ''' Pooled degree array (length n) from an {eid: (f, m, lkey)} dict, optionally by layer '''
    deg = np.zeros(n, dtype=np.int64)
    for f, m, layer in edges.values():
        if lkey is not None and layer != lkey:
            continue
        deg[f] += 1
        deg[m] += 1
    return deg


def pmf(values, maximum):
    values = np.asarray(values, dtype=np.int64)
    histogram = np.bincount(values, minlength=maximum + 1)[:maximum + 1].astype(float)
    total = histogram.sum()
    return histogram / total if total else histogram


def plot_pmf(ax, samples, labels, colors, styles, markers, xlabel, title):
    maximum = max((int(np.max(s)) for s in samples if np.asarray(s).size), default=0)
    x = np.arange(maximum + 1)
    for sample, label, color, style, marker in zip(samples, labels, colors, styles, markers):
        ax.plot(x, pmf(sample, maximum), linestyle=style, marker=marker,
                markersize=3.5, color=color, alpha=0.85, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('fraction of people')
    ax.set_yscale('log')
    ax.legend(fontsize=8)


def safe_mean(values):
    values = np.asarray(values)
    return float(values.mean()) if values.size else float('nan')


def year_of(sim, t):
    return int(sim.yearvec[t] - sim['start']) + 1


def window_of(year):
    if year == EARLY_YEAR:
        return 'early'
    if year == LATE_YEAR:
        return 'late'
    return None


def compute_duration_stats(nh):
    '''
    Completed-partnership durations (in months) by type, reconstructed purely from eid
    add/remove events in the delta history. Edges present in the t=0 initial snapshot are
    excluded (left-censored -- true formation time unknown); edges still live at the end of
    the run are excluded too (right-censored), matching the demo's own convention.
    '''
    left_censored = set(int(e) for e in nh.initial_snapshot.added_edges.eid)
    born_at = {}  # eid -> (t, lkey)
    durations = {LKEY_SHORT: [], LKEY_LONG: []}
    for t in sorted(nh.deltas.keys()):
        delta = nh.deltas[t]
        for eid, layer in zip(delta.added_edges.eid, delta.added_edges.layer):
            born_at[int(eid)] = (t, nh.layer_map[layer])
        for eid in delta.removed_edges.eid:
            eid = int(eid)
            if eid in left_censored:
                continue
            info = born_at.pop(eid, None)
            if info is None:
                continue  # Defensive: shouldn't happen (every non-left-censored eid was added first)
            born_t, lkey = info
            durations[lkey].append((t - born_t) * DT * 12.0)
    return durations


def main():
    sim = hpv.Sim(
        location='united kingdom',
        network='francesco',
        n_agents=N_AGENTS,
        start=START,
        n_years=YEARS,
        dt=DT,
        rand_seed=SEED,
        verbose=0,
        francesco_pars=USER_PARAMS,
        analyzers=[hpv.network_history(), _ActiveTracker()],
    )
    sim.run()

    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    layer_map = nh.layer_map
    n_uid = len(sim.people)

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

        # --- Panel 1/4 inputs: instantaneous degree, pooled and by type ---
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

        # --- Panel 7 input: standing long fraction ---
        n_edges_now = len(edges_now)
        n_long_now = sum(1 for _, _, lkey in edges_now.values() if lkey == LKEY_LONG)
        standing_long_t.append(n_long_now / n_edges_now if n_edges_now else 0.0)

        # --- Panel 2 input: quarterly degree = union of (live at end of prev step, added this step) ---
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

        # --- Panel 3/8 accumulation: annual union + observed population ---
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

    print('\n' + '=' * 68)
    print('FRANCESCO NETWORK, VIA HPVSIM -- SUMMARY')
    print('=' * 68)
    print(f'mean partners/year: realised {np.mean(yearly_mean_t):.3f}; target {target_k:.3f}')
    print(f'standing long fraction: realised {np.mean(standing_long_t):.3f}; '
          f'target {target_f:.3f}; formation probability {p_form_long:.3f}')
    print(f'integrated quarterly long fraction: {np.mean(quarterly_long_fracs):.3f}')
    print(f'completed durations: short {safe_mean(short_durations):.2f} months; '
          f'long {safe_mean(long_durations):.2f} months')
    print('completed durations exclude partnerships live at t=0 (left-censored, formed during '
          "FrancescoNetworkBackend's internal burn-in) and partnerships still active at the end "
          '(right-censored)')
    print(f'mean degree: instantaneous {np.mean(instantaneous_mean_t):.3f}; '
          f'quarterly union {np.mean(integrated_mean_t):.3f}')

    window_label = {'early': f'year {EARLY_YEAR}', 'late': f'year {LATE_YEAR}'}

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
    ax.axvline(USER_PARAMS['D_mean_short'], linestyle='--', color=TYPE_COLOR[LKEY_SHORT], alpha=0.6)
    ax.axvline(USER_PARAMS['D_mean_long'], linestyle='--', color=TYPE_COLOR[LKEY_LONG], alpha=0.6)
    ax.set_title('5. Completed partnership durations\n(HPVsim-step resolution)')
    ax.set_xlabel('duration in months')
    ax.set_ylabel('fraction of completed partnerships')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    years = np.arange(0, YEARS + 1)
    ax.axhline(target_k, linestyle='--', color='gray', label='target')
    ax.plot(years[:len(yearly_mean_t)], yearly_mean_t, 'o-', color='C2', label='realised')
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
    ax.set_title('7. Standing long-partnership fraction\n(already equilibrated at t=0 -- see module docstring)')
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
        f'Francesco network via HPVsim -- n_agents={N_AGENTS}, partners/year={target_k}, '
        f"gamma shape={USER_PARAMS['gamma_shape']}, durations="
        f"{USER_PARAMS['D_mean_short']}/{USER_PARAMS['D_mean_long']} months, "
        f'standing long fraction={target_f}',
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, dpi=120)
    print(f'\nsaved figure -> {OUT_PNG}')


if __name__ == '__main__':
    main()

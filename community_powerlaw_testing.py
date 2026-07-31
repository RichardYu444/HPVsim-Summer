#!/usr/bin/env python3
"""
community_powerlaw_testing
===========================

Diagnostics for the **age + community** bipartite network as parameterised by
basePars_community_powerlaw.py -- i.e. the power-law (Pareto) partner-propensity variant,
run through pars['network'] == 'community' / CommunityNetworkBackend.

This is community_testing.py's diagnostic suite pointed at a different parameter set: it
produces the same printed summary block and the same two figures (an 8-panel degree/duration
figure and a 2x2 age/community mixing figure). The only difference is where the parameters
come from -- basePars_community_powerlaw.base_pars, imported wholesale, rather than the
hand-written COMMUNITY_PARS block community_testing.py defines inline.

It then saves a third figure, powerlaw_log_log_testing_distributions.png: the same 8 panels
off the same data, with the distribution panels' x-axes on a log scale as well as their
y-axes. That is the view a power-law propensity actually wants -- a Pareto tail is a straight
line on log-log, and this parameterisation samples theta from a Pareto with alpha =
community_pars['gamma_shape'] (see powerlaw.py). The mixing figure gets no log-log twin: its
axes are categorical age-band and community indices, so log-scaling them would be meaningless.

Consequences of using base_pars as-is:

1. Full epidemiological configuration. base_pars carries genotypes, calibrated genotype_pars,
   beta, cross-layer mixing and a 1980-2055 horizon at 200,000 agents, so this is a much
   heavier run than community_testing.py's stripped-down 100,000-agent/50-year sim. None of
   that affects the network statistics below, but it does dominate the runtime.

2. Age mixing comes from powerlaw.AGE_MIXING, not sim['mixing']['s']. base_pars sets
   community_pars['age_mixing']/['age_band_edges'] explicitly (see that file's comment on why
   the default sim['mixing']['s'] path is not used), so figure 2's input kernel is the shared
   blended kernel, not the married/casual matrix. community_testing.py takes the other route.

3. n_communities == 1. base_pars runs a single community, so the community-mixing panels
   (figure 2, panels 3 and 4) are 1x1 and structurally degenerate -- every edge is
   within-community by construction. They are still drawn, so the figure layout matches
   community_testing.py's, but they are labelled as degenerate and carry no information.
   community_pars also has no 'community_off_diag' key for the same reason.

4. rand_seed is not pinned by base_pars, so it takes HPVsim's default of 1 (parameters.py)
   -- the same seed community_testing.py and default_network_testing.py hard-code.

Everything else -- the quarterly-resolution sampling, the _ActiveTracker debut/death
workaround, and the left/right-censoring convention for panel 5's completed durations --
follows community_testing.py exactly and carries the same caveats:

* Resolution. The network model's own internal clock is monthly; HPVsim's dt (0.25 years =
  3 months) is the finest resolution anything outside CommunityNetworkBackend can observe.
  Note this bites harder here than the panel suggests: because the backend advances 3
  network-months per HPVsim step, a partnership can form and dissolve entirely inside one
  step, and panel 5 records it at 0 months rather than the 1-2 it actually lived.

* Network-node eligibility. NetworkDelta's node bookkeeping tracks literal HPVsim
  birth/death, not sexual debut, so _ActiveTracker records sim.people.is_active each step.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

import hpvsim_working as hpv
from basePars_community_powerlaw import base_pars, community_pars

# Everything the diagnostics need about the run is read off base_pars rather than restated,
# so editing basePars_community_powerlaw.py alone keeps this script in step with it.
N_AGENTS = base_pars['n_agents']
START = base_pars['start']
END = base_pars['end']
DT = base_pars['dt']
YEARS = END - START

EARLY_YEAR = 3
LATE_YEAR = 10
OUT_PNG = 'community_powerlaw_testing_distributions.png'
OUT_MIXING_PNG = 'community_powerlaw_testing_mixing.png'
# Second copy of the same 8 panels with the distribution x-axes on a log scale as well as the
# y-axes (see apply_logx). Same run, same data, same layout -- only the x-scaling differs.
OUT_LOGLOG_PNG = 'powerlaw_log_log_testing_distributions.png'

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


class _MixingTracker(hpv.Analyzer):
    '''
    Accumulates band x band and community x community edge counts over the whole run,
    read directly off CommunityNetworkBackend's own internal state each step (age band and
    community are backend-internal bookkeeping -- see community_network.py's module
    docstring point 2 -- so, unlike degree/duration, they can't be reconstructed from
    network_history's NetworkDelta alone).

    Runs its apply() after CommunityNetworkBackend.step() has already run this timestep (see
    Sim.step()'s ordering: network_backend.step() happens before analyzers are applied), so
    sim.network_backend._state reflects this timestep's *live* (instantaneous) network -- the
    same "state after formation/dissolution, before the next step" moment the model's own
    build_snapshot() would capture.
    '''

    def __init__(self, label='mixing_tracker'):
        super().__init__(label=label)
        return

    def initialize(self, sim):
        super().initialize(sim)
        params = sim.network_backend._params
        self.n_bands = int(params['n_bands'])
        self.n_communities = int(params['n_communities'])
        self.Mage = np.zeros((self.n_bands, self.n_bands), dtype=np.int64)
        self.Mcomm = np.zeros((self.n_communities, self.n_communities), dtype=np.int64)
        self._record(sim)
        return

    def apply(self, sim):
        self._record(sim)
        return

    @staticmethod
    def _positions_of(uid_array, query):
        '''
        Position of each ``query`` uid within ``uid_array``, which is NOT guaranteed globally
        sorted (new debut arrivals are appended in the order people reach sexual debut, not in
        person-index order -- a later-born person can debut before an earlier-born one, given
        debut age is itself a distribution). A plain np.searchsorted on the raw array would
        silently return wrong/out-of-range positions once that happens, so sort once per call
        via argsort and map back through it instead.
        '''
        order = np.argsort(uid_array, kind='stable')
        sorted_uid = uid_array[order]
        pos_in_sorted = np.searchsorted(sorted_uid, query)
        return order[pos_in_sorted]

    def _record(self, sim):
        state = sim.network_backend._state
        eu, ev = state['edges_u'], state['edges_v']
        if not eu.size:
            return
        u_pos = self._positions_of(state['u_uid'], eu)
        v_pos = self._positions_of(state['v_uid'], ev)
        bu, bv = state['u_band'][u_pos], state['v_band'][v_pos]
        cu, cv = state['u_comm'][u_pos].astype(np.int64), state['v_comm'][v_pos].astype(np.int64)
        np.add.at(self.Mage, (bu, bv), 1)
        np.add.at(self.Mcomm, (cu, cv), 1)
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


def apply_logx(ax):
    '''
    Put a distribution panel's x-axis on a log scale for the log-log output set.

    symlog, not log: degree 0 is a real and common outcome here -- p_single_annual is a headline
    input, so the zero-partner column is one of the things worth reading off these panels -- and
    a plain log x-axis drops every zero silently. symlog with linthresh=1 keeps [0, 1] linear
    and goes logarithmic above it, so the zero column survives while the Pareto tail straightens
    out into the line a power law is supposed to give.
    '''
    ax.set_xscale('symlog', linthresh=1)
    ax.set_xlim(left=0)
    return


def plot_pmf(ax, samples, labels, colors, styles, markers, xlabel, title, logx=False):
    maximum = max((int(np.max(s)) for s in samples if np.asarray(s).size), default=0)
    x = np.arange(maximum + 1)
    for sample, label, color, style, marker in zip(samples, labels, colors, styles, markers):
        ax.plot(x, pmf(sample, maximum), linestyle=style, marker=marker,
                markersize=3.5, color=color, alpha=0.85, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('fraction of people')
    ax.set_yscale('log')
    if logx:
        apply_logx(ax)
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
    the run are excluded too (right-censored), matching community_testing.py's convention.
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


def band_labels(params):
    '''
    Human-readable label per age band, from age_band_edges alone. params['age_range'] is NOT
    used here: it stays the network model's own unrelated default and would mislabel the
    open-ended first/last bands.
    '''
    edges = params['age_band_edges']
    labels = [f'<{int(edges[0])}']
    labels += [f'{int(lo)}-{int(hi)}' for lo, hi in zip(edges[:-1], edges[1:])]
    labels.append(f'{int(edges[-1])}+')
    return labels


def build_sim():
    '''
    base_pars with the two extra diagnostic analyzers appended. Deep-copied first so importing
    this module never mutates basePars_community_powerlaw.base_pars for anything else in the
    process (base_pars already carries its own hpv.network_history() instance, which the copy
    duplicates rather than shares).
    '''
    pars = sc.dcp(base_pars)
    analyzers = sc.tolist(pars.get('analyzers'))
    if not any(isinstance(a, hpv.network_history) for a in analyzers):
        analyzers.append(hpv.network_history())
    analyzers += [_ActiveTracker(), _MixingTracker()]
    pars['analyzers'] = analyzers
    return hpv.Sim(pars)


def main():
    sim = build_sim()
    sim.run()

    nh = sim.get_analyzer('network_history')
    act = sim.get_analyzer('active_tracker')
    mix = sim.get_analyzer('mixing_tracker')
    layer_map = nh.layer_map
    n_uid = len(sim.people)

    be = sim.network_backend
    target_k = be._params['mean_partners_per_year']
    target_f = be._params['frac_long_target']
    p_form_long = be._params['p_form_long']
    n_bands = be._params['n_bands']
    n_comm = be._params['n_communities']
    blab = band_labels(be._params)

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

    # edges_at(t) replays every delta from t=0, so calling it three times per step is O(npts^2)
    # -- painful over base_pars' 300+ steps at 200k agents. The previous iteration's edges_now
    # IS edges_at(t-1) by construction, so carry it forward instead of recomputing it.
    prev_live = {}

    for t in range(sim.npts):
        year = year_of(sim, t)
        window = window_of(year)

        if year != running_year:
            if running_year is not None:
                finalize_year(window_of(running_year))
            running_year = year
            annual_base = prev_live
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

        prev_live = edges_now

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

    # --- Age / community mixing diagnostics (accumulated over the whole run by _MixingTracker) ---
    Mage, Mcomm = mix.Mage, mix.Mcomm
    Mage_cond = Mage / Mage.sum(axis=1, keepdims=True).clip(min=1)
    Mcomm_cond = Mcomm / Mcomm.sum(axis=1, keepdims=True).clip(min=1)
    within_comm_frac = float(np.trace(Mcomm) / Mcomm.sum()) if Mcomm.sum() else float('nan')
    random_baseline = 1.0 / n_comm

    # Availability-weighted recovery check for age mixing: row-normalise the input kernel A
    # weighted by how many V-side (male) nodes sit in each band at the end of the run, then
    # correlate against the empirical row-normalised mixing.
    final_state = be._state
    nV_band = np.bincount(final_state['v_band'], minlength=n_bands).astype(float)
    A_pred = be._params['A_age'] * nV_band[None, :]
    A_pred = A_pred / A_pred.sum(axis=1, keepdims=True).clip(min=1e-12)
    age_corr = float(np.corrcoef(Mage_cond.ravel(), A_pred.ravel())[0, 1])

    print('\n' + '=' * 68)
    print('COMMUNITY NETWORK (POWER-LAW BASE PARS), VIA HPVSIM -- SUMMARY')
    print('=' * 68)
    print(f'mean partners/year: realised {np.mean(yearly_mean_t):.3f}; target {target_k:.3f}')
    print(f'standing long fraction: realised {np.mean(standing_long_t):.3f}; '
          f'target {target_f:.3f}; formation probability {p_form_long:.3f}')
    print(f'integrated quarterly long fraction: {np.mean(quarterly_long_fracs):.3f}')
    print(f'completed durations: short {safe_mean(short_durations):.2f} months; '
          f'long {safe_mean(long_durations):.2f} months')
    print('completed durations exclude partnerships live at t=0 (left-censored, formed during '
          "CommunityNetworkBackend's internal burn-in) and partnerships still active at the end "
          '(right-censored)')
    print(f'mean degree: instantaneous {np.mean(instantaneous_mean_t):.3f}; '
          f'quarterly union {np.mean(integrated_mean_t):.3f}')
    comm_note = ' -- degenerate, base_pars runs a single community' if n_comm == 1 else ''
    print(f'within-community edge fraction: {within_comm_frac:.3f} '
          f'(uniform-random baseline over {n_comm} communities = {random_baseline:.3f}){comm_note}')
    print(f'age-mixing recovery: corr(realised, availability-weighted input A) = {age_corr:.3f}')

    window_label = {'early': f'year {EARLY_YEAR}', 'late': f'year {LATE_YEAR}'}

    # =====================================================================
    # Figure 1: degree/duration diagnostics (2x4 panel layout), rendered twice
    # =====================================================================
    def build_distribution_figure(logx):
        '''
        Build the 8-panel degree/duration figure. Called twice off the *same* computed data --
        once with linear x-axes (OUT_PNG) and once with log x-axes (OUT_LOGLOG_PNG) -- so the
        two output sets are identical numbers rendered two ways, not two runs.

        Only the six distribution panels take a log x-axis. Panels 6 and 7 plot against
        measurement year and HPVsim timestep: those are time axes, not distributions, and
        log-scaling them would say nothing about the Pareto tail, so they stay linear in both
        sets.
        '''
        fig, axes = plt.subplots(2, 4, figsize=(21, 10))

        plot_pmf(
            axes[0, 0],
            [instantaneous_pool['early'], instantaneous_pool['late']],
            [window_label['early'], window_label['late']],
            [WINDOW_COLOR['early'], WINDOW_COLOR['late']],
            ['-', '-'], ['o', 'o'],
            'current partners',
            '1. Instantaneous degree\n(HPVsim quarterly steps pooled within year)',
            logx=logx,
        )

        plot_pmf(
            axes[0, 1],
            [integrated_pool['early'], integrated_pool['late']],
            [window_label['early'], window_label['late']],
            [WINDOW_COLOR['early'], WINDOW_COLOR['late']],
            ['-', '-'], ['s', 's'],
            'partners in the quarter',
            '2. Quarterly degree\n(union over one HPVsim step)',
            logx=logx,
        )

        # Panels 3 and 5 are histograms on unit-width linear bins, unlike the pmf panels. On a
        # log x-axis those bins render progressively narrower towards the tail; the plotted
        # densities are unaffected (bin width is still 1), only the visual spacing changes.
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
        if logx:
            apply_logx(ax)
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
            logx=logx,
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
        ax.axvline(community_pars['D_mean_short'], linestyle='--', color=TYPE_COLOR[LKEY_SHORT], alpha=0.6)
        ax.axvline(community_pars['D_mean_long'], linestyle='--', color=TYPE_COLOR[LKEY_LONG], alpha=0.6)
        ax.set_title('5. Completed partnership durations\n(HPVsim-step resolution)')
        ax.set_xlabel('duration in months')
        ax.set_ylabel('fraction of completed partnerships')
        ax.set_yscale('log')
        if logx:
            apply_logx(ax)
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        # Length-driven rather than np.arange(0, YEARS+1): base_pars' yearvec runs a little past
        # `end`, so year_of() yields more measurement years than END-START.
        years = np.arange(len(yearly_mean_t))
        ax.axhline(target_k, linestyle='--', color='gray', label='target')
        ax.plot(years, yearly_mean_t, 'o-', color='C2', label='realised')
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
            logx=logx,
        )

        scale_note = ' -- LOG-LOG (log x on the distribution panels)' if logx else ''
        fig.suptitle(
            f'Community network (power-law base pars) via HPVsim -- n_agents={N_AGENTS}, '
            f'{START}-{END}, partners/year={target_k}, '
            f"Pareto alpha={community_pars['gamma_shape']}, durations="
            f"{community_pars['D_mean_short']}/{community_pars['D_mean_long']} months, "
            f'standing long fraction={target_f:.4f}{scale_note}',
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    print()
    for logx, out_path in ((False, OUT_PNG), (True, OUT_LOGLOG_PNG)):
        fig = build_distribution_figure(logx)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'saved figure -> {out_path}')

    # =====================================================================
    # Figure 2: age / community mixing diagnostics. Kept at the same 2x2 layout as
    # community_testing.py's, but with n_communities == 1 the bottom row is 1x1 and carries
    # no information -- see module docstring point 3.
    # =====================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 12))
    degenerate = '\n(degenerate: n_communities=1)' if n_comm == 1 else ''
    off_diag = community_pars.get('community_off_diag')
    off_diag_label = 'not set' if off_diag is None else f'{off_diag}'

    ax = axes2[0, 0]
    im = ax.imshow(be._params['A_age'], origin='lower', cmap='viridis', aspect='auto')
    ax.set_title('1. Input age-mixing kernel A\n(from community_pars[\'age_mixing\'] = powerlaw.AGE_MIXING)')
    ax.set_xlabel('V-side age band')
    ax.set_ylabel('U-side age band')
    ax.set_xticks(range(n_bands)); ax.set_xticklabels(blab, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_bands)); ax.set_yticklabels(blab, fontsize=7)
    fig2.colorbar(im, ax=ax, fraction=0.046)

    ax = axes2[0, 1]
    im = ax.imshow(Mage_cond, origin='lower', cmap='viridis', aspect='auto', vmin=0)
    ax.set_title(f'2. Realised age mixing (row-normalised)\nP(V band | U band); corr with input A = {age_corr:.2f}')
    ax.set_xlabel('V-side age band')
    ax.set_ylabel('U-side age band')
    ax.set_xticks(range(n_bands)); ax.set_xticklabels(blab, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n_bands)); ax.set_yticklabels(blab, fontsize=7)
    fig2.colorbar(im, ax=ax, fraction=0.046)

    ax = axes2[1, 0]
    im = ax.imshow(be._params['C_comm'], origin='lower', cmap='magma', aspect='auto')
    ax.set_title(f'3. Input community-mixing kernel C\n(n_communities={n_comm}, off_diag={off_diag_label}){degenerate}')
    ax.set_xlabel('V-side community')
    ax.set_ylabel('U-side community')
    ax.set_xticks(range(n_comm)); ax.set_yticks(range(n_comm))
    fig2.colorbar(im, ax=ax, fraction=0.046)

    ax = axes2[1, 1]
    im = ax.imshow(Mcomm_cond, origin='lower', cmap='magma', aspect='auto', vmin=0, vmax=1)
    ax.set_title(f'4. Realised community mixing (row-normalised)\n'
                 f'within-community edge fraction = {within_comm_frac:.2f}{degenerate}')
    ax.set_xlabel('V-side community')
    ax.set_ylabel('U-side community')
    ax.set_xticks(range(n_comm)); ax.set_yticks(range(n_comm))
    for i in range(n_comm):
        for j in range(n_comm):
            ax.text(j, i, f'{Mcomm_cond[i, j]:.2f}', ha='center', va='center',
                    color='w' if Mcomm_cond[i, j] < 0.5 else 'k', fontsize=8)
    fig2.colorbar(im, ax=ax, fraction=0.046)

    fig2.suptitle(
        f'Community network (power-law base pars) via HPVsim -- age & community mixing '
        f'diagnostics (n_agents={N_AGENTS}, {n_comm} communities, {n_bands} age bands)',
        fontsize=12,
    )
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.savefig(OUT_MIXING_PNG, dpi=120)
    print(f'saved figure -> {OUT_MIXING_PNG}')


if __name__ == '__main__':
    main()

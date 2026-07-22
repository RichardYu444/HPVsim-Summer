'''
FrancescoNetworkBackend: wires bipartite_network_model.py (a temporal bipartite contact-network
model with Gamma-distributed partner propensities and two partnership-duration classes, short and
long) into HPVsim as pars['network'] == 'francesco'.

Architecture
------------
Unlike DefaultNetworkBackend (which drives HPVsim's own People.create_partnerships /
dissolve_partnerships), this backend owns partnership formation AND dissolution itself: each
HPVsim timestep it sub-steps the bipartite model's own monthly network_step() (its formation and
dissolution hazards, q_short/q_long, are calibrated per-month) and diffs the resulting edge set
directly into people.contacts['s']/['l'].

Node demography is deliberately NOT owned by the bipartite model. Its own natural-death rate
(delta, derived from 'tau') is forced to 0 immediately after parameter conversion, so it never
invents its own deaths or replacement births. Instead:
  - Real deaths come from people.network_removed_node_log (populated by
    People.update_states_pre -> remove_people, plus any earlier HIV-mortality step this
    timestep) and are passed in as extra_deaths_U/V.
  - Real network arrivals are people crossing sexual debut (people.is_active going False->True),
    NOT literal HPVsim births -- the bipartite model has no age concept, and pre-debut people
    aren't network-relevant. Newly-debuted people are injected into the bipartite state directly
    (with freshly-sampled Gamma propensities), since the model has no public API for externally
    chosen UIDs.

U/V node identity is chosen to equal HPVsim person indices directly (U = female persons, V = male
persons; a person's sex is fixed for life, so this partition is stable) -- this sidesteps needing
any UID-remapping table, at the cost of one relabeling pass right after init_network_state()
(whose dense 0..nU-1/0..nV-1 UIDs must be overwritten, position-for-position, onto the real
person indices that own each sampled propensity).

dur/end convention
-------------------
HPVsim's Layer requires a dur/start/end per edge, but this backend -- not date-based expiry -- is
what actually ends a partnership (via the bipartite model's own monthly Bernoulli hazard). dur/end
are populated with the *expected* duration for the edge's class (1/q_short or 1/q_long, converted
to years) purely as a plausible value for anyone inspecting people.contacts directly; they are not
used anywhere to decide when an edge actually dissolves.
'''

import numpy as np
import sciris as sc

from . import network as hpnet
from . import bipartite_network_model as bnm
from . import utils as hpu
from . import defaults as hpd
from .bipartite_network_model import _sample_side_theta
from .population import age_scale_acts

__all__ = ['FrancescoNetworkBackend']

# Canonical (f, m) pairing key for vectorised edge-set membership tests. Must exceed the largest
# possible person index by a wide margin (2**40 is far beyond any realistic n_agents).
_KEY = np.int64(1) << 40


def _pair_keys(f, m):
    return np.asarray(f, dtype=np.int64) * _KEY + np.asarray(m, dtype=np.int64)


class FrancescoNetworkBackend(hpnet.NetworkBackend):
    '''
    NetworkBackend for the bipartite short/long-partnership network (see module docstring).
    Requires exactly two contact layers, 's' (short) and 'l' (long) -- see
    parameters.reset_layer_pars()'s layer_defaults['francesco'].
    '''

    LKEY_SHORT = 's'
    LKEY_LONG = 'l'

    def initialize(self, sim):
        people = sim.people

        self.layer_map = list(people.layer_keys())
        for req in (self.LKEY_SHORT, self.LKEY_LONG):
            if req not in self.layer_map:
                errormsg = (f"FrancescoNetworkBackend requires layer keys '{self.LKEY_SHORT}' and "
                            f"'{self.LKEY_LONG}'; got {self.layer_map}. Check "
                            f"parameters.reset_layer_pars()'s layer_defaults['francesco'].")
                raise ValueError(errormsg)
        self._layer_code = {lkey: np.int8(i) for i, lkey in enumerate(self.layer_map)}
        self._lno = {lkey: i for i, lkey in enumerate(self.layer_map)}

        self._months_per_step = self._validate_months_per_step(sim['dt'])

        # Reset per-step bookkeeping so initialization is safe to call more than once
        people.network_added_node_log = []
        people.network_removed_node_log = []
        people.network_added_edges = {}
        people.network_dissolved_edges = {}

        # Only currently-debuted people are network-relevant at t=0
        is_active0 = people.is_active.copy()
        u_real = np.where(is_active0 & people.is_female)[0].astype(np.int64)
        v_real = np.where(is_active0 & people.is_male)[0].astype(np.int64)
        if u_real.size == 0 or v_real.size == 0:
            errormsg = ('FrancescoNetworkBackend needs at least one sexually active person of '
                        'each sex at t=0 to initialize the bipartite network.')
            raise ValueError(errormsg)

        user_params = sc.dcp(sim['francesco_pars'])
        calibrate_kwargs = user_params.pop('calibrate_kwargs', {})

        params = bnm.interpretable_to_params(user_params, int(u_real.size), int(v_real.size),
                                              verbose=(sim['verbose'] > 0))
        # HPVsim's own demography (people.update_states_pre / remove_people) drives node turnover
        # for this network -- disable the bundle's own natural-death/replacement-birth process so
        # the two don't run in parallel and silently diverge from the sim's real population.
        params['delta'] = 0.0

        model = bnm.build_model(int(u_real.size), int(v_real.size))
        rng = np.random.default_rng(int(sim['rand_seed']))
        params = bnm.calibrate(model, params, verbose=(sim['verbose'] > 0), **calibrate_kwargs)
        state = bnm.init_network_state(model, params, rng)

        # init_network_state()'s UIDs are dense 0..nU-1 / 0..nV-1; relabel them onto the real
        # HPVsim person indices that will own these nodes. Position i keeps whichever theta/entry
        # it was sampled with -- only the *label* at that position changes.
        state['u_uid'] = u_real.copy()
        state['v_uid'] = v_real.copy()
        if state['edges_u'].size:
            state['edges_u'] = u_real[state['edges_u']]
        if state['edges_v'].size:
            state['edges_v'] = v_real[state['edges_v']]
        state['next_uid_U'] = int(len(people)) + 1
        state['next_uid_V'] = int(len(people)) + 1

        self._model, self._params, self._state, self._rng = model, params, state, rng
        self._month = 0

        # init_network_state() colors each edge as short/long independently at formation time
        # (via p_form_long); the target *standing* long-edge fraction (frac_long) is only reached
        # after the short/long populations equilibrate under their different dissolution rates
        # (q_short vs q_long) -- so, as bipartite_network_model.simulate() itself always does,
        # burn in before taking the snapshot used to seed HPVsim's contacts.
        burn_months = bnm.default_burn_months(params)
        for _ in range(burn_months):
            self._month += 1
            bnm.network_step(state, model, params, rng, self._month)

        snap0 = bnm.build_snapshot(state, model)
        self._apply_added_contacts(sim, snap0['edges_u'], snap0['edges_v'], snap0['edges_type'])

        self._active_prev = is_active0

        # Capture the initial network as a delta of everything "added" at t=0 (mirrors
        # DefaultNetworkBackend.initialize())
        alive_inds = hpu.true(people.alive)
        added_nodes = sc.objdict(
            uid=alive_inds.astype(hpd.default_int),
            sex=people.sex[alive_inds].astype(np.int8),
            age=people.age[alive_inds].astype(hpd.default_float),
            cluster=people.cluster[alive_inds].astype(hpd.default_int),
            entry_kind=np.full(len(alive_inds), hpnet.ENTRY_INITIAL, dtype=np.int8),
        )
        added_edges = hpnet.stack_edges(people.contacts, self._layer_code)
        self.initial_snapshot = hpnet.NetworkDelta(t=sim.t, added_nodes=added_nodes, added_edges=added_edges)

        self.delta = None
        self.initialized = True
        return

    @staticmethod
    def _validate_months_per_step(dt):
        months = dt * 12.0
        rounded = round(months)
        if rounded < 1 or abs(months - rounded) > 1e-6:
            errormsg = (f"FrancescoNetworkBackend requires sim['dt'] to correspond to a whole "
                        f"number of months (the bipartite model's own timestep); got dt={dt} "
                        f"years = {months} months. Use e.g. dt=1/12, 1/6, 0.25, or 1.0.")
            raise ValueError(errormsg)
        return int(rounded)

    def step(self, sim):
        people = sim.people
        t = sim.t

        people.update_states_pre(t=t, year=sim.yearvec[t])  # Real ages/deaths/births/migration

        removed_log = people.network_removed_node_log
        if len(removed_log):
            dead_inds = np.concatenate([inds for inds, _ in removed_log])
        else:
            dead_inds = np.empty(0, dtype=hpd.default_int)
        dead_sex = people.sex[dead_inds] if dead_inds.size else np.empty(0, dtype=np.int8)
        dead_female = dead_inds[dead_sex == 0].astype(np.int64)
        dead_male = dead_inds[dead_sex == 1].astype(np.int64)

        # Debut arrivals: people who just became sexually active join the bipartite network as
        # brand-new nodes (not triggered by literal HPVsim births -- see module docstring).
        is_active_now = people.is_active
        prev_len, cur_len = self._active_prev.size, is_active_now.size
        if cur_len > prev_len:
            active_prev_padded = np.concatenate([self._active_prev, np.zeros(cur_len - prev_len, dtype=bool)])
        else:
            active_prev_padded = self._active_prev
        new_idx = np.where(is_active_now & ~active_prev_padded)[0]
        new_female = new_idx[people.sex[new_idx] == 0].astype(np.int64)
        new_male = new_idx[people.sex[new_idx] == 1].astype(np.int64)
        self._inject_arrivals(new_female, new_male)

        # This HPVsim step covers self._months_per_step bundle-months (e.g. 3, for dt=0.25). The
        # bundle's own monthly hazards mean a short partnership (mean duration ~a couple of
        # months) can form AND fully dissolve inside a single HPVsim step -- diffing only the
        # state before vs. after the whole step would miss such partnerships entirely, since
        # they're absent from both endpoints. So instead: an edge is ADDED to people.contacts the
        # first time it's seen at ANY monthly sub-boundary this step (giving it at least one
        # chance at this step's transmission calculation), and edges are only ever REMOVED once,
        # at the very end, if they're not part of the bundle's final live state -- matching the
        # bundle's own documented convention that its quarterly output is "the union of all nodes
        # and edges present at monthly boundaries in that interval". An edge that flickers off and
        # back on within the same step is therefore never popped/re-added, avoiding pointless eid
        # churn, and a same-step dissolve-and-reform under a DIFFERENT duration class is still
        # handled correctly since keys encode (u, v, type).
        prev_snap = bnm.build_snapshot(self._state, self._model)
        seen_keys = self._triple_keys(prev_snap)  # Everything already live/in people.contacts

        added_u_parts, added_v_parts, added_t_parts = [], [], []
        month_snap = prev_snap
        for i in range(self._months_per_step):
            self._month += 1
            bnm.network_step(
                self._state, self._model, self._params, self._rng, self._month,
                extra_deaths_U=(dead_female if i == 0 else None),
                extra_deaths_V=(dead_male if i == 0 else None),
                track_durations=False,
            )
            month_snap = bnm.build_snapshot(self._state, self._model)
            month_keys = self._triple_keys(month_snap)
            new_mask = ~np.isin(month_keys, seen_keys)
            if new_mask.any():
                added_u_parts.append(month_snap['edges_u'][new_mask])
                added_v_parts.append(month_snap['edges_v'][new_mask])
                added_t_parts.append(month_snap['edges_type'][new_mask])
                seen_keys = np.concatenate([seen_keys, month_keys[new_mask]])

        final_keys = self._triple_keys(month_snap)  # Live state as of the last sub-month
        removed_keys = seen_keys[~np.isin(seen_keys, final_keys)]
        removed_u, removed_v, removed_t = self._decode_triple_keys(removed_keys)

        added_u = np.concatenate(added_u_parts) if added_u_parts else np.empty(0, dtype=np.int64)
        added_v = np.concatenate(added_v_parts) if added_v_parts else np.empty(0, dtype=np.int64)
        added_t = np.concatenate(added_t_parts) if added_t_parts else np.empty(0, dtype=np.int8)

        people.network_added_edges = self._apply_added_contacts(sim, added_u, added_v, added_t)
        people.network_dissolved_edges = self._apply_removed_contacts(
            sim, removed_u, removed_v, removed_t, dead_inds)

        self._active_prev = is_active_now.copy()
        return

    @staticmethod
    def _triple_keys(snap):
        return _pair_keys(snap['edges_u'], snap['edges_v']) * 2 + snap['edges_type'].astype(np.int64)

    @staticmethod
    def _decode_triple_keys(keys):
        ''' Invert _triple_keys/_pair_keys: recover (u, v, type) arrays from combined keys '''
        keys = np.asarray(keys, dtype=np.int64)
        ty = (keys % 2).astype(np.int8)
        pair = keys // 2
        v = pair % _KEY
        u = pair // _KEY
        return u, v, ty

    def _inject_arrivals(self, new_female, new_male):
        state, params, rng = self._state, self._params, self._rng
        if new_female.size:
            theta = _sample_side_theta(new_female.size, params['gamma_shape_U'], rng)
            state['u_uid'] = np.concatenate([state['u_uid'], new_female])
            state['u_theta'] = np.concatenate([state['u_theta'], theta])
            state['u_entry'] = np.concatenate(
                [state['u_entry'], np.full(new_female.size, bnm.ENTRY_BIRTH, dtype=np.int8)])
        if new_male.size:
            theta = _sample_side_theta(new_male.size, params['gamma_shape_V'], rng)
            state['v_uid'] = np.concatenate([state['v_uid'], new_male])
            state['v_theta'] = np.concatenate([state['v_theta'], theta])
            state['v_entry'] = np.concatenate(
                [state['v_entry'], np.full(new_male.size, bnm.ENTRY_BIRTH, dtype=np.int8)])
        return

    def _build_contacts_dict(self, sim, lkey, f_idx, m_idx, q):
        people = sim.people
        n = f_idx.size
        if n == 0:
            return {}
        acts = hpu.sample(**sim['acts'][lkey], size=n)
        scaled_acts = age_scale_acts(
            acts=acts, age_act_pars=sim['age_act_pars'][lkey],
            age_f=people.age[f_idx], age_m=people.age[m_idx],
            debut_f=people.debut[f_idx], debut_m=people.debut[m_idx],
        )
        start = np.full(n, sim.t, dtype=hpd.default_float)
        # Placeholder only -- see module docstring's "dur/end convention"
        dur = np.full(n, (1.0 / q) / 12.0, dtype=hpd.default_float)
        end = start + dur
        return dict(
            f=f_idx.astype(hpd.default_int), m=m_idx.astype(hpd.default_int),
            age_f=people.age[f_idx], age_m=people.age[m_idx],
            dur=dur, acts=scaled_acts, start=start, end=end,
            cluster_f=people.cluster[f_idx].astype(hpd.default_int),
            cluster_m=people.cluster[m_idx].astype(hpd.default_int),
        )

    def _apply_added_contacts(self, sim, f_all, m_all, type_all):
        '''
        Insert new edges into people.contacts via people.add_contacts() (the single choke point
        that assigns persistent eids) and update the same current_partners/rship_*/ever_partnered
        bookkeeping People.create_partnerships normally maintains. Returns the eid-enriched
        per-layer dict, directly usable as people.network_added_edges.
        '''
        people = sim.people
        params = self._params
        new_contacts = {}
        for lkey, ty, q in ((self.LKEY_SHORT, bnm.EDGE_SHORT, params['q_short']),
                            (self.LKEY_LONG, bnm.EDGE_LONG, params['q_long'])):
            sel = (type_all == ty)
            new_contacts[lkey] = self._build_contacts_dict(sim, lkey, f_all[sel], m_all[sel], q)

        added = people.add_contacts(new_contacts)

        for lkey in (self.LKEY_SHORT, self.LKEY_LONG):
            layer_data = added.get(lkey) or {}
            f_idx = np.asarray(layer_data.get('f', []), dtype=hpd.default_int)
            m_idx = np.asarray(layer_data.get('m', []), dtype=hpd.default_int)
            if f_idx.size == 0:
                continue
            lno = self._lno[lkey]
            both = np.concatenate([f_idx, m_idx])
            people.ever_partnered[both] = True
            unique, counts = hpu.unique(both)
            people.current_partners[lno, unique] += counts
            people.rship_start_dates[lno, both] = sim.t
            people.n_rships[lno, unique] += counts
            lags = people.rship_start_dates[lno, unique] - people.rship_end_dates[lno, unique]
            people.rship_lags[lkey] += np.histogram(lags, people.lag_bins)[0]
        return added

    def _apply_removed_contacts(self, sim, f_all, m_all, type_all, dead_inds):
        '''
        Directly pop dissolved edges out of people.contacts (bypassing People.dissolve_partnerships,
        which this backend replaces) and update current_partners/rship_end_dates to match. Returns
        a per-layer dict directly usable as people.network_dissolved_edges.
        '''
        people = sim.people
        out = {}
        for lkey, ty in ((self.LKEY_SHORT, bnm.EDGE_SHORT), (self.LKEY_LONG, bnm.EDGE_LONG)):
            sel = (type_all == ty)
            f_sel, m_sel = f_all[sel], m_all[sel]
            if f_sel.size == 0:
                continue
            layer = people.contacts[lkey]
            removed_keys = _pair_keys(f_sel, m_sel)
            layer_keys = _pair_keys(layer['f'], layer['m'])
            to_dissolve = np.isin(layer_keys, removed_keys)
            if not to_dissolve.any():
                continue
            dissolved = layer.pop_inds(to_dissolve)
            unique, counts = hpu.unique(np.concatenate([dissolved['f'], dissolved['m']]))
            lno = self._lno[lkey]
            people.current_partners[lno, unique] -= counts
            people.rship_end_dates[lno, unique] = sim.t
            node_removed = np.isin(dissolved['f'], dead_inds) | np.isin(dissolved['m'], dead_inds)
            reason = np.where(node_removed, hpnet.EDGE_NODE_REMOVED, hpnet.EDGE_DISSOLVED).astype(np.int8)
            out[lkey] = dict(eid=dissolved['eid'], f=dissolved['f'], m=dissolved['m'], reason=reason)
        return out

    def finalize_step(self, sim):
        people = sim.people
        t = sim.t

        added_log = people.network_added_node_log
        if len(added_log):
            add_inds = np.concatenate([inds for inds, _ in added_log])
            add_kinds = np.concatenate([np.full(len(inds), kind, dtype=np.int8) for inds, kind in added_log])
        else:
            add_inds = np.empty(0, dtype=hpd.default_int)
            add_kinds = np.empty(0, dtype=np.int8)
        added_nodes = sc.objdict(
            uid=add_inds.astype(hpd.default_int),
            sex=people.sex[add_inds].astype(np.int8),
            age=people.age[add_inds].astype(hpd.default_float),
            cluster=people.cluster[add_inds].astype(hpd.default_int),
            entry_kind=add_kinds,
        )

        removed_log = people.network_removed_node_log
        if len(removed_log):
            rem_inds = np.concatenate([inds for inds, _ in removed_log])
            rem_reasons = np.concatenate([
                np.full(len(inds), hpnet._CAUSE_TO_REASON[cause], dtype=np.int8) for inds, cause in removed_log
            ])
        else:
            rem_inds = np.empty(0, dtype=hpd.default_int)
            rem_reasons = np.empty(0, dtype=np.int8)
        removed_nodes = sc.objdict(uid=rem_inds.astype(hpd.default_int), reason=rem_reasons)

        added_edges = hpnet.stack_edges(people.network_added_edges, self._layer_code)
        removed_edges = hpnet.stack_edges(people.network_dissolved_edges, self._layer_code, include_reason=True)

        delta = hpnet.NetworkDelta(t=t, added_nodes=added_nodes, removed_nodes=removed_nodes,
                                    added_edges=added_edges, removed_edges=removed_edges)
        self.delta = delta
        return delta

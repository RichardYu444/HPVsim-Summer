'''
CommunityNetworkBackend: wires age_community_bipartite_network_model.py (a temporal bipartite
contact-network model with Gamma-distributed partner propensities, age-band mixing, community
mixing, and two partnership-duration classes, short and long) into HPVsim as
pars['network'] == 'community'.

Architecture
------------
This backend owns partnership formation AND dissolution itself, sub-stepping the network
model's own monthly network_step() within each HPVsim timestep and diffing the resulting edge
set directly into people.contacts['s']/['l']. Partnership durations use a dur/end convention
where dissolution is driven by the network model's own monthly hazard rather than HPVsim
reaching the dur/end date on a contact; each HPVsim step covers self._months_per_step network
months, and the edge set added since the previous snapshot is computed via a union over each
monthly sub-step boundary. Node identity (U/V) in the network model is kept equal to the
arriving person's real HPVsim uid throughout.

Design notes
------------
1. Age mixing. age_community_bipartite_network_model has no natural-demography rate to disable
   (its population changes ONLY via externally supplied deaths/births -- there is no 'delta'/'tau'
   to force to zero). But it DOES need each node's age kept in sync with reality, since it decides
   *who partners whom* based on a static per-node age band -- ageing is deliberately NOT simulated
   inside the network model itself. Every HPVsim step, this backend re-derives every live node's
   age (and hence age band/block) directly from people.age[uid] (see _refresh_bands()). The
   age-mixing kernel A[band_man, band_woman] itself is NOT invented here or defaulted by the
   network model -- it is built once, at initialize(), from HPVsim's own existing age-mixing
   parameter, sim['mixing'] (see parameters.get_mixing()'s 'community' branch). The network model
   only has a single (layer-agnostic) age-mixing kernel, so only sim['mixing'][LKEY_SHORT] ('s')
   is actually read for this purpose; 's' and 'l' default to the same matrix, so override 's' if
   you want to change the network's age assortativity.

2. Community assignment. Every node additionally carries a community tag drawn once, at the point
   it first enters the network (t=0 for the initial population, or at sexual debut for later
   arrivals), from pars['community_pars']['community_probs'] (uniform over n_communities by
   default). Communities are for life -- there is no migration between communities modelled here.
   Community/age/theta state lives entirely inside this backend's self._state; it is not mirrored
   onto People.

3. No births_U/V via the network model's own entrant API. age_community_bipartite_network_model
   supports a births_U/V dict path in network_step() that auto-assigns dense sequential UIDs --
   but node identity here must equal the arriving person's real HPVsim index, which won't in
   general be dense/sequential. So new debuts are injected directly into state via
   _inject_arrivals(); only extra_deaths_U/V (which key off UID membership, not assignment order)
   are passed into network_step().

4. Annual singleness control (opt-in, community_pars['p_single_annual'], default 0.0 = off).
   Singleness here is a controlled INPUT, not an emergent outcome, because the underlying
   propensity-driven sampler structurally cannot produce a realistic level of it: partner
   counts come out Poisson-with-mean-proportional-to-theta, so P(0 partners) can never fall
   below exp(-mean degree), and heterogeneity only pushes it higher (see
   _force_pair_ungated()'s docstring for the full argument). Two cooperating pieces, both run
   every 12 network-months, gate first:

   a) _refresh_annual_gate() redraws an independent Bernoulli(p_single_annual) mask and zeroes
      the gated individuals' theta for the year, rescaling the survivors by 1/(1-p) so total
      network connectivity is unchanged. This is FORMATION-ONLY: zero theta removes someone
      from _sample_edges' weighted endpoint draw, but dissolution depends only on the fixed
      q_short/q_long hazards, never theta, so a gated person's EXISTING partnerships are left
      untouched (a deliberate choice -- see the project discussion of 2026-07-30).
   b) _force_pair_ungated() then pairs up every UNGATED person who currently holds no
      partnership, respecting the age/community mixing kernel, so the ungated majority is
      guaranteed at least one partner. This mirrors how HPVsim's own 'default' network
      guarantees layer participants a partner via its 'poisson1' partner-count distributions.

   self._theta_true_u/_v (uid -> true theta dicts, separate from state['u_theta']/['v_theta']
   itself) are the source of truth the gate rebuilds from each time; see
   _refresh_annual_gate()'s own docstring for why a dict (not a parallel array) is used.

5. Mortality-aware calibration. The network model's own calibrate() solves for equilibrium with
   NO demography, but this backend feeds real HPVsim deaths in via extra_deaths_U/V, and every
   edge incident to a dead node is destroyed. For long partnerships that second dissolution
   channel is as large as their own q_long hazard, so an uncorrected calibration overshoots
   long-tie duration badly (measured: realised ~167 months against a 583-month target, while
   short ties matched exactly). initialize() therefore estimates a per-edge monthly death
   hazard (see _estimate_edge_mortality_hazard(), or pin it with
   community_pars['mortality_hazard']) and calibrates against q_short/q_long inflated by it --
   NOT params['q'], which drives formation and would cancel the correction out. The sim itself
   then runs with the true, uninflated q values.
'''

import numpy as np
import sciris as sc

from . import network as hpnet
from . import age_community_bipartite_network_model as acbnm
from . import utils as hpu
from . import defaults as hpd
from .age_community_bipartite_network_model import _sample_side_theta, _age_to_band, _block_of
from .population import age_scale_acts

__all__ = ['CommunityNetworkBackend']

# Canonical (f, m) pairing key for vectorised edge-set membership tests. Must exceed the largest
# possible person index by a wide margin (2**40 is far beyond any realistic n_agents).
_KEY = np.int64(1) << 40


def _pair_keys(f, m):
    return np.asarray(f, dtype=np.int64) * _KEY + np.asarray(m, dtype=np.int64)


def _age_mixing_from_hpvsim_matrix(mat):
    '''
    Convert one of HPVsim's own age-mixing matrices (parameters.get_mixing()'s format: first
    column = age-bin lower edges, remaining columns = the row/column-square mixing weights,
    males in rows / females in columns) into age_community_bipartite_network_model's
    (band_edges, A) representation, where band_edges are upper-exclusive band boundaries and
    A[band_man, band_woman] is square with shape (len(band_edges)+1, len(band_edges)+1).
    '''
    mat = np.asarray(mat, dtype=float)
    n = mat.shape[0]
    if mat.ndim != 2 or mat.shape != (n, n + 1):
        errormsg = (f"CommunityNetworkBackend expected an HPVsim-style age-mixing matrix "
                    f"(n x (n+1), first column = age-bin edges), got shape {mat.shape}.")
        raise ValueError(errormsg)
    age_bins = mat[:, 0]
    A = mat[:, 1:]
    band_edges = age_bins[1:]  # Drop the leading (always-zero) bin edge -- upper-exclusive edges
    return band_edges, A


class CommunityNetworkBackend(hpnet.NetworkBackend):
    '''
    NetworkBackend for the age+community bipartite short/long-partnership network (see module
    docstring). Requires exactly two contact layers, 's' (short) and 'l' (long) -- see
    parameters.reset_layer_pars()'s layer_defaults['community'].
    '''

    LKEY_SHORT = 's'
    LKEY_LONG = 'l'

    def initialize(self, sim):
        people = sim.people

        self.layer_map = list(people.layer_keys())
        for req in (self.LKEY_SHORT, self.LKEY_LONG):
            if req not in self.layer_map:
                errormsg = (f"CommunityNetworkBackend requires layer keys '{self.LKEY_SHORT}' and "
                            f"'{self.LKEY_LONG}'; got {self.layer_map}. Check "
                            f"parameters.reset_layer_pars()'s layer_defaults['community'].")
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
            errormsg = ('CommunityNetworkBackend needs at least one sexually active person of '
                        'each sex at t=0 to initialize the network.')
            raise ValueError(errormsg)

        user_params = sc.dcp(sim['community_pars'])
        calibrate_kwargs = user_params.pop('calibrate_kwargs', {})

        # Opt-in annual singleness gate (module docstring point 4) -- popped here (not left
        # for interpretable_to_params to read) since it's HPVsim-specific backend bookkeeping,
        # not one of the vendored model's own recognised parameters.
        p_single_annual = float(user_params.pop('p_single_annual', 0.0))
        if not (0.0 <= p_single_annual <= 1.0):
            errormsg = f"community_pars['p_single_annual'] must be in [0, 1], got {p_single_annual}"
            raise ValueError(errormsg)
        self._p_single_annual = p_single_annual

        # Mortality correction for calibrate() (module docstring point 5). None => auto-estimate
        # from HPVsim's own death rates; a float pins it; 0.0 disables the correction entirely.
        mortality_hazard = user_params.pop('mortality_hazard', None)

        # Age mixing is sourced from HPVsim's own existing age-mixing parameter, sim['mixing']
        # (see module docstring point 1), NOT invented/defaulted by the network model -- unless
        # the caller has already put an explicit age_mixing/age_band_edges pair in
        # community_pars, in which case that override wins.
        if 'age_mixing' not in user_params or 'age_band_edges' not in user_params:
            band_edges, A = _age_mixing_from_hpvsim_matrix(sim['mixing'][self.LKEY_SHORT])
            user_params.setdefault('age_band_edges', band_edges)
            user_params.setdefault('age_mixing', A)

        params = acbnm.interpretable_to_params(user_params, int(u_real.size), int(v_real.size))

        model = acbnm.build_model(int(u_real.size), int(v_real.size))
        rng = np.random.default_rng(int(sim['rand_seed']))

        # Mortality-aware calibration (module docstring point 5). acbnm.calibrate() solves the
        # equilibrium edges ~ formation/q with ZERO demography, but the real sim has a second,
        # comparable dissolution channel: _apply_external_turnover() deletes every edge incident
        # to a dead node. For long ties that channel is as large as q_long itself (q_long can be
        # ~0.0017/month against a per-edge death hazard of similar order), which is why realised
        # long-tie durations came out ~3.5x shorter than D_mean_long while short ties (q_short
        # ~0.14/month, which swamps mortality) matched their target exactly.
        #
        # Fix: calibrate against the EFFECTIVE dissolution hazard by inflating q_short/q_long
        # only. params['q'] is deliberately NOT inflated -- _sample_edges' formation scale is
        # params['q'] * eff_rho, so inflating it too would raise formation and dissolution
        # together and leave the equilibrium edge count unchanged (a no-op). With only the
        # dissolution rates raised, calibrate()'s empirical loop finds the larger rho that the
        # real, mortality-exposed network needs. The sim itself then runs with the ORIGINAL q
        # values, since its true dissolution is q + actual deaths.
        mu_edge = (self._estimate_edge_mortality_hazard(sim, params, u_real, v_real)
                   if mortality_hazard is None else float(mortality_hazard))
        cal_params = dict(params)
        if mu_edge > 0:
            cal_params['q_short'] = min(params['q_short'] + mu_edge, 0.999)
            cal_params['q_long'] = min(params['q_long'] + mu_edge, 0.999)
        calibrated = acbnm.calibrate(model, cal_params, verbose=(sim['verbose'] > 0), **calibrate_kwargs)
        # Take only the calibrated knobs across; keep the true (uninflated) q values.
        params = dict(params)
        for key in ('rho', 'p_form_long', 'rho_correction_factor', 'calibrated'):
            if key in calibrated:
                params[key] = calibrated[key]
        params['mortality_hazard_used'] = mu_edge
        if sim['verbose'] > 0:
            print(f"  mortality-aware calibration: per-edge death hazard mu={mu_edge:.6f}/month "
                  f"(vs q_short={params['q_short']:.4f}, q_long={params['q_long']:.4f})")

        # Community assignment happens here, at population initialization (module docstring
        # point 2), using the model's own (validated/normalised) community_probs -- real ages are
        # supplied directly so init_network_state() does NOT sample synthetic ones.
        n_comm = int(params['n_communities'])
        community_probs = params['community_probs']
        params['init_ages_U'] = people.age[u_real].astype(float)
        params['init_comm_U'] = rng.choice(n_comm, size=u_real.size, p=community_probs).astype(np.int16)
        params['init_ages_V'] = people.age[v_real].astype(float)
        params['init_comm_V'] = rng.choice(n_comm, size=v_real.size, p=community_probs).astype(np.int16)

        state = acbnm.init_network_state(model, params, rng)

        # init_network_state()'s UIDs are dense 0..nU-1 / 0..nV-1; relabel them onto the real
        # HPVsim person indices that will own these nodes. Position i keeps whichever
        # theta/age/community it was sampled with -- only the *label* at that position changes.
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

        #NEW!! Mirror the community tags this backend just drew onto People, so results (and any
        # analyzer) can stratify by community without reaching into self._state. Done here rather
        # than beside the rng.choice() calls above so it is robust to any reordering inside
        # init_network_state(); consumes no rng, so it cannot perturb the simulation.
        people.community[state['u_uid']] = state['u_comm']
        people.community[state['v_uid']] = state['v_comm']

        # True (ungated) theta per uid -- the source of truth _refresh_annual_gate() rebuilds
        # state['u_theta']/['v_theta'] from every 12 months (module docstring point 4). Must be
        # built AFTER the uid-relabeling above (state['u_uid']/['v_uid'] are real HPVsim uids by
        # this point, not init_network_state()'s dense 0..N-1 labels). Built unconditionally
        # (cheap, no rng draws) even with the gate off, so turning it on/off elsewhere doesn't
        # change rng consumption here.
        self._theta_true_u = dict(zip(state['u_uid'].tolist(), state['u_theta'].tolist()))
        self._theta_true_v = dict(zip(state['v_uid'].tolist(), state['v_theta'].tolist()))
        # (free women, free men) left unpaired by the most recent _force_pair_ungated() call
        self._force_pair_residual = (0, 0)
        # (women, men) by which the most recent gate fell short of round(p*N) genuinely-single
        # people -- non-zero means the population had too few unpartnered people to hit the
        # p_single_annual target without dissolving partnerships (see _choose_gated())
        self._gate_shortfall = (0, 0)

        # Burn in (no external turnover) before taking the snapshot used to seed HPVsim's
        # contacts, so short/long populations have equilibrated. Deliberately 1x
        # D_mean_long here, NOT the vendored model's own acbnm._default_burn_months()
        # (~5x D_mean_long, floored at 120 months) -- that 5x figure is sized for
        # age_community_bipartite_network_model.calibrate()'s own small cal-sized burn-in
        # (n_cal, default 2500), not for this backend's burn-in at HPVsim's full
        # population size, where 5x made initialize() prohibitively slow (e.g. 4800
        # months at D_mean_long=960). This is an HPVsim-specific override, hence living
        # here rather than in the vendored model file (see that module's own "vendored,
        # unmodified" docstring).
        burn_months = int(max(params['D_mean_long'], 120))
        for _ in range(burn_months):
            self._month += 1
            if self._p_single_annual > 0 and self._month % 12 == 0:
                self._refresh_annual_gate()
                self._force_pair_ungated()  # Must follow the gate: theta>0 identifies "ungated"
            acbnm.network_step(state, model, params, rng, self._month)

        snap0 = acbnm.build_snapshot(state, model)
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
            errormsg = (f"CommunityNetworkBackend requires sim['dt'] to correspond to a whole "
                        f"number of months (the network model's own timestep); got dt={dt} "
                        f"years = {months} months. Use e.g. dt=1/12, 1/6, 0.25, or 1.0.")
            raise ValueError(errormsg)
        return int(rounded)

    def _refresh_annual_gate(self):
        '''
        Every 12 network-months (module docstring point 4), redraw which round(p_single_annual
        * N) people are designated single for the coming year (see _choose_gated() for how
        that set is picked, and why it is drawn from the currently-unpartnered rather than
        uniformly at random) and rebuild state['u_theta']/['v_theta'] from
        self._theta_true_u/_v with those individuals' theta zeroed. Indexed by uid, not
        cached position, since array order shifts on death/debut between calls. Formation-only: zero theta removes a gated person from
        _sample_edges' weighted endpoint draw (zero selection weight and zero contribution
        to the S_U/S_V sums driving the total candidate-edge count) but dissolution depends
        only on q_short/q_long, never theta, so existing partnerships are untouched -- by
        design, per project discussion (2026-07-30): the gate manufactures new-partnership-
        formation singleness, it does not force anyone already partnered to become single.
        Independent draw every call -- no persistence/memory across years.

        Surviving (ungated) thetas are scaled up by 1/(1-p) so that sum(theta) is preserved
        in expectation. This matters because _sample_edges draws its TOTAL candidate count as
        Poisson(scale * S_U * S_V) -- without the rescale, gating p on both sides would cut
        the whole network's edge count by (1-p)**2 (0.64 at p=0.2, measured as a ~30% drop in
        mean degree), which acbnm.calibrate() cannot see or compensate for since it calibrates
        rho on a gate-free cal model. The outer calibration loop
        (calibrate_community_powerlaw.py) would then raise mean_partners_per_year to undo the
        shortfall, redistributing the lost formation back onto the ungated majority and
        cancelling the gate's effect entirely -- which is exactly what happened before this
        rescale was added. With it, the gate changes WHO forms ties, not HOW MANY exist.

        A dict keyed by uid (rather than a plain array parallel to state['u_theta']) is used
        for self._theta_true_u/_v because nothing in the vendored _drop_side_nodes (which
        shrinks state's own arrays in lockstep with state['u_uid']/['v_uid'] on death) knows
        about a second array living outside `state` -- a dict avoids needing to re-derive
        the same keep-mask ourselves to stay aligned. Dead uids are popped from these dicts
        in step() (see the dead_female/dead_male cleanup there) to avoid unbounded growth.
        '''
        state, rng, p = self._state, self._rng, self._p_single_annual
        boost = 1.0 / (1.0 - p) if p < 1.0 else 0.0  # p==1 => everyone gated, S_U==0 (guarded in _sample_edges)

        shortfall = {}
        for side, uid_key, theta_key, edge_key, true_map in (
                ('u', 'u_uid', 'u_theta', 'edges_u', self._theta_true_u),
                ('v', 'v_uid', 'v_theta', 'edges_v', self._theta_true_v)):
            uid = state[uid_key]
            true_theta = np.fromiter((true_map[u] for u in uid.tolist()),
                                     dtype=float, count=uid.size)
            gated = self._choose_gated(uid, state[edge_key], p, rng)
            shortfall[side] = int(round(p * uid.size)) - int(gated.size)
            theta = true_theta * boost
            theta[gated] = 0.0
            state[theta_key] = theta
        self._gate_shortfall = (shortfall['u'], shortfall['v'])
        return

    @staticmethod
    def _choose_gated(uid, edges_side, p, rng):
        '''
        Pick the positions to gate (theta -> 0) for the coming year: a fresh, uniformly
        random sample of size round(p * N) drawn PREFERENTIALLY FROM THE CURRENTLY-
        UNPARTNERED.

        Why not a plain Bernoulli(p) over everyone: the gate is formation-only by design (it
        never dissolves an existing tie), so gating someone who already holds a partnership
        does not make them single -- they simply keep that partner all year. A uniform gate
        therefore delivers realised singleness of only p * P(unpartnered | gated), which
        measured 0.20 * 0.41 = 0.083 against a 0.20 input. Drawing the gated set from people
        who are already unpartnered makes every gated person genuinely single, so realised
        singleness lands on p by construction -- which is the whole point of the knob.

        If fewer than round(p*N) people are currently unpartnered, all of them are gated and
        the remainder is topped up at random from the partnered (those top-ups keep their
        partners and so will not register as single -- the caller records the resulting
        shortfall on self._gate_shortfall rather than silently missing the target).

        The draw is independent each year: someone gated this year is no more or less likely
        to be gated next year, beyond the fact that being single is itself persistent.
        '''
        n_target = int(round(p * uid.size))
        if n_target <= 0:
            return np.empty(0, dtype=np.int64)
        deg = np.bincount(CommunityNetworkBackend._positions_of(uid, edges_side),
                          minlength=uid.size)
        unpartnered = np.where(deg == 0)[0]
        if unpartnered.size >= n_target:
            return rng.choice(unpartnered, size=n_target, replace=False)
        partnered = np.where(deg > 0)[0]
        n_extra = min(n_target - unpartnered.size, partnered.size)
        if n_extra <= 0:
            return unpartnered
        return np.concatenate([unpartnered, rng.choice(partnered, size=n_extra, replace=False)])

    @staticmethod
    def _estimate_edge_mortality_hazard(sim, params, u_real, v_real):
        '''
        Approximate per-edge monthly removal hazard due to death: the chance that EITHER
        endpoint of a partnership dies in a given month, so mu_edge ~ mu_female + mu_male.

        Read straight off HPVsim's own death-rate tables (sim['death_rates'], the same ones
        People.apply_death_rates() uses), evaluated over the sexually-active population at
        t=0. Per-sex rates are averaged with weights proportional to each person's EXPECTED
        DEGREE under the age-mixing kernel, not uniformly -- the kernel pushes partnerships
        toward older ages, so a flat average over active people would understate the hazard
        actual edge endpoints face. Expected degree for a woman in age band g is
        proportional to sum_h A[g, h] * (men in band h), and symmetrically for men; A is
        indexed exactly as _sample_edges indexes it. Community mixing is left out of the
        weighting since community is assigned independently of age, so it cancels.

        Deliberately approximate: it fixes the age distribution and death rates at t=0
        (mortality falls over the run) and ignores theta. Residual error is absorbed by the
        outer calibration loop (calibrate_community_powerlaw.py), which measures empirically.
        Returns 0.0 if no death-rate data is available.
        '''
        people = sim.people
        death_pars = sim.pars.get('death_rates')
        if not death_pars:
            return 0.0
        all_years = np.array(list(death_pars.keys()))
        age_bins = death_pars[all_years[0]]['m'][:, 0]
        nearest = all_years[sc.findnearest(all_years, sim['start'])]
        mx = {'f': death_pars[nearest]['f'][:, 1], 'm': death_pars[nearest]['m'][:, 1]}

        A = np.asarray(params['A_age'], dtype=float)
        n_bands = int(params['n_bands'])
        band_edges = params['age_band_edges']
        band_f = _age_to_band(people.age[u_real].astype(float), band_edges)
        band_m = _age_to_band(people.age[v_real].astype(float), band_edges)
        n_f = np.bincount(band_f, minlength=n_bands).astype(float)
        n_m = np.bincount(band_m, minlength=n_bands).astype(float)
        # Expected-degree profile per band (A[female_band, male_band], matching _sample_edges)
        w_by_band = {'f': A @ n_m, 'm': A.T @ n_f}

        mu = {}
        for key, inds, bands in (('f', u_real, band_f), ('m', v_real, band_m)):
            rates = mx[key]
            age_inds = np.clip(np.digitize(people.age[inds], age_bins) - 1, 0, rates.size - 1)
            w = w_by_band[key][bands]
            wsum = w.sum()
            mu[key] = (float(np.average(rates[age_inds], weights=w)) if wsum > 0
                       else float(np.mean(rates[age_inds]))) / 12.0  # Annual rate -> monthly
        return (mu['f'] + mu['m']) * float(sim['rel_death'])

    @staticmethod
    def _positions_of(uid_arr, query):
        ''' Positions within uid_arr of each entry of query (uid_arr need not be sorted) '''
        if query.size == 0:
            return np.empty(0, dtype=np.int64)
        order = np.argsort(uid_arr, kind='stable')
        return order[np.searchsorted(uid_arr[order], query)]

    def _force_pair_ungated(self):
        '''
        Guarantee that every UNGATED person holds at least one partnership, by pairing up
        those who currently have none. Called at each 12-month boundary immediately after
        _refresh_annual_gate(), so `theta > 0` cleanly identifies "ungated this year".

        Why this exists (module docstring point 4, and the project discussion of 2026-07-30):
        the model allocates partnerships by drawing a Poisson total and handing endpoints out
        in proportion to theta, so each person's partner count is Poisson with mean
        proportional to their own theta. For ANY theta distribution, Jensen's inequality then
        forces P(0 partners) >= exp(-population mean degree) -- and heterogeneity only ever
        ADDS zeros. With the Natsal targets (mean 1.4 partners/yr among the partnered, 20%
        single => population mean 1.12) that floor is exp(-1.12) = 33%, so a purely
        propensity-driven network cannot reach 20% singleness no matter how rho, alpha or the
        annual gate are set. Real partner-count data is UNDER-dispersed at zero (pair
        bonding), which a Poisson mixture cannot represent. This method supplies the missing
        structure, mirroring how HPVsim's own 'default' network guarantees participants at
        least one partner via its 'poisson1' (Poisson+1) partner-count distributions; here
        singleness becomes a controlled input (p_single_annual) rather than a residual.

        Pairing respects the age/community mixing kernel: unpartnered women in block g are
        allocated across man-blocks h in proportion to W[g, h] * (unpartnered men in h),
        capped by the men actually available in each block. W is indexed exactly as
        _sample_edges indexes it, so this method inherits the same convention (see the
        deferred-transpose note in the project plan) rather than silently half-fixing it.

        LIMITATION: only min(#free women, #free men) pairs can form -- and rather fewer in
        practice, since the mixing kernel can leave a woman's compatible man-blocks empty. So
        some ungated people do stay unpartnered, and realised singleness settles a little
        BELOW p_single_annual overall (gated people who already hold a partnership keep it and
        so are not single). self._force_pair_residual records
        (unpaired women, unpaired men) from the most recent call -- check it rather than
        assuming the guarantee is exact.
        '''
        state, params, rng = self._state, self._params, self._rng
        W = params['W_block']

        u_uid, v_uid = state['u_uid'], state['v_uid']
        deg_u = np.bincount(self._positions_of(u_uid, state['edges_u']), minlength=u_uid.size)
        deg_v = np.bincount(self._positions_of(v_uid, state['edges_v']), minlength=v_uid.size)

        free_u = np.where((deg_u == 0) & (state['u_theta'] > 0))[0]
        free_v = np.where((deg_v == 0) & (state['v_theta'] > 0))[0]
        self._force_pair_residual = (int(free_u.size), int(free_v.size))  # Updated below once paired
        if free_u.size == 0 or free_v.size == 0:
            return

        n_blocks = int(params['n_blocks'])
        bu, bv = state['u_block'][free_u], state['v_block'][free_v]
        # Per man-block pools of still-available free men, consumed as we go
        pool_v = [free_v[bv == h] for h in range(n_blocks)]
        avail = np.array([p.size for p in pool_v], dtype=np.int64)
        for pool in pool_v:
            rng.shuffle(pool)
        taken = np.zeros(n_blocks, dtype=np.int64)

        pair_u, pair_v = [], []
        # Random block order so no woman-block systematically gets first pick of scarce men
        for g in rng.permutation(n_blocks):
            women = free_u[bu == g]
            if women.size == 0:
                continue
            remaining = avail - taken
            probs = W[g, :] * remaining
            total = probs.sum()
            if total <= 0:
                continue  # No compatible man-block has anyone left
            counts = rng.multinomial(women.size, probs / total)
            counts = np.minimum(counts, remaining)  # Never over-draw a block
            n_take = int(counts.sum())
            if n_take == 0:
                continue
            rng.shuffle(women)
            pair_u.append(women[:n_take])
            for h in np.nonzero(counts)[0]:
                c = int(counts[h])
                pair_v.append(pool_v[h][taken[h]:taken[h] + c])
                taken[h] += c

        if not pair_u:
            return
        new_u_pos = np.concatenate(pair_u)
        new_v_pos = np.concatenate(pair_v)
        n_new = new_u_pos.size
        self._force_pair_residual = (int(free_u.size - n_new), int(free_v.size - n_new))

        # Both endpoints had degree 0, so no new edge can duplicate an existing one.
        new_types = (rng.random(n_new) < float(params['p_form_long'])).astype(np.int8)
        state['edges_u'] = np.concatenate([state['edges_u'], u_uid[new_u_pos]])
        state['edges_v'] = np.concatenate([state['edges_v'], v_uid[new_v_pos]])
        state['edges_type'] = np.concatenate([state['edges_type'], new_types])
        state['edge_birth'] = np.concatenate(
            [state['edge_birth'], np.full(n_new, self._month, dtype=np.int64)])
        return

    def _refresh_bands(self, people):
        '''
        Re-derive age (and hence age band/block) for every currently-tracked node directly from
        real HPVsim ages -- this is the "hook up per timestep" referred to in the module
        docstring: ageing itself is handled entirely by HPVsim (people.age), not by the network
        model, so this backend just re-reads it each step. Community never changes once assigned.
        '''
        state, params = self._state, self._params
        band_edges, n_bands = params['age_band_edges'], params['n_bands']

        state['u_age'] = people.age[state['u_uid']].astype(float)
        state['u_band'] = _age_to_band(state['u_age'], band_edges)
        state['u_block'] = _block_of(state['u_comm'], state['u_band'], n_bands)

        state['v_age'] = people.age[state['v_uid']].astype(float)
        state['v_band'] = _age_to_band(state['v_age'], band_edges)
        state['v_block'] = _block_of(state['v_comm'], state['v_band'], n_bands)
        return

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

        # Ages (and hence bands/blocks) of every surviving node are refreshed from real HPVsim
        # ages before this step's partnership formation runs (module docstring point 1). Nodes
        # about to die this step get refreshed too, harmlessly -- they're removed inside
        # network_step() via extra_deaths_U/V regardless of their age.
        self._refresh_bands(people)

        # Debut arrivals: people who just became sexually active join the network as brand-new
        # nodes, with a freshly-sampled community tag (not triggered by literal HPVsim births --
        # see module docstring point 3).
        is_active_now = people.is_active
        prev_len, cur_len = self._active_prev.size, is_active_now.size
        if cur_len > prev_len:
            active_prev_padded = np.concatenate([self._active_prev, np.zeros(cur_len - prev_len, dtype=bool)])
        else:
            active_prev_padded = self._active_prev
        new_idx = np.where(is_active_now & ~active_prev_padded)[0]
        new_female = new_idx[people.sex[new_idx] == 0].astype(np.int64)
        new_male = new_idx[people.sex[new_idx] == 1].astype(np.int64)
        self._inject_arrivals(new_female, new_male, people)

        # This HPVsim step covers self._months_per_step network-months (e.g. 3, for dt=0.25).
        # Added edges are computed as a union of newly-seen edges across each monthly sub-step
        # boundary within the HPVsim timestep, so nothing formed-then-dissolved mid-step is missed.
        prev_snap = acbnm.build_snapshot(self._state, self._model)
        seen_keys = self._triple_keys(prev_snap)  # Everything already live/in people.contacts

        added_u_parts, added_v_parts, added_t_parts = [], [], []
        month_snap = prev_snap
        for i in range(self._months_per_step):
            self._month += 1
            if self._p_single_annual > 0 and self._month % 12 == 0:
                self._refresh_annual_gate()
                self._force_pair_ungated()  # Must follow the gate: theta>0 identifies "ungated"
            acbnm.network_step(
                self._state, self._model, self._params, self._rng, self._month,
                extra_deaths_U=(dead_female if i == 0 else None),
                extra_deaths_V=(dead_male if i == 0 else None),
                track_durations=False,
            )
            month_snap = acbnm.build_snapshot(self._state, self._model)
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

        # Prevent unbounded growth of self._theta_true_u/_v (module docstring point 4): nothing
        # in the vendored _drop_side_nodes touches these separate uid-keyed dicts when it shrinks
        # state's own arrays, so dead uids must be popped explicitly. Safe/no-op for uids never
        # recorded (e.g. someone who died before debut). Done here, AFTER the monthly loop above
        # (not earlier in this method) -- dead_female/dead_male are only actually removed from
        # state['u_uid']/['v_uid'] inside that loop's first network_step() call (i==0), so
        # popping them from the dict any earlier would desync it from state['u_uid'] for the
        # remainder of this method: a same-step _refresh_annual_gate() call (line ~364, if
        # self._month%12==0 lands on this step's i==0) would still see the about-to-die uids in
        # state['u_uid'] but no longer find them in the dict -- a KeyError hit in practice during
        # the first full-scale recalibration run after this gate was added.
        for uid in dead_female.tolist():
            self._theta_true_u.pop(uid, None)
        for uid in dead_male.tolist():
            self._theta_true_v.pop(uid, None)

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

    def _inject_arrivals(self, new_female, new_male, people):
        state, params, rng = self._state, self._params, self._rng
        band_edges, n_bands = params['age_band_edges'], params['n_bands']
        n_comm = int(params['n_communities'])
        community_probs = params['community_probs']
        floor = params.get('theta_floor', 0.0)

        if new_female.size:
            theta = _sample_side_theta(new_female.size, params['gamma_shape_U'], rng, floor=floor)
            # Record true (ungated) theta for the annual-singleness gate (module docstring
            # point 4) -- new debuts stay ungated until the next 12-month boundary redraws
            # everyone's mask, since _refresh_annual_gate() rebuilds state['u_theta'] from
            # this dict rather than masking it in place.
            for uid, th in zip(new_female.tolist(), theta.tolist()):
                self._theta_true_u[uid] = th
            age = people.age[new_female].astype(float)
            comm = rng.choice(n_comm, size=new_female.size, p=community_probs).astype(np.int16)
            people.community[new_female] = comm  #NEW!! mirror onto People (see initialize())
            band = _age_to_band(age, band_edges)
            block = _block_of(comm, band, n_bands)
            state['u_uid'] = np.concatenate([state['u_uid'], new_female])
            state['u_theta'] = np.concatenate([state['u_theta'], theta])
            state['u_entry'] = np.concatenate(
                [state['u_entry'], np.full(new_female.size, acbnm.ENTRY_BIRTH, dtype=np.int8)])
            state['u_age'] = np.concatenate([state['u_age'], age])
            state['u_comm'] = np.concatenate([state['u_comm'], comm])
            state['u_band'] = np.concatenate([state['u_band'], band])
            state['u_block'] = np.concatenate([state['u_block'], block])
        if new_male.size:
            theta = _sample_side_theta(new_male.size, params['gamma_shape_V'], rng, floor=floor)
            for uid, th in zip(new_male.tolist(), theta.tolist()):
                self._theta_true_v[uid] = th
            age = people.age[new_male].astype(float)
            comm = rng.choice(n_comm, size=new_male.size, p=community_probs).astype(np.int16)
            people.community[new_male] = comm  #NEW!! mirror onto People (see initialize())
            band = _age_to_band(age, band_edges)
            block = _block_of(comm, band, n_bands)
            state['v_uid'] = np.concatenate([state['v_uid'], new_male])
            state['v_theta'] = np.concatenate([state['v_theta'], theta])
            state['v_entry'] = np.concatenate(
                [state['v_entry'], np.full(new_male.size, acbnm.ENTRY_BIRTH, dtype=np.int8)])
            state['v_age'] = np.concatenate([state['v_age'], age])
            state['v_comm'] = np.concatenate([state['v_comm'], comm])
            state['v_band'] = np.concatenate([state['v_band'], band])
            state['v_block'] = np.concatenate([state['v_block'], block])
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
        # Placeholder only -- dissolution is owned by the network model's own monthly hazard, not
        # by HPVsim reaching this dur/end date
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
        for lkey, ty, q in ((self.LKEY_SHORT, acbnm.EDGE_SHORT, params['q_short']),
                            (self.LKEY_LONG, acbnm.EDGE_LONG, params['q_long'])):
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
        for lkey, ty in ((self.LKEY_SHORT, acbnm.EDGE_SHORT), (self.LKEY_LONG, acbnm.EDGE_LONG)):
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

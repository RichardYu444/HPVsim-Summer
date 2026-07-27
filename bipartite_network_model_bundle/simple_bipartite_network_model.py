#!/usr/bin/env python3
"""
simple_bipartite_network_model_revised
======================================

A compact temporal bipartite contact-network simulator.

The population is divided into two disjoint node sets, ``U`` and ``V``. Every
edge joins one U node to one V node. Nodes carry Gamma-distributed partner
propensities, partnerships form and dissolve monthly, natural deaths are
replaced exactly by births, and externally supplied deaths are not replaced.

Partnerships have two duration classes:

* ``EDGE_SHORT = 0``
* ``EDGE_LONG = 1``

``pi_long`` is the target standing fraction of long partnerships. The model
derives the formation probability required to produce that standing mixture.
All mixture-dependent quantities are kept consistent: effective duration,
formation rate, instantaneous mean degree, and analytic edge density.

The internal timestep is one month. Outputs are three-month interval networks.
For each quarter, the emitted network is the union of all nodes and edges present
at monthly boundaries in that interval. For ``T`` quarters, ``simulate`` returns
``T`` networks and ``T + 1`` change times; network ``i`` applies on
``[change_times[i], change_times[i + 1])``.

Side-specific UIDs are encoded for epidemic use without collisions:

* U node ``u`` -> global ID ``2*u``
* V node ``v`` -> global ID ``2*v + 1``

Public functions
----------------
``interpretable_to_params``
``build_model``
``init_network_state``
``network_step``
``build_snapshot``
``simulate``
``calibrate``
``default_burn_months``
``encode_epidemic_node_ids``
"""

import os

import numpy as np
from scipy.sparse import csr_matrix

ENTRY_INIT = 0
ENTRY_BIRTH = 1
EDGE_SHORT = 0
EDGE_LONG = 1
OBS_WINDOW_MONTHS = 12.0
MONTHS_PER_STEP = 3


def _validate_size(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _nonnegative_int(value, name):
    result = int(value)
    if result < 0 or result != value:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _rho_for_pooled_mean_degree(mean_degree, nU, nV):
    """Density giving the requested mean degree pooled over U and V."""
    return float(mean_degree) * (nU + nV) / (2.0 * nU * nV)


def sample_gamma(n, shape, scale=1.0, rng=None):
    """Sample ``n`` values from a Gamma(shape, scale) distribution."""
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("Gamma shape and scale must be > 0")
    if rng is None:
        rng = np.random.default_rng()
    return rng.gamma(shape, scale, size=int(n))


def gamma_mean(shape, scale=1.0):
    """Analytical mean of a Gamma(shape, scale) distribution."""
    return float(shape) * float(scale)


def _sample_side_theta(n, shape, rng, exact_mean_one=False):
    values = sample_gamma(n, shape, scale=1.0 / shape, rng=rng)
    if exact_mean_one and n:
        values /= values.mean()
    return values


def _lookup_sorted_positions(sorted_ids, query_ids):
    sorted_ids = np.asarray(sorted_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if query_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, query_ids)
    valid = pos < sorted_ids.size
    matched = np.zeros(query_ids.size, dtype=bool)
    matched[valid] = sorted_ids[pos[valid]] == query_ids[valid]
    if not np.all(matched):
        raise ValueError(f"edge references unknown node IDs: {query_ids[~matched][:5].tolist()}")
    return pos


def _canonicalize_edges(edges_u, edges_v, edges_type):
    """Collapse duplicate pairs, retaining the largest type value."""
    eu = np.asarray(edges_u, dtype=np.int64)
    ev = np.asarray(edges_v, dtype=np.int64)
    et = np.asarray(edges_type, dtype=np.int8)
    if not (eu.size == ev.size == et.size):
        raise ValueError("edge arrays must have equal length")
    edge_map = {}
    for u, v, ty in zip(eu.tolist(), ev.tolist(), et.tolist()):
        if ty not in (EDGE_SHORT, EDGE_LONG):
            raise ValueError(f"invalid edge type: {ty}")
        key = (int(u), int(v))
        edge_map[key] = max(edge_map.get(key, -1), int(ty))
    pairs = sorted(edge_map)
    out_u = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    out_v = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    out_t = np.fromiter((edge_map[p] for p in pairs), dtype=np.int8, count=len(pairs))
    return out_u, out_v, out_t


def _refresh_derived_params(params, nU, nV, *, reset_rho=False):
    """Refresh every parameter determined by the partnership mixture."""
    p = float(params["p_form_long"])
    D_short = float(params["D_mean_short"])
    D_long = float(params["D_mean_long"])
    D_eff = (1.0 - p) * D_short + p * D_long
    q_eff = 1.0 / D_eff
    observation_scale = 1.0 + OBS_WINDOW_MONTHS * q_eff
    k_snap = float(params["mean_partners_per_year"]) / observation_scale
    rho_analytic = _rho_for_pooled_mean_degree(k_snap, nU, nV)
    params.update(
        D_mean_eff=D_eff,
        q=q_eff,
        observation_scale=observation_scale,
        k_snap=k_snap,
        rho_analytic=rho_analytic,
    )
    if reset_rho:
        params["rho"] = rho_analytic
    return rho_analytic


def interpretable_to_params(user_params, nU, nV=None, *, verbose=True):
    """
    Convert interpretable inputs into runtime parameters.

    ``mean_partners_per_year`` is the pooled mean number of distinct partners
    observed over 12 months. ``gamma_shape`` controls propensity heterogeneity.
    ``D_mean_short`` and ``D_mean_long`` are monthly geometric mean durations.
    ``pi_long`` is the target standing fraction of long edges. ``tau`` is the
    mean time to natural death in months.
    """
    if nV is None:
        nV = nU
    NU = _validate_size(nU, "nU")
    NV = _validate_size(nV, "nV")

    shape_default = float(user_params.get("gamma_shape", 1.0))
    shape_U = float(user_params.get("gamma_shape_U", shape_default))
    shape_V = float(user_params.get("gamma_shape_V", shape_default))
    if shape_U <= 0.0 or shape_V <= 0.0:
        raise ValueError("Gamma shapes must be > 0")

    D_default = float(user_params.get("D_mean", 0.0))
    D_short = float(user_params.get("D_mean_short", D_default))
    D_long = float(user_params.get("D_mean_long", D_default))
    if D_short < 1.0 or D_long < 1.0:
        raise ValueError("mean durations must be >= 1 month")

    frac_long = float(user_params.get("frac_long", user_params.get("pi_long", 0.0)))
    if not 0.0 <= frac_long <= 1.0:
        raise ValueError("pi_long must be in [0, 1]")
    if frac_long == 0.0:
        p_form_long = 0.0
    elif frac_long == 1.0:
        p_form_long = 1.0
    else:
        p_form_long = (frac_long * D_short) / (
            (1.0 - frac_long) * D_long + frac_long * D_short
        )

    k_year = float(user_params.get("mean_partners_per_year", user_params.get("k_mean", 0.0)))
    if k_year <= 0.0:
        raise ValueError("mean_partners_per_year must be > 0")
    tau = float(user_params.get("tau", 360.0))
    if tau <= 0.0:
        raise ValueError("tau must be > 0")

    params = {
        "gamma_shape_U": shape_U,
        "gamma_shape_V": shape_V,
        "q_short": 1.0 / D_short,
        "q_long": 1.0 / D_long,
        "p_form_long": p_form_long,
        "delta": 1.0 / tau,
        "frac_long_target": frac_long,
        "D_mean_short": D_short,
        "D_mean_long": D_long,
        "tau": tau,
        "mean_partners_per_year": k_year,
    }
    _refresh_derived_params(params, NU, NV, reset_rho=True)

    if verbose:
        expected_edges = params["rho"] * NU * NV
        print("-" * 68)
        print("Bipartite temporal network parameters")
        print("-" * 68)
        print(f"  population: U={NU}, V={NV}")
        print(f"  partners/year, pooled: {k_year:.4f}")
        print(f"  instantaneous degree, pooled: {params['k_snap']:.4f}")
        print(f"  instantaneous degree by side: U={expected_edges/NU:.4f}, V={expected_edges/NV:.4f}")
        print(f"  Gamma shapes: U={shape_U:.4f}, V={shape_V:.4f}")
        print(f"  durations: short={D_short:.4f}, long={D_long:.4f} months")
        print(f"  standing long-edge target: {frac_long:.4f}")
        print(f"  long-edge formation probability: {p_form_long:.6f}")
        print(f"  effective mixture duration: {params['D_mean_eff']:.4f} months")
        print(f"  monthly natural mortality: {params['delta']:.6f}")
        print(f"  analytic rho: {params['rho']:.6e}")
        print("-" * 68)
    return params


def build_model(nU, nV=None):
    """Create the static population descriptor."""
    if nV is None:
        nV = nU
    return {
        "nU_init": _validate_size(nU, "nU"),
        "nV_init": _validate_size(nV, "nV"),
        "months_per_step": MONTHS_PER_STEP,
    }


def encode_epidemic_node_ids(u_ids, v_ids):
    """Encode UIDs as disjoint even and odd signed 64-bit integers."""
    u = np.asarray(u_ids, dtype=np.int64)
    v = np.asarray(v_ids, dtype=np.int64)
    max_safe = np.iinfo(np.int64).max // 2
    if (u.size and (u.min() < 0 or u.max() > max_safe)) or (
        v.size and (v.min() < 0 or v.max() > max_safe)
    ):
        raise OverflowError("UID outside the supported int64 encoding range")
    return 2 * u, 2 * v + 1


def init_network_state(model, params, rng):
    """Create the initial node arrays and sample a stationary-density graph."""
    NU, NV = model["nU_init"], model["nV_init"]
    state = {
        "u_uid": np.arange(NU, dtype=np.int64),
        "u_theta": _sample_side_theta(NU, params["gamma_shape_U"], rng, True),
        "u_entry": np.full(NU, ENTRY_INIT, dtype=np.int8),
        "v_uid": np.arange(NV, dtype=np.int64),
        "v_theta": _sample_side_theta(NV, params["gamma_shape_V"], rng, True),
        "v_entry": np.full(NV, ENTRY_INIT, dtype=np.int8),
        "edges_u": np.empty(0, dtype=np.int64),
        "edges_v": np.empty(0, dtype=np.int64),
        "edges_type": np.empty(0, dtype=np.int8),
        "edge_birth": np.empty(0, dtype=np.int64),
        "next_uid_U": NU,
        "next_uid_V": NV,
    }
    eu, ev, et = _sample_edges(params, state, params["rho"], rng)
    state["edges_u"], state["edges_v"], state["edges_type"] = eu, ev, et
    state["edge_birth"] = np.zeros(eu.size, dtype=np.int64)
    return state


def _sample_edges(params, state, scale, rng):
    """Sample unique new edges with propensity-weighted endpoints."""
    empty = (
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int8),
    )
    u_uid, v_uid = state["u_uid"], state["v_uid"]
    if not u_uid.size or not v_uid.size or scale <= 0.0:
        return empty
    theta, phi = state["u_theta"], state["v_theta"]
    sum_u, sum_v = float(theta.sum()), float(phi.sum())
    if sum_u <= 0.0 or sum_v <= 0.0:
        return empty
    m = int(rng.poisson(scale * sum_u * sum_v))
    if m == 0:
        return empty

    iu = np.searchsorted(np.cumsum(theta) / sum_u, rng.random(m), side="right")
    iv = np.searchsorted(np.cumsum(phi) / sum_v, rng.random(m), side="right")
    a = u_uid[np.minimum(iu, u_uid.size - 1)]
    b = v_uid[np.minimum(iv, v_uid.size - 1)]

    pairs = np.column_stack((a, b))
    _, first = np.unique(pairs, axis=0, return_index=True)
    first.sort()
    a, b = a[first], b[first]
    if state["edges_u"].size:
        existing = set(zip(state["edges_u"].tolist(), state["edges_v"].tolist()))
        fresh = np.fromiter(
            ((int(u), int(v)) not in existing for u, v in zip(a, b)),
            dtype=bool,
            count=a.size,
        )
        a, b = a[fresh], b[fresh]
    types = (rng.random(a.size) < params["p_form_long"]).astype(np.int8)
    return a.astype(np.int64), b.astype(np.int64), types

# =====================================================================
# Monthly dynamics
# =====================================================================


def _side_deaths(uid, delta, rng, extra_deaths):
    """Return disjoint natural and external deaths; external deaths take priority."""
    if extra_deaths is None:
        external = np.empty(0, dtype=np.int64)
    else:
        requested = np.unique(np.asarray(extra_deaths, dtype=np.int64))
        external = requested[np.isin(requested, uid)]
    eligible = uid[~np.isin(uid, external)] if external.size else uid
    if delta > 0.0 and eligible.size:
        natural = eligible[rng.random(eligible.size) < delta]
    else:
        natural = np.empty(0, dtype=np.int64)
    return natural, external


def _record_durations(durations, durations_type, t, state, mask):
    if durations is None:
        return
    durations.extend((t - state["edge_birth"][mask]).tolist())
    if durations_type is not None:
        durations_type.extend(state["edges_type"][mask].tolist())


def _add_births(state, params, rng, t, side, count, node_events):
    """Add exactly ``count`` births to one side."""
    count = int(count)
    if count <= 0:
        return
    if side == "U":
        uid_key, theta_key, entry_key, next_key = "u_uid", "u_theta", "u_entry", "next_uid_U"
        shape = params["gamma_shape_U"]
    else:
        uid_key, theta_key, entry_key, next_key = "v_uid", "v_theta", "v_entry", "next_uid_V"
        shape = params["gamma_shape_V"]
    first = state[next_key]
    born = np.arange(first, first + count, dtype=np.int64)
    state[next_key] += count
    state[uid_key] = np.concatenate([state[uid_key], born])
    state[theta_key] = np.concatenate([
        state[theta_key], _sample_side_theta(count, shape, rng)
    ])
    state[entry_key] = np.concatenate([
        state[entry_key], np.full(count, ENTRY_BIRTH, dtype=np.int8)
    ])
    node_events.extend((t, int(uid), side, "birth") for uid in born)


def _apply_turnover(params, state, rng, t, node_events, track_durations,
                    durations, durations_type, extra_deaths_U, extra_deaths_V):
    dead_U_nat, dead_U_ext = _side_deaths(
        state["u_uid"], params["delta"], rng, extra_deaths_U
    )
    dead_V_nat, dead_V_ext = _side_deaths(
        state["v_uid"], params["delta"], rng, extra_deaths_V
    )
    dead_U = np.concatenate([dead_U_nat, dead_U_ext])
    dead_V = np.concatenate([dead_V_nat, dead_V_ext])

    if dead_U.size or dead_V.size:
        incident = np.zeros(state["edges_u"].size, dtype=bool)
        if dead_U.size:
            incident |= np.isin(state["edges_u"], dead_U)
        if dead_V.size:
            incident |= np.isin(state["edges_v"], dead_V)
        if track_durations and incident.any():
            _record_durations(durations, durations_type, t, state, incident)
        keep_edges = ~incident
        for key in ("edges_u", "edges_v", "edges_type", "edge_birth"):
            state[key] = state[key][keep_edges]

    if dead_U.size:
        keep = ~np.isin(state["u_uid"], dead_U)
        state["u_uid"] = state["u_uid"][keep]
        state["u_theta"] = state["u_theta"][keep]
        state["u_entry"] = state["u_entry"][keep]
        node_events.extend((t, int(uid), "U", "death") for uid in dead_U_nat)
        node_events.extend((t, int(uid), "U", "external_death") for uid in dead_U_ext)
    if dead_V.size:
        keep = ~np.isin(state["v_uid"], dead_V)
        state["v_uid"] = state["v_uid"][keep]
        state["v_theta"] = state["v_theta"][keep]
        state["v_entry"] = state["v_entry"][keep]
        node_events.extend((t, int(uid), "V", "death") for uid in dead_V_nat)
        node_events.extend((t, int(uid), "V", "external_death") for uid in dead_V_ext)

    _add_births(state, params, rng, t, "U", dead_U_nat.size, node_events)
    _add_births(state, params, rng, t, "V", dead_V_nat.size, node_events)


def _density_multiplier(model, state):
    """Preserve pooled mean degree after externally driven population loss."""
    nU, nV = state["u_uid"].size, state["v_uid"].size
    if nU == 0 or nV == 0:
        return 0.0
    NU, NV = model["nU_init"], model["nV_init"]
    return ((nU + nV) / (nU * nV)) / ((NU + NV) / (NU * NV))


def network_step(state, model, params, rng, t, *, extra_deaths_U=None,
                 extra_deaths_V=None, track_durations=False, durations=None,
                 durations_type=None, node_events=None):
    """
    Advance the network by one month.

    The update order is turnover, dissolution, and formation. Natural deaths are
    replaced exactly; external deaths are not replaced.
    """
    if node_events is None:
        node_events = []
    if track_durations and durations is None:
        durations = []
    has_external = (
        extra_deaths_U is not None and len(extra_deaths_U) > 0
    ) or (
        extra_deaths_V is not None and len(extra_deaths_V) > 0
    )
    if params["delta"] > 0.0 or has_external:
        _apply_turnover(
            params, state, rng, t, node_events, track_durations,
            durations, durations_type, extra_deaths_U, extra_deaths_V
        )

    if state["edges_u"].size:
        q_edge = np.where(
            state["edges_type"] == EDGE_LONG,
            params["q_long"],
            params["q_short"],
        )
        drop = rng.random(q_edge.size) < q_edge
        if track_durations and drop.any():
            _record_durations(durations, durations_type, t, state, drop)
        keep = ~drop
        for key in ("edges_u", "edges_v", "edges_type", "edge_birth"):
            state[key] = state[key][keep]

    scale = params["q"] * params["rho"] * _density_multiplier(model, state)
    eu, ev, et = _sample_edges(params, state, scale, rng)
    if eu.size:
        state["edges_u"] = np.concatenate([state["edges_u"], eu])
        state["edges_v"] = np.concatenate([state["edges_v"], ev])
        state["edges_type"] = np.concatenate([state["edges_type"], et])
        state["edge_birth"] = np.concatenate([
            state["edge_birth"], np.full(eu.size, t, dtype=np.int64)
        ])


# =====================================================================
# Snapshots
# =====================================================================


def _active_dict(uid, entry):
    return {int(node): {"entry": int(kind)} for node, kind in zip(uid, entry)}


def build_snapshot(state, model, *, edges_u=None, edges_v=None, edges_type=None):
    """Return an instantaneous raw snapshot of the current active graph."""
    del model
    if edges_u is None:
        edges_u, edges_v, edges_type = (
            state["edges_u"], state["edges_v"], state["edges_type"]
        )
    eu, ev, et = _canonicalize_edges(edges_u, edges_v, edges_type)
    if eu.size:
        alive = np.isin(eu, state["u_uid"]) & np.isin(ev, state["v_uid"])
        eu, ev, et = eu[alive], ev[alive], et[alive]
    return {
        "edges_u": eu.copy(),
        "edges_v": ev.copy(),
        "edges_type": et.copy(),
        "active_U": _active_dict(state["u_uid"], state["u_entry"]),
        "active_V": _active_dict(state["v_uid"], state["v_entry"]),
    }


def _accumulate_interval(state, edge_map, active_U, active_V):
    """Add one monthly boundary state to an interval accumulator."""
    active_U.update(_active_dict(state["u_uid"], state["u_entry"]))
    active_V.update(_active_dict(state["v_uid"], state["v_entry"]))
    for u, v, ty in zip(
        state["edges_u"].tolist(),
        state["edges_v"].tolist(),
        state["edges_type"].tolist(),
    ):
        key = (int(u), int(v))
        edge_map[key] = max(edge_map.get(key, -1), int(ty))


def _interval_snapshot(edge_map, active_U, active_V):
    pairs = sorted(edge_map)
    return {
        "edges_u": np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs)),
        "edges_v": np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs)),
        "edges_type": np.fromiter((edge_map[p] for p in pairs), dtype=np.int8, count=len(pairs)),
        "active_U": dict(active_U),
        "active_V": dict(active_V),
    }


def _build_epidemic_snapshot(snapshot):
    """Convert a raw snapshot to sparse adjacency and aligned node metadata."""
    u_ids = np.asarray(sorted(snapshot["active_U"]), dtype=np.int64)
    v_ids = np.asarray(sorted(snapshot["active_V"]), dtype=np.int64)
    epi_u, epi_v = encode_epidemic_node_ids(u_ids, v_ids)
    active_nodes = np.concatenate([epi_u, epi_v])
    node_types = np.concatenate([
        np.zeros(u_ids.size, dtype=np.int8),
        np.ones(v_ids.size, dtype=np.int8),
    ])
    entry_kind = np.asarray(
        [snapshot["active_U"][int(uid)]["entry"] for uid in u_ids]
        + [snapshot["active_V"][int(uid)]["entry"] for uid in v_ids],
        dtype=np.int8,
    )

    eu, ev, et = _canonicalize_edges(
        snapshot["edges_u"], snapshot["edges_v"], snapshot["edges_type"]
    )
    n = active_nodes.size
    if eu.size == 0:
        return (
            csr_matrix((n, n), dtype=np.int8),
            csr_matrix((n, n), dtype=np.int8),
            active_nodes,
            node_types,
            entry_kind,
        )
    rows = _lookup_sorted_positions(u_ids, eu)
    cols = _lookup_sorted_positions(v_ids, ev) + u_ids.size
    rr = np.concatenate([rows, cols])
    cc = np.concatenate([cols, rows])
    adj = csr_matrix(
        (np.ones(rr.size, dtype=np.int8), (rr, cc)), shape=(n, n), dtype=np.int8
    )
    type_values = np.concatenate([et + 1, et + 1]).astype(np.int8)
    adj_type = csr_matrix((type_values, (rr, cc)), shape=(n, n), dtype=np.int8)
    return adj, adj_type, active_nodes, node_types, entry_kind

# =====================================================================
# Simulation
# =====================================================================


def default_burn_months(params):
    """Return a burn-in covering several long-partnership lifetimes."""
    return int(max(5.0 * params["D_mean_long"], 120.0))


def _persist_snapshot(path, snapshot, quarter):
    np.savez_compressed(
        path,
        edges_u=np.asarray(snapshot["edges_u"], dtype=np.int64),
        edges_v=np.asarray(snapshot["edges_v"], dtype=np.int64),
        edges_type=np.asarray(snapshot["edges_type"], dtype=np.int8),
        active_u=np.asarray(sorted(snapshot["active_U"]), dtype=np.int64),
        active_v=np.asarray(sorted(snapshot["active_V"]), dtype=np.int64),
        quarter=np.int64(quarter),
    )


def simulate(model, params, T, *, seed=1, burn_in_months=None,
             save_edge_snapshots=(), max_snapshots=50, track_durations=True,
             base_dir=None, return_raw_snapshots=True,
             extra_deaths_U_by_month=None, extra_deaths_V_by_month=None):
    """
    Simulate ``T`` quarterly intervals.

    The returned lists contain one integrated network per quarter. External-death
    schedules are dictionaries keyed by post-burn-in month, beginning at one.
    ``node_events`` uses the same relative month scale.
    """
    T = _nonnegative_int(T, "T")
    months_per_step = _validate_size(
        model.get("months_per_step", MONTHS_PER_STEP), "months_per_step"
    )
    if not 0.0 < params["q_short"] <= 1.0:
        raise ValueError("q_short must be in (0, 1]")
    if not 0.0 < params["q_long"] <= 1.0:
        raise ValueError("q_long must be in (0, 1]")

    if burn_in_months is None:
        burn_in_months = default_burn_months(params)
    burn_in_months = _nonnegative_int(burn_in_months, "burn_in_months")

    requested = sorted({_nonnegative_int(q, "snapshot quarter") for q in save_edge_snapshots})
    invalid = [q for q in requested if q < 1 or q > T]
    if invalid:
        raise ValueError(f"snapshot quarters must lie in 1..T: {invalid}")
    if max_snapshots is not None:
        max_snapshots = _nonnegative_int(max_snapshots, "max_snapshots")
        if len(requested) > max_snapshots:
            raise ValueError(
                f"requested {len(requested)} snapshots; max_snapshots={max_snapshots}"
            )

    if base_dir is None:
        base_dir = os.path.expanduser("~/Downloads/simple_bipartite_network")
    if requested:
        os.makedirs(base_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    state = init_network_state(model, params, rng)
    for month in range(1, burn_in_months + 1):
        network_step(state, model, params, rng, month)

    raw_snapshots = [] if return_raw_snapshots else None
    adj_list, adj_type_list = [], []
    active_nodes_list, node_types_list, entry_kind_list = [], [], []
    saved_paths, node_events_absolute = [], []
    durations = [] if track_durations else None
    durations_type = [] if track_durations else None
    save_set = set(requested)

    for quarter in range(1, T + 1):
        edge_map, active_U, active_V = {}, {}, {}
        _accumulate_interval(state, edge_map, active_U, active_V)
        for substep in range(months_per_step):
            relative_month = (quarter - 1) * months_per_step + substep + 1
            absolute_month = burn_in_months + relative_month
            deaths_U = (
                extra_deaths_U_by_month.get(relative_month)
                if extra_deaths_U_by_month is not None else None
            )
            deaths_V = (
                extra_deaths_V_by_month.get(relative_month)
                if extra_deaths_V_by_month is not None else None
            )
            network_step(
                state, model, params, rng, absolute_month,
                extra_deaths_U=deaths_U,
                extra_deaths_V=deaths_V,
                track_durations=track_durations,
                durations=durations,
                durations_type=durations_type,
                node_events=node_events_absolute,
            )
            _accumulate_interval(state, edge_map, active_U, active_V)

        snapshot = _interval_snapshot(edge_map, active_U, active_V)
        adj, adj_type, nodes, types, entries = _build_epidemic_snapshot(snapshot)
        adj_list.append(adj)
        adj_type_list.append(adj_type)
        active_nodes_list.append(nodes)
        node_types_list.append(types)
        entry_kind_list.append(entries)
        if return_raw_snapshots:
            raw_snapshots.append(snapshot)
        if quarter in save_set:
            path = os.path.join(base_dir, f"edges_q{quarter}.npz")
            _persist_snapshot(path, snapshot, quarter)
            saved_paths.append(path)

    final_month = burn_in_months + T * months_per_step
    node_events = [
        (month - burn_in_months, uid, side, event)
        for month, uid, side, event in node_events_absolute
    ]
    return {
        "T": T,
        "months_per_step": months_per_step,
        "state": state,
        "raw_snapshots": raw_snapshots,
        "node_events": node_events,
        "durations": durations,
        "durations_type": durations_type,
        "censored_ages": (
            (final_month - state["edge_birth"]).tolist() if track_durations else None
        ),
        "censored_types": state["edges_type"].tolist() if track_durations else None,
        "snapshots": saved_paths,
        "base_dir": base_dir,
        "adj_list": adj_list,
        "adj_type_list": adj_type_list,
        "active_nodes_list": active_nodes_list,
        "node_types_list": node_types_list,
        "entry_kind_list": entry_kind_list,
        "change_times": np.arange(T + 1, dtype=float),
        "epi_id_scheme": "U=2*uid, V=2*uid+1",
    }


# =====================================================================
# Calibration
# =====================================================================


def _measure_burnin(model, params, burn_months, seed):
    """Measure 12-month distinct partners and standing long-edge fraction."""
    rng = np.random.default_rng(seed)
    state = init_network_state(model, params, rng)
    for month in range(1, burn_months + 1):
        network_step(state, model, params, rng, month)

    partners_U, partners_V = {}, {}
    observed_U, observed_V = set(), set()
    long_fractions = []

    def observe_boundary():
        observed_U.update(state["u_uid"].tolist())
        observed_V.update(state["v_uid"].tolist())
        for u, v in zip(state["edges_u"].tolist(), state["edges_v"].tolist()):
            partners_U.setdefault(u, set()).add(v)
            partners_V.setdefault(v, set()).add(u)

    observe_boundary()
    for offset in range(1, 13):
        network_step(state, model, params, rng, burn_months + offset)
        observe_boundary()
        edge_type = state["edges_type"]
        long_fractions.append(
            float((edge_type == EDGE_LONG).mean()) if edge_type.size else 0.0
        )

    counts = [len(partners_U.get(uid, ())) for uid in observed_U]
    counts.extend(len(partners_V.get(uid, ())) for uid in observed_V)
    return (
        float(np.mean(counts)) if counts else 0.0,
        float(np.mean(long_fractions)) if long_fractions else 0.0,
    )


def _update_long_probability(current, target, realised):
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 1.0
    p = np.clip(current, 1e-8, 1.0 - 1e-8)
    f = np.clip(realised, 1e-8, 1.0 - 1e-8)
    odds = (p / (1.0 - p)) * (
        (target / (1.0 - target)) / (f / (1.0 - f))
    )
    return float(odds / (1.0 + odds))


def calibrate(model, params, *, n_cal=2500, burn_months=None, window=12,
              max_iters=6, tol=0.03, seed=12345, verbose=True):
    """
    Calibrate edge density and long-edge composition against annual targets.

    The calibration population preserves the requested U:V ratio. The returned
    parameter dictionary includes the measured validation values and a
    convergence flag.
    """
    if window != 12:
        raise ValueError("calibration window must be 12 months")
    max_iters = _validate_size(max_iters, "max_iters")
    if tol <= 0.0:
        raise ValueError("tol must be > 0")

    nU_cal = min(model["nU_init"], _validate_size(n_cal, "n_cal"))
    ratio = model["nV_init"] / model["nU_init"]
    nV_cal = max(1, round(nU_cal * ratio))
    cal_model = build_model(nU_cal, nV_cal)
    if burn_months is None:
        burn_months = default_burn_months(params)
    burn_months = _nonnegative_int(burn_months, "burn_months")

    target_k = float(params["mean_partners_per_year"])
    target_f = float(params["frac_long_target"])
    candidate = dict(params)
    _refresh_derived_params(candidate, nU_cal, nV_cal, reset_rho=True)

    if verbose:
        print("-" * 68)
        print(f"Calibration population: U={nU_cal}, V={nV_cal}")
        print(f"Targets: partners/year={target_k:.4f}, long fraction={target_f:.4f}")

    converged = False
    last_k = last_f = np.nan
    updated_after_measurement = False
    for iteration in range(max_iters):
        last_k, last_f = _measure_burnin(
            cal_model, candidate, burn_months, seed + iteration
        )
        ok_k = abs(last_k - target_k) <= tol * target_k
        ok_f = target_f in (0.0, 1.0) or abs(last_f - target_f) <= tol
        if verbose:
            print(
                f"  iteration {iteration + 1}: partners/year={last_k:.4f}, "
                f"long fraction={last_f:.4f}, p_form_long={candidate['p_form_long']:.6f}"
            )
        if ok_k and ok_f:
            converged = True
            updated_after_measurement = False
            break
        candidate["rho"] *= target_k / max(last_k, 1e-12)
        if not ok_f:
            candidate["p_form_long"] = _update_long_probability(
                candidate["p_form_long"], target_f, last_f
            )
        _refresh_derived_params(candidate, nU_cal, nV_cal, reset_rho=False)
        updated_after_measurement = True

    if updated_after_measurement:
        last_k, last_f = _measure_burnin(
            cal_model, candidate, burn_months, seed + max_iters
        )
        converged = (
            abs(last_k - target_k) <= tol * target_k
            and (target_f in (0.0, 1.0) or abs(last_f - target_f) <= tol)
        )
        if verbose:
            print(
                f"  validation: partners/year={last_k:.4f}, "
                f"long fraction={last_f:.4f}"
            )

    analytic_cal = _rho_for_pooled_mean_degree(candidate["k_snap"], nU_cal, nV_cal)
    correction = candidate["rho"] / analytic_cal
    out = dict(params)
    out["p_form_long"] = candidate["p_form_long"]
    _refresh_derived_params(out, model["nU_init"], model["nV_init"], reset_rho=True)
    out["rho"] *= correction
    out.update(
        rho_correction_factor=float(correction),
        calibrated=True,
        calibration_converged=bool(converged),
        calibration_realised_partners_per_year=float(last_k),
        calibration_realised_long_fraction=float(last_f),
    )
    if verbose:
        print(f"Density correction factor: {correction:.6f}")
        print(f"Production p_form_long: {out['p_form_long']:.6f}")
        print(f"Production D_mean_eff: {out['D_mean_eff']:.6f}")
        print(f"Production q: {out['q']:.6f}")
        print(f"Converged: {out['calibration_converged']}")
        print("-" * 68)
    return out


__all__ = [
    "interpretable_to_params", "build_model", "init_network_state",
    "build_snapshot", "network_step", "simulate", "calibrate",
    "default_burn_months", "encode_epidemic_node_ids", "sample_gamma",
    "gamma_mean", "MONTHS_PER_STEP", "OBS_WINDOW_MONTHS", "ENTRY_INIT",
    "ENTRY_BIRTH", "EDGE_SHORT", "EDGE_LONG",
]


if __name__ == "__main__":
    user_params = {
        "mean_partners_per_year": 2.0,
        "gamma_shape": 1.5,
        "D_mean_short": 2.0,
        "D_mean_long": 36.0,
        "pi_long": 0.5,
        "tau": 360.0,
    }
    model = build_model(2000, 2000)
    params = interpretable_to_params(user_params, 2000, 2000)
    params = calibrate(model, params)
    result = simulate(model, params, T=8, seed=0)
    print(f"Produced {len(result['adj_list'])} quarterly interval networks.")

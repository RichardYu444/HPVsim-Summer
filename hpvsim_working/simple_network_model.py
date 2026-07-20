#!/usr/bin/env python3
"""
simple_network_model
====================

A minimal temporal sexual-contact network simulator: a unipartite graph
whose nodes have power-law-distributed partner propensities, evolved under
STERGM-style separable formation and dissolution with a single relationship
duration and node turnover (births and Bernoulli deaths). An external hook
lets an epidemic engine feed back per-timestep deaths (e.g. HIV mortality).

This module is a *toy*: it is parameterised by a handful of directly
interpretable quantities and is meant to demonstrate the input/output
contract that a contact-network component exposes to a coupled disease
simulation. It produces exactly the snapshot and epidemic-input shapes the
epidemic engine and the coupled driver consume, so it can stand in wherever
that contract is needed.

Note, the model itself distinguishes two types of edges and nodes (nodes_u, edges_u vs
nodes v, edges v). They are the same in this toy model. In the more general
model they are meant to represent links connecting men (u) to women (v).

I kept the whole structure as it is currently in my model so it is easy to
implement.

The core function is network_step, within simulate. All the other functions
are utilities to make the code run.

Interpretable inputs
--------------------
The survey-level statistics passed to ``interpretable_to_params`` are:

* ``mean_partners_per_year`` — expected number of distinct partners a node
  accumulates over a 12-month window.
* ``exponent`` and ``kappa`` — the shape of the partner-propensity
  distribution ``p(x) ∝ x^{-exponent} · exp(-x/kappa)``, which sets how
  heterogeneous nodes are in their number of partners (the degree
  distribution).
* ``D_mean`` — the average duration of a relationship, in months.

Optional knobs (sensible defaults supplied):

* ``tau`` — mean time a node stays in the population, in months.
* ``epsilon`` — net annual population growth rate.
* ``xmin`` — lower support bound of the propensity distribution.

Public API
----------
``interpretable_to_params(user_params, nU, comm_sizes_U, *, bipartite=False)``
    Convert the interpretable statistics above into the runtime parameters.

``build_model(nU, comm_sizes_U, params, *, seed=0)``
    Build the static structural descriptor.

``init_network_state(model, params, rng)``
    Allocate the t=0 state: sample propensities and the initial edges.

``build_snapshot(state, model)``
    Freeze the current state into the snapshot dict the epidemic engine
    consumes.

``network_step(state, model, params, rng, t, *, extra_deaths_U=None, …)``
    One monthly update. Order: turnover (natural deaths + any externally
    supplied deaths, then births) → edge dissolution → edge formation.
    Externally supplied deaths (``extra_deaths_U``) are removed without
    replacement, so the live population can shrink under epidemic pressure.

``simulate(model, params, T, …)``
    Drive ``network_step`` for ``T`` months and return the raw outputs plus
    the epidemic-model inputs (``adj_list``, ``active_nodes_list``,
    ``node_types_list``, ``entry_kind_list``, ``change_times``).

``calibrate_rho(model, params, …)``
    Provided for API symmetry; returns the parameters unchanged.

Conventions
-----------
* All UIDs are globally unique monotone integers and are never reused after
  a node dies.
* Edges are stored canonically as ``(min, max)`` UID pairs in the parallel
  ``edges_u`` / ``edges_v`` arrays; all nodes are a single type (0).

Sparse approximation
--------------------
Edge presence follows ``p_ij ≈ ρ · θ_i · θ_j`` (valid when ρ ≪ 1). Per step,
the number of new ties is a single ``Poisson`` draw and the two endpoints of
each tie are drawn proportionally to node propensity ``θ``.
"""

import os

import numpy as np
from scipy.integrate import quad
from scipy.sparse import csr_matrix

# Entry-kind codes. They tag how each node
# entered the population so the epidemic engine can decide who may arrive
# infected: natural births always enter susceptible; only migrants may be
# infected "imports".
ENTRY_INIT      = 0   # present at t=0
ENTRY_BIRTH     = 1   # natural birth (rate delta)
ENTRY_MIGRATION = 2   # migrant (rate epsilon)

# One observation year, used to relate the yearly distinct-partner count to
# the momentary (snapshot) number of partners a node holds at any instant.
_OBS_WINDOW_MONTHS = 12.0

# Canonical-key base for vectorised edge de-duplication: an edge (a, b) with
# a < b maps to a*KEY + b, unique as long as b < KEY.
_KEY = np.int64(1) << 33


# =====================================================================
# Propensity distribution helpers
# =====================================================================

def sample_powerlaw_cutoff(n, alpha, kappa, xmin=1.0, rng=None):
    """
    Sample n values from p(x) ∝ x^{-alpha} · exp(-x/kappa), x ≥ xmin.

    Parameters
    ----------
    alpha : float > 1   power-law exponent (controls tail heaviness)
    kappa : float > 0   exponential cutoff scale (soft upper bound on
                        partner propensity)
    xmin  : float > 0   lower support bound (default 1.0)

    Algorithm
    ---------
    Rejection sampling with a Pareto(alpha, xmin) envelope; acceptance
    probability exp(-(x - xmin)/kappa) ∈ (0, 1].
    """
    if alpha <= 1.0:
        raise ValueError("alpha must be > 1")
    if kappa <= 0:
        raise ValueError("kappa must be > 0")
    if rng is None:
        rng = np.random.default_rng()
    out = np.empty(n)
    filled = 0
    while filled < n:
        batch = max((n - filled) * 3, 512)
        u = rng.random(batch)
        x = xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))
        accept = rng.random(batch) < np.exp(-(x - xmin) / kappa)
        accepted = x[accept]
        take = min(len(accepted), n - filled)
        out[filled:filled + take] = accepted[:take]
        filled += take
    return out


def powerlaw_cutoff_mean(alpha, kappa, xmin=1.0):
    """
    Analytical mean of the cutoff power-law via numerical quadrature.

      E[X] = ∫ x · x^{-alpha} e^{-x/kappa} dx  /  ∫ x^{-alpha} e^{-x/kappa} dx

    over [xmin, ∞). Called once at setup; cost is negligible.
    """
    kernel = lambda x: x ** (-alpha) * np.exp(-x / kappa)
    denom, _ = quad(kernel, xmin, np.inf)
    numer, _ = quad(lambda x: x * kernel(x), xmin, np.inf)
    if denom <= 0:
        raise ValueError(
            f"powerlaw_cutoff_mean: zero denominator "
            f"(alpha={alpha}, kappa={kappa}, xmin={xmin})")
    return numer / denom


def _validate_population_size(comm_sizes_U, nU, name="comm_sizes_U"):
    """Check the supplied sizes sum to nU and return nU as a plain int.

    The population is treated as a single homogeneous group; ``comm_sizes_U``
    is accepted (and its total validated) so call sites can pass either a bare
    size or a list of sizes that sum to the total.
    """
    cs = np.atleast_1d(np.asarray(comm_sizes_U, dtype=np.int64))
    if cs.size == 0 or cs.sum() != int(nU):
        raise ValueError(
            f"{name} must sum to nU={int(nU)} (got sum={int(cs.sum())}).")
    return int(nU)


def _lookup_sorted_positions(sorted_ids, query_ids):
    """Vectorised exact lookup of query_ids in a sorted integer ID array."""
    sorted_ids = np.asarray(sorted_ids, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if query_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    pos = np.searchsorted(sorted_ids, query_ids)
    assert np.all(pos < sorted_ids.size) and np.all(sorted_ids[pos] == query_ids), \
        "_lookup_sorted_positions: UID not found — edge referencing unknown node"
    return pos


# =====================================================================
# Snapshot -> epidemic-format conversion
# =====================================================================

def _build_epidemic_snapshot(edges_u, edges_v, active_U):
    """
    Convert one network snapshot to epidemic-model format.

    Returns ``(adj, node_ids, node_types)`` with ``adj`` a symmetric binary
    CSR adjacency over the active nodes (sorted ascending by UID) and all
    nodes a single type (0).
    """
    node_ids = np.fromiter(active_U.keys(), dtype=np.int64, count=len(active_U))
    node_ids.sort()
    node_types = np.zeros(len(node_ids), dtype=np.int8)

    n = len(node_ids)
    if n == 0:
        return csr_matrix((0, 0), dtype=np.int8), node_ids, node_types

    eu = np.asarray(edges_u, dtype=np.int64)
    ev = np.asarray(edges_v, dtype=np.int64)
    if eu.size == 0:
        return csr_matrix((n, n), dtype=np.int8), node_ids, node_types

    rows = _lookup_sorted_positions(node_ids, eu)
    cols = _lookup_sorted_positions(node_ids, ev)
    data = np.ones(2 * eu.size, dtype=np.int8)
    adj = csr_matrix((data, (np.r_[rows, cols], np.r_[cols, rows])),
                     shape=(n, n), dtype=np.int8)
    adj.sum_duplicates()
    if adj.nnz:
        adj.data[:] = 1
    return adj, node_ids, node_types


def _epi_snapshot_tuple(snap):
    """Convert one raw snapshot to the epidemic-input tuple.

    Returns ``(adj, active_nodes, node_types, entry_kind)`` with ``entry_kind``
    aligned row-for-row with ``active_nodes``.
    """
    adj, active_nodes, node_types = _build_epidemic_snapshot(
        edges_u=snap["edges_u"], edges_v=snap["edges_v"],
        active_U=snap["active_U"])
    au = snap["active_U"]
    entry_kind = np.fromiter(
        (au[int(u)].get("entry", ENTRY_INIT) for u in active_nodes),
        dtype=np.int8, count=active_nodes.size)
    return adj, active_nodes, node_types, entry_kind


def _build_epidemic_temporal_inputs(raw_snapshots, change_times):
    """Build epidemic-model inputs from a sequence of snapshots."""
    adj_list, active_nodes_list, node_types_list, entry_kind_list = [], [], [], []
    for snap in raw_snapshots:
        adj, active_nodes, node_types, entry_kind = _epi_snapshot_tuple(snap)
        adj_list.append(adj)
        active_nodes_list.append(active_nodes)
        node_types_list.append(node_types)
        entry_kind_list.append(entry_kind)
    return {
        "adj_list": adj_list,
        "active_nodes_list": active_nodes_list,
        "node_types_list": node_types_list,
        "entry_kind_list": entry_kind_list,
        "change_times": np.asarray(change_times, dtype=float),
    }


# =====================================================================
# Parameter conversion: interpretable statistics -> runtime parameters
# =====================================================================

def interpretable_to_params(user_params, nU, comm_sizes_U, *,
                            nV=None, comm_sizes_V=None,
                            bipartite=False, bridge_fraction=0.0):
    """
    Convert interpretable survey statistics into runtime parameters.

    Parameters
    ----------
    user_params : dict
        Interpretable statistics. Recognised keys:

        ``mean_partners_per_year`` (or ``k_mean``) : float
            Expected distinct partners over a 12-month window.
        ``exponent`` (or ``exponent_U``) : float > 1
            Power-law exponent of the partner-propensity distribution.
        ``kappa`` (or ``kappa_U``) : float > 0
            Exponential cutoff of the propensity distribution.
        ``D_mean`` : float >= 1
            Mean relationship duration, in months.
        ``tau`` : float, optional
            Mean time a node remains in the population, in months
            (default 360).
        ``epsilon`` : float, optional
            Net annual population growth rate (default 0).
        ``xmin`` : float, optional
            Lower support bound of the propensity distribution (default 1).
    nU : int
        Number of nodes at t=0.
    comm_sizes_U : array-like of int
        Must sum to nU.

    Returns
    -------
    params : dict
        Runtime parameters consumed by ``build_model`` and ``simulate``.
    """
    if bipartite:
        raise NotImplementedError(
            "simple_network_model supports the unipartite population only; "
            "call with bipartite=False.")
    if bridge_fraction != 0.0:
        raise NotImplementedError("bridge_fraction must be 0.0.")

    N = _validate_population_size(comm_sizes_U, nU)

    alpha = float(user_params.get("exponent", user_params.get("exponent_U")))
    kappa = float(user_params.get("kappa", user_params.get("kappa_U")))
    xmin = float(user_params.get("xmin", 1.0))

    D_mean = float(user_params["D_mean"])
    if D_mean < 1:
        raise ValueError("D_mean must be >= 1 timestep")
    q = 1.0 / D_mean

    if "mean_partners_per_year" in user_params:
        k_year = float(user_params["mean_partners_per_year"])
    else:
        k_year = float(user_params["k_mean"])
    if k_year <= 0:
        raise ValueError("mean_partners_per_year must be > 0")

    # Distinct partners observed over a year = the momentary partners a node
    # holds plus the ones formed and dissolved within the window. With ties
    # ending at rate q, this gives k_year = k_snapshot · (1 + q · W). Invert to
    # recover the momentary mean degree the network actually carries.
    scale = 1.0 + _OBS_WINDOW_MONTHS * q
    k_snap = k_year / scale

    # Edge-formation sparsity: with propensities normalised to mean 1, the
    # expected momentary mean degree equals N · rho, so rho = k_snap / N.
    rho = k_snap / N

    tau = float(user_params.get("tau", 360.0))
    if tau <= 0:
        raise ValueError("tau must be > 0")
    delta = 1.0 / tau

    epsilon_annual = float(user_params.get("epsilon", 0.0))
    epsilon_monthly = epsilon_annual / 12.0

    mu_theta = powerlaw_cutoff_mean(alpha, kappa, xmin)

    w = 60
    print("─" * w)
    print("Interpretable → runtime parameters  [unipartite toy network]")
    print("─" * w)
    print(f"  N={int(N)}")
    print(f"  partners/yr = {k_year}  →  momentary mean degree = {k_snap:.4f}")
    print(f"    (scale = 1 + {int(_OBS_WINDOW_MONTHS)}·q = {scale:.4f})")
    print(f"  α={alpha}  κ={kappa}  →  E[θ]={mu_theta:.4f}")
    print(f"  D_mean={D_mean} mo  →  dissolution rate q={q:.6f}")
    print(f"  τ={tau} mo  →  death rate δ={delta:.6f}")
    if epsilon_annual != 0.0:
        print(f"  ε={epsilon_annual}/yr  →  ε_monthly={epsilon_monthly:.6f}")
    print(f"  ρ={rho:.6f}  (should be ≪ 1)")
    print("─" * w)

    return {
        "rho":             rho,
        "N_init":          int(N),
        "omega":           np.array([[1.0]], dtype=float),
        "exponent":        alpha,
        "kappa":           kappa,
        "xmin":            xmin,
        "q":               q,
        "delta":           delta,
        "epsilon_monthly": epsilon_monthly,
        "D_mean":          D_mean,
        "tau":             tau,
        "mean_partners_per_year": k_year,
        "k_snap":          k_snap,
    }


# =====================================================================
# Model descriptor
# =====================================================================

def build_model(nU, comm_sizes_U, params, *,
                nV=None, comm_sizes_V=None,
                bipartite=False, seed=0,
                theta_normalization="per_block",
                block_assigner=None,
                block_assigner_V=None,
                age_config=None):
    """
    Build the static structural descriptor of the network model.

    Returns a dict with a ``bipartite`` key (always False here) on which the
    runtime entry points dispatch. ``seed`` is accepted for API symmetry; the
    descriptor draws no randomness.
    """
    if bipartite:
        raise NotImplementedError(
            "simple_network_model supports the unipartite population only; "
            "call with bipartite=False.")

    N = _validate_population_size(comm_sizes_U, nU)
    theta_mean_ref = powerlaw_cutoff_mean(
        params["exponent"], params["kappa"], params["xmin"])

    return {
        "N_init":            int(N),
        "nU_init":           int(N),
        "K":                 1,
        "comm_sizes":        np.array([N], dtype=np.int64),
        "comm_sizes_U":      np.array([N], dtype=np.int64),
        "frac":              np.array([1.0], dtype=float),
        "theta_mean_ref":    float(theta_mean_ref),
        "theta_normalization": theta_normalization,
        "bipartite":         False,
    }


# =====================================================================
# State initialisation
# =====================================================================

def init_network_state(model, params, rng):
    """Build the t=0 network state: sample propensities and initial edges."""
    N = model["N_init"]
    xmin = params["xmin"]
    theta_mean_ref = model["theta_mean_ref"]
    theta_norm = model.get("theta_normalization", "per_block")

    theta_raw = sample_powerlaw_cutoff(
        N, params["exponent"], params["kappa"], xmin=xmin, rng=rng)
    # Normalise propensities so their population mean is 1; the overall degree
    # level is then carried entirely by rho.
    if theta_norm == "per_block":
        theta = theta_raw / theta_raw.mean()
    else:
        theta = theta_raw / theta_mean_ref

    state = {
        "u_uid":      np.arange(N, dtype=np.int64),
        "u_theta":    np.ascontiguousarray(theta, dtype=float),
        "u_entry":    np.full(N, ENTRY_INIT, dtype=np.int8),
        "edges_u":    np.empty(0, dtype=np.int64),
        "edges_v":    np.empty(0, dtype=np.int64),
        "edges_type": np.empty(0, dtype=np.int8),
        "edge_birth": np.empty(0, dtype=np.int64),
        "next_uid":   int(N),
    }

    new_u, new_v, new_t = _sample_edges(params, state, scale=params["rho"], rng=rng)
    state["edges_u"] = new_u
    state["edges_v"] = new_v
    state["edges_type"] = new_t
    # Tie-birth times, aligned positionally with the edge arrays; initial ties
    # are born at t=0. Maintained at every edge mutation for duration tracking.
    state["edge_birth"] = np.zeros(new_u.size, dtype=np.int64)
    return state


# =====================================================================
# Edge sampler
# =====================================================================

def _sample_edges(params, state, scale, rng):
    """Sample new unique edges from the sparse propensity model.

    The number of candidate ties is ``Poisson(scale · (Σθ)² / 2)`` (the
    diagonal halved so each unordered pair is counted once); both endpoints of
    each candidate are drawn proportionally to node propensity ``θ``. Self
    pairs, within-batch duplicates, and ties that already exist are dropped.
    """
    theta = state["u_theta"]
    uid = state["u_uid"]
    n = uid.size
    empty = (np.empty(0, dtype=np.int64),
             np.empty(0, dtype=np.int64),
             np.empty(0, dtype=np.int8))
    if n < 2 or scale <= 0.0:
        return empty

    S = float(theta.sum())
    if S <= 0.0:
        return empty
    lam = scale * S * S / 2.0
    m = int(rng.poisson(lam))
    if m == 0:
        return empty

    cdf = np.cumsum(theta) / S
    a = uid[np.clip(np.searchsorted(cdf, rng.random(m), side="right"), 0, n - 1)]
    b = uid[np.clip(np.searchsorted(cdf, rng.random(m), side="right"), 0, n - 1)]

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        return empty

    cand_keys = lo * _KEY + hi
    cand_keys, idx = np.unique(cand_keys, return_index=True)
    lo, hi = lo[idx], hi[idx]

    if state["edges_u"].size:
        existing = state["edges_u"] * _KEY + state["edges_v"]
        fresh = ~np.isin(cand_keys, existing)
        lo, hi = lo[fresh], hi[fresh]

    return (lo.astype(np.int64), hi.astype(np.int64),
            np.zeros(lo.size, dtype=np.int8))


# =====================================================================
# Turnover: deaths then births
# =====================================================================

def _apply_turnover(model, params, state, rng, t,
                    node_events, track_durations, durations,
                    *, extra_deaths=None, edge_events=None):
    """
    Apply one round of node deaths then births.

    Natural deaths (Bernoulli at rate ``delta``) and any externally supplied
    deaths (``extra_deaths``, e.g. HIV mortality from the epidemic engine) are
    removed together with their incident edges. Births replace natural exits
    and add ``epsilon`` growth; externally supplied deaths are NOT replaced, so
    the live population can shrink. ``node_events`` records natural deaths
    before HIV deaths, then births, each as ``(t, uid, "U", kind)``.
    """
    delta = params["delta"]
    eps_mo = params.get("epsilon_monthly", 0.0)
    t_ref = model["theta_mean_ref"]
    xmin = params["xmin"]

    u_uid = state["u_uid"]
    n_active_before = int(u_uid.size)

    # --- natural deaths (one Bernoulli draw per active node) ---
    if delta > 0.0 and n_active_before:
        dead_mask = rng.random(n_active_before) < delta
        dead_natural = u_uid[dead_mask]
    else:
        dead_natural = np.empty(0, dtype=np.int64)

    # --- externally supplied deaths (no RNG; keep only active, drop overlap) ---
    extra = (np.asarray(extra_deaths, dtype=np.int64)
             if extra_deaths is not None and len(extra_deaths) > 0
             else np.empty(0, dtype=np.int64))
    if extra.size:
        pos = np.searchsorted(u_uid, extra)
        in_range = pos < u_uid.size
        valid = np.zeros(extra.size, dtype=bool)
        valid[in_range] = u_uid[pos[in_range]] == extra[in_range]
        dead_hiv = extra[valid]
        if dead_natural.size and dead_hiv.size:
            dead_hiv = np.setdiff1d(dead_hiv, dead_natural, assume_unique=False)
    else:
        dead_hiv = np.empty(0, dtype=np.int64)

    if dead_natural.size and dead_hiv.size:
        dead_all = np.concatenate([dead_natural, dead_hiv])
    elif dead_natural.size:
        dead_all = dead_natural
    else:
        dead_all = dead_hiv

    removed_eu = np.empty(0, dtype=np.int64)
    removed_ev = np.empty(0, dtype=np.int64)

    if dead_all.size:
        eu, ev = state["edges_u"], state["edges_v"]
        if eu.size:
            incident = np.isin(eu, dead_all) | np.isin(ev, dead_all)
            alive_edge = ~incident
            removed_eu = eu[incident]
            removed_ev = ev[incident]
            eb = state["edge_birth"]
            if track_durations and removed_eu.size:
                durations.extend((t - eb[incident]).tolist())
            state["edges_u"]    = eu[alive_edge]
            state["edges_v"]    = ev[alive_edge]
            state["edges_type"] = state["edges_type"][alive_edge]
            state["edge_birth"] = eb[alive_edge]
        alive_node = ~np.isin(state["u_uid"], dead_all)
        state["u_uid"]   = state["u_uid"][alive_node]
        state["u_theta"] = state["u_theta"][alive_node]
        state["u_entry"] = state["u_entry"][alive_node]
        for uid in dead_natural.tolist():
            node_events.append((t, uid, "U", "death"))
        for uid in dead_hiv.tolist():
            node_events.append((t, uid, "U", "hiv_death"))

    # --- births: replace natural exits + epsilon growth ---
    n_born = int(rng.poisson((delta + eps_mo) * n_active_before))
    born_uids = np.empty(0, dtype=np.int64)
    if n_born > 0:
        start_uid = state["next_uid"]
        born_uids = np.arange(start_uid, start_uid + n_born, dtype=np.int64)
        state["next_uid"] += n_born
        thetas = sample_powerlaw_cutoff(
            n_born, params["exponent"], params["kappa"], xmin=xmin, rng=rng) / t_ref
        p_natural = float(delta) / max(float(delta) + float(eps_mo), 1e-12)
        is_natural = rng.random(n_born) < p_natural
        entry_codes = np.where(is_natural, ENTRY_BIRTH, ENTRY_MIGRATION).astype(np.int8)

        state["u_uid"]   = np.concatenate([state["u_uid"], born_uids])
        state["u_theta"] = np.concatenate(
            [state["u_theta"], np.asarray(thetas, dtype=float)])
        state["u_entry"] = np.concatenate([state["u_entry"], entry_codes])
        ev_kind = np.where(entry_codes == ENTRY_BIRTH, "birth", "migration")
        for uid, kind in zip(born_uids.tolist(), ev_kind.tolist()):
            node_events.append((t, uid, "U", kind))

    if edge_events is not None:
        edge_events["removed_edges_turnover"] = (removed_eu, removed_ev)
        edge_events["removed_nodes_natural"] = dead_natural
        edge_events["removed_nodes_hiv"] = dead_hiv
        edge_events["added_nodes"] = born_uids


# =====================================================================
# Snapshot
# =====================================================================

def build_snapshot(state, model):
    """
    Return a raw snapshot of the current network state for the epidemic model.

    The returned dict has the exact shape the epidemic engine and coupled
    driver consume: copies of the canonical edge arrays plus an ``active_U``
    mapping ``{uid: {"theta", "block", "entry"}}``. Arrays and per-node dicts
    are copies, so later mutation of the live state cannot rewrite this
    snapshot.
    """
    uid = state["u_uid"]
    theta = state["u_theta"]
    entry = state["u_entry"]
    active_U = {int(uid[i]): {"theta": float(theta[i]),
                              "block": 0,
                              "entry": int(entry[i])}
                for i in range(uid.size)}
    return {
        "edges_u":    state["edges_u"].copy(),
        "edges_v":    state["edges_v"].copy(),
        "edges_type": state["edges_type"].copy(),
        "active_U":   active_U,
    }


# =====================================================================
# One timestep
# =====================================================================

def network_step(state, model, params, rng, t, *,
                 extra_deaths_U=None, extra_deaths_V=None,
                 track_durations=False,
                 durations=None, node_events=None):
    """
    Advance the network state by one monthly timestep in place.

    Order: turnover (natural deaths + ``extra_deaths_U``, then births) → edge
    dissolution → edge formation.

    Parameters
    ----------
    extra_deaths_U : array-like of int UIDs, optional
        Deaths supplied by the epidemic engine (e.g. HIV mortality), removed
        alongside natural turnover and without triggering replacement births.
    extra_deaths_V : must be empty
        Accepted for API symmetry; this model has a single node namespace.
    track_durations, durations, node_events
        Optional accumulators, modified in place; when None, tracking is off.

    Returns
    -------
    edge_events : dict
        ``removed_edges_turnover`` / ``removed_edges_dissolution`` :
            ``(eu_del, ev_del)`` — ties lost to deaths vs. to dissolution.
        ``added_edges`` : ``(eu_new, ev_new)`` and ``added_edges_type``.
        ``removed_nodes_natural`` / ``removed_nodes_hiv`` / ``added_nodes`` :
            UID arrays. All arrays are in global UID space.
    """
    if extra_deaths_V is not None and len(extra_deaths_V) > 0:
        raise ValueError(
            "network_step: extra_deaths_V is non-empty but the model has one "
            "namespace only. Pass external deaths via extra_deaths_U.")

    q = params["q"]
    delta = params["delta"]
    N_init = params.get("N_init", model["N_init"])

    if node_events is None:
        node_events = []
    if durations is None:
        durations = []

    edge_events = {
        "removed_edges_turnover":    (np.empty(0, dtype=np.int64),
                                      np.empty(0, dtype=np.int64)),
        "removed_edges_dissolution": (np.empty(0, dtype=np.int64),
                                      np.empty(0, dtype=np.int64)),
        "added_edges":               (np.empty(0, dtype=np.int64),
                                      np.empty(0, dtype=np.int64)),
        "added_edges_type":          np.empty(0, dtype=np.int8),
        "removed_nodes_natural":     np.empty(0, dtype=np.int64),
        "removed_nodes_hiv":         np.empty(0, dtype=np.int64),
        "added_nodes":               np.empty(0, dtype=np.int64),
    }

    # --- 1. turnover (natural + external deaths, then births) ---
    extra_size = (0 if extra_deaths_U is None else len(extra_deaths_U))
    if delta > 0 or extra_size > 0:
        _apply_turnover(
            model, params, state, rng, t,
            node_events, track_durations, durations,
            extra_deaths=extra_deaths_U, edge_events=edge_events)

    # --- 2. edge dissolution (each tie ends independently with prob q) ---
    eu = state["edges_u"]
    ev = state["edges_v"]
    if eu.size > 0:
        keep = rng.random(eu.size) >= q
        removed_eu = eu[~keep]
        removed_ev = ev[~keep]
        eb = state["edge_birth"]
        if track_durations and removed_eu.size:
            durations.extend((t - eb[~keep]).tolist())
        state["edges_u"]    = eu[keep]
        state["edges_v"]    = ev[keep]
        state["edges_type"] = state["edges_type"][keep]
        state["edge_birth"] = eb[keep]
        edge_events["removed_edges_dissolution"] = (removed_eu, removed_ev)

    # --- 3. edge formation ---
    # Hold the per-capita degree steady as the population drifts: scaling rho by
    # N_init / N_now keeps the expected number of ties proportional to N_now.
    N_now = int(state["u_uid"].size)
    eff_rho = params["rho"] * N_init / max(N_now, 1)
    nu, nv, nt = _sample_edges(params, state, scale=q * eff_rho, rng=rng)
    if nu.size:
        state["edges_u"]    = np.concatenate([state["edges_u"], nu])
        state["edges_v"]    = np.concatenate([state["edges_v"], nv])
        state["edges_type"] = np.concatenate([state["edges_type"], nt])
        state["edge_birth"] = np.concatenate(
            [state["edge_birth"], np.full(nu.size, t, dtype=np.int64)])
        edge_events["added_edges"] = (nu, nv)
        edge_events["added_edges_type"] = nt

    return edge_events


# =====================================================================
# Main simulation loop
# =====================================================================

def _persist_snapshot(path, edges_u, edges_v, active_ids, t):
    """Save one edge snapshot to an .npz file."""
    np.savez_compressed(
        path,
        edges_u=np.asarray(edges_u, dtype=np.int64),
        edges_v=np.asarray(edges_v, dtype=np.int64),
        active=np.asarray(active_ids, dtype=np.int64),
        t=np.int64(t))


def simulate(model, params, T, *,
             seed=1,
             save_edge_snapshots=(),
             max_snapshots=50,
             track_durations=True,
             base_dir=None,
             return_raw_snapshots=True,
             **kwargs):
    """
    Run the temporal network simulation for ``T`` monthly timesteps.

    Returns a dict carrying the final state, the raw per-step snapshots (when
    requested), and the epidemic-model inputs (``adj_list``,
    ``active_nodes_list``, ``node_types_list``, ``entry_kind_list``,
    ``change_times``). ``epi_v_offset`` is 0 (a single node namespace).
    """
    if params["q"] <= 0 or params["q"] >= 1:
        raise ValueError("q must be in (0, 1) (equivalently D_mean > 1)")

    if base_dir is None:
        base_dir = os.path.expanduser("~/Downloads/simple_network")
    os.makedirs(base_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    state = init_network_state(model, params, rng)

    node_events = []
    durations = [] if track_durations else None

    snap_times = sorted({int(s) for s in save_edge_snapshots
                         if 0 <= int(s) <= T})[:max_snapshots]
    snap_times_set = set(snap_times)
    saved_paths = []

    def _save(t_snap):
        path = os.path.join(base_dir, f"edges_t{t_snap}.npz")
        _persist_snapshot(path, state["edges_u"], state["edges_v"],
                          state["u_uid"], t_snap)
        saved_paths.append(path)

    if 0 in snap_times_set:
        _save(0)

    keep_raw = bool(return_raw_snapshots)
    raw_snapshots = [] if keep_raw else None
    adj_list, active_nodes_list, node_types_list, entry_kind_list = [], [], [], []

    def _ingest(snap):
        adj, an, nt, ek = _epi_snapshot_tuple(snap)
        adj_list.append(adj)
        active_nodes_list.append(an)
        node_types_list.append(nt)
        entry_kind_list.append(ek)
        if keep_raw:
            raw_snapshots.append(snap)

    if T > 0:
        _ingest(build_snapshot(state, model))

    for t in range(1, T + 1):
        network_step(
            state, model, params, rng, t,
            extra_deaths_U=None,
            track_durations=track_durations,
            durations=durations, node_events=node_events)
        if t in snap_times_set:
            _save(t)
        if t < T:
            _ingest(build_snapshot(state, model))

    censored_ages = ((T - state["edge_birth"]).tolist()
                     if track_durations else None)

    return {
        "T":                 T,
        "state":             state,
        "raw_snapshots":     raw_snapshots,
        "node_events":       node_events,
        "durations":         durations,
        "censored_ages":     censored_ages,
        "snapshots":         saved_paths,
        "base_dir":          base_dir,
        "adj_list":          adj_list,
        "active_nodes_list": active_nodes_list,
        "node_types_list":   node_types_list,
        "entry_kind_list":   entry_kind_list,
        "change_times":      np.arange(T + 1, dtype=float),
        "epi_v_offset":      0,
    }


# =====================================================================
# Calibration (API symmetry)
# =====================================================================

def calibrate_rho(model, params, *,
                  T_burn=50, max_iters=3, tol=0.02,
                  rng_seed=42, verbose=False):
    """Return the parameters unchanged.

    The momentary mean degree is fixed analytically by ``rho`` at parameter
    conversion, so no empirical refinement is needed. Provided so callers can
    invoke it uniformly.
    """
    refined = dict(params)
    refined["rho_calibration_info"] = {"converged": True, "iters": 0,
                                       "msg": "no-op"}
    return refined


__all__ = [
    "interpretable_to_params",
    "build_model",
    "init_network_state",
    "build_snapshot",
    "network_step",
    "simulate",
    "calibrate_rho",
    "sample_powerlaw_cutoff",
    "powerlaw_cutoff_mean",
    "ENTRY_INIT",
    "ENTRY_BIRTH",
    "ENTRY_MIGRATION",
]

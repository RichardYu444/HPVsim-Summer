#!/usr/bin/env python3
"""
Vendored, unmodified, from "Francesco simple codes/age_community_bipartite_network_model.py".
Treat that path as the upstream reference copy; edit this file only to pull in upstream
changes, not to add HPVsim-specific logic (that belongs in community_network.py).

age_community_bipartite_network_model
=====================================

A bipartite temporal sexual-contact network with **age mixing**, **community
mixing**, and **fully external demography**. It is the successor of
``simple_bipartite_network_model.py`` and keeps the same input/output contract
a coupled disease simulation expects, so it plugs into the epidemic driver
wherever a contact network is expected — but with three deliberate changes.

What changed relative to the simple bipartite toy
-------------------------------------------------
1. **No natural demography.** ``tau`` / ``delta`` and one-for-one birth
   replacement are gone. The population changes *only* through externally
   supplied deaths and births. Nothing dies or is born unless the caller says
   so. (No migration either — same closed-except-for-external-turnover idea, but
   now the turnover itself is external.)

2. **Age mixing (banded).** Every individual carries a **static age**. Ageing is
   NOT simulated here — it is assumed handled outside and fed in; the network
   only decides *who partners whom* given the ages it is handed. Age enters
   through a **band × band age-mixing matrix** ``A``: an age is binned into one
   of ``n_bands`` bands only to look up ``A[band_man, band_woman]``, which
   multiplies the propensity of a candidate man–woman tie.

3. **Four communities + community mixing.** Every individual belongs to one of
   ``n_communities`` (default 4) communities. A **community × community mixing
   matrix** ``C`` multiplies the propensity of a candidate tie by
   ``C[comm_man, comm_woman]``, so partnerships are (by default) mostly
   within-community.

Everything else is retained: Gamma "partner propensity" heterogeneity, the two
partnership types (short / long) with their own durations, the sparse
degree-corrected sampler, empirical ``calibrate``, and the quarterly integrated
snapshot. The epidemic output additionally exposes each active node's **age**
(``node_ages_list``) and **community** (``node_comm_list``).

The model as an ERGM
--------------------
Cross-sectionally this is a **degree-corrected stochastic block model** (dcSBM):
the per-dyad edge probability is

    p_uv  ≈  scale · θ_u · φ_v · C[c_u, c_v] · A[b_u, b_v],

i.e. the sparse dyad-independent ERGM of the simple toy with two extra
*dyad-independent* factors (community and age-band main effects / mixing). Being
dyad-independent, it is sampled directly with no MCMC.

Efficient sampling (why it stays O(#edges))
-------------------------------------------
The mixing factors are constant within a *block pair*, where a block is a
(community, age-band) stratum: for a man in block ``g`` and a woman in block
``h`` the weight ``W[g,h] = C[c_g,c_h]·A[b_g,b_h]`` is the same for every dyad in
that block pair. So we (i) draw the total number of candidate ties as one
Poisson (identical to the simple toy — the mean degree is therefore unchanged by
the mixing structure), (ii) split them across block pairs with a single
multinomial using probabilities ∝ ``W[g,h]·S_U^g·S_V^h`` (block propensity
sums), and (iii) draw each endpoint ∝ propensity *within* its block. This is the
Poisson–multinomial thinning of the dcSBM; no rejection sampling, no dense
matrix, cost ~ ``O(n_blocks² + #candidates)``.

Because the *total* candidate count is drawn exactly as in the single-block toy,
adding age/community mixing reshapes *where* edges go without changing the
overall mean degree — the interpretable inputs keep their meaning and
``calibrate`` still applies.

Demography API (external only)
------------------------------
* Deaths: pass UID arrays ``extra_deaths_U`` / ``extra_deaths_V`` to
  ``network_step`` (or ``..._by_month`` dicts to ``simulate``). Dead nodes and
  their incident edges are removed; they are NOT replaced.
* Births / entries: pass ``births_U`` / ``births_V`` dicts, each with explicit
  per-entrant ``"age"`` and ``"comm"`` arrays (and optional ``"theta"``). The
  model assigns fresh UIDs and places them. Entrants ALWAYS carry a
  caller-supplied age and community — ageing/entry demographics live outside.

Public API
----------
``interpretable_to_params(user_params, nU, nV=None)``   -> params
``build_model(nU, nV=None)``                            -> model
``init_network_state(model, params, rng)``              -> state
``network_step(state, model, params, rng, t, *, extra_deaths_U/V, births_U/V, ...)``
``build_snapshot(state, model, *, edges_u=None, edges_v=None, edges_type=None)``
``simulate(model, params, T, *, extra_deaths_*_by_month, extra_births_*_by_month, ...)``
``calibrate(model, params, ...)``                       -> params
"""

import os

import numpy as np
from scipy.sparse import csr_matrix

# Entry-kind codes: how a node entered the population.
ENTRY_INIT  = 0   # present at t=0
ENTRY_BIRTH = 1   # externally supplied entry (a.k.a. "birth")

# Edge-type codes.
EDGE_SHORT = 0    # casual tie, mean duration D_mean_short
EDGE_LONG  = 1    # steady tie, mean duration D_mean_long

# One observation year: relates the yearly distinct-partner count to the
# momentary number of partners a node holds at any instant.
_OBS_WINDOW_MONTHS = 12.0

# Internal monthly substeps per emitted output step: output unit of time = 3 mo.
MONTHS_PER_STEP = 3

# Canonical-key base for vectorised (U,V) edge de-duplication: (u, v) -> u*KEY+v.
_KEY = np.int64(1) << 33

# ---- Age / community structural defaults --------------------------------
DEFAULT_N_COMMUNITIES = 4
# Internal band boundaries (upper-exclusive). Natsal-like 6 bands over 16-74:
#   [16,25) [25,35) [35,45) [45,55) [55,65) [65,74]
DEFAULT_AGE_BAND_EDGES = (25.0, 35.0, 45.0, 55.0, 65.0)
DEFAULT_AGE_RANGE = (16.0, 74.0)


# =====================================================================
# Propensity distribution helpers  (Gamma)
# =====================================================================

def sample_gamma(n, shape, scale=1.0, rng=None):
    """Sample ``n`` values from Gamma p(x) ∝ x^{shape-1} e^{-x/scale}."""
    if rng is None:
        rng = np.random.default_rng()
    if shape <= 0:
        raise ValueError("gamma shape must be > 0")
    if scale <= 0:
        raise ValueError("gamma scale must be > 0")
    return rng.gamma(shape, scale, size=n)


def gamma_mean(shape, scale=1.0):
    """Analytical mean of a Gamma(shape, scale) distribution = shape * scale."""
    return shape * scale


def _sample_side_theta(n, shape, rng, floor=0.0, exact_mean_one=False):
    """Draw n partner propensities with mean 1.

    ``floor`` in [0, 1) puts a hard lower bound on the propensity (as a fraction
    of the mean): ``theta = floor + (1 - floor) * Gamma_mean1``. This leaves
    **zero probability mass below** ``floor`` — i.e. it removes the near-zero
    "never active" tail of the plain Gamma — while keeping mean 1. The
    coefficient of variation shrinks to ``CV = (1 - floor) / sqrt(shape)``, so
    ``floor -> 1`` approaches an identical-propensity (Poisson-degree) population.
    ``floor = 0`` recovers the original Gamma.

    With ``exact_mean_one`` the Gamma part is renormalised to mean exactly 1, so
    the realised connectivity level is carried entirely by ``rho``.
    """
    if n <= 0:
        return np.empty(0, dtype=float)
    raw = sample_gamma(n, shape, scale=1.0 / shape, rng=rng)
    if exact_mean_one:
        mu = float(raw.mean())
        if mu > 0:
            raw = raw / mu
    if floor:
        raw = float(floor) + (1.0 - float(floor)) * raw
    return raw


# =====================================================================
# Age / community helpers
# =====================================================================

def _age_to_band(ages, band_edges):
    """Bin continuous ages into band indices 0..n_bands-1 (upper-exclusive edges)."""
    ages = np.asarray(ages, dtype=float)
    band = np.searchsorted(np.asarray(band_edges, dtype=float), ages, side="right")
    return band.astype(np.int64)


def _block_of(comm, band, n_bands):
    """Combined stratum id: block = community * n_bands + age_band."""
    return (np.asarray(comm, dtype=np.int64) * int(n_bands)
            + np.asarray(band, dtype=np.int64))


def _band_centers(band_edges, age_range):
    """Representative age at the centre of each band (for the default kernel)."""
    lo = np.array([age_range[0], *band_edges], dtype=float)
    hi = np.array([*band_edges, age_range[1]], dtype=float)
    return 0.5 * (lo + hi)


def default_age_mixing(band_edges, age_range, sigma=8.0, male_older=2.0):
    """Assortative age-mixing matrix A[band_man, band_woman].

    Built from a Gaussian on the age gap with a male-older offset: a man of age
    ``a`` prefers a woman of age ``a - male_older``, with spread ``sigma`` years.
    Relative weights only; the absolute scale is irrelevant (normalised away in
    the sampler).
    """
    c = _band_centers(band_edges, age_range)
    diff = c[:, None] - c[None, :] - float(male_older)   # man(row) - woman(col)
    A = np.exp(-0.5 * (diff / float(sigma)) ** 2)
    return A


def default_community_mixing(n_comm, off_diagonal=0.1):
    """Assortative community-mixing matrix C: 1 on the diagonal, small off-diagonal."""
    C = np.full((n_comm, n_comm), float(off_diagonal), dtype=float)
    np.fill_diagonal(C, 1.0)
    return C


def _build_W_block(C, A):
    """Combined block-weight matrix W[g,h] = C[c_g,c_h]*A[b_g,b_h] = kron(C, A).

    With block id g = comm*n_bands + band, the Kronecker product indexes exactly
    as required.
    """
    return np.kron(np.asarray(C, float), np.asarray(A, float))


# =====================================================================
# Snapshot -> epidemic-format conversion  (bipartite)
# =====================================================================

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


def _build_epidemic_snapshot(edges_u, edges_v, edges_type, active_U, active_V,
                             v_offset):
    """
    Convert one bipartite network snapshot to epidemic-model format.

    Returns ``(adj, adj_type, active_nodes, node_types)`` with ``active_nodes`` =
    [sorted U ids, sorted (v_offset + V ids)], ``node_types`` 0 for men / 1 for
    women, ``adj`` a symmetric binary CSR adjacency (U–U and V–V blocks empty),
    and ``adj_type`` the same sparsity with stored value = edge type + 1
    (1 = short, 2 = long).
    """
    u_ids = np.fromiter(active_U.keys(), dtype=np.int64, count=len(active_U))
    u_ids.sort()
    v_ids_raw = np.fromiter(active_V.keys(), dtype=np.int64, count=len(active_V))
    v_ids_raw.sort()
    v_ids = v_ids_raw + np.int64(v_offset)

    active_nodes = np.concatenate([u_ids, v_ids])
    node_types = np.concatenate([
        np.zeros(u_ids.size, dtype=np.int8),
        np.ones(v_ids.size, dtype=np.int8),
    ])

    n = active_nodes.size
    if n == 0:
        z = csr_matrix((0, 0), dtype=np.int8)
        return z, z, active_nodes, node_types

    eu = np.asarray(edges_u, dtype=np.int64)
    ev = np.asarray(edges_v, dtype=np.int64)
    et = np.asarray(edges_type, dtype=np.int8)
    if eu.size == 0:
        z = csr_matrix((n, n), dtype=np.int8)
        return z, z, active_nodes, node_types

    rows = _lookup_sorted_positions(u_ids, eu)
    cols = _lookup_sorted_positions(v_ids_raw, ev) + u_ids.size
    rr = np.r_[rows, cols]
    cc = np.r_[cols, rows]

    adj = csr_matrix((np.ones(2 * eu.size, dtype=np.int8), (rr, cc)),
                     shape=(n, n), dtype=np.int8)
    adj.sum_duplicates()
    if adj.nnz:
        adj.data[:] = 1

    type_vals = np.r_[et, et].astype(np.int8) + np.int8(1)
    adj_type = csr_matrix((type_vals, (rr, cc)), shape=(n, n), dtype=np.int8)

    return adj, adj_type, active_nodes, node_types


def _epi_snapshot_tuple(snap, v_offset):
    """Convert one raw snapshot to
    ``(adj, adj_type, active_nodes, node_types, entry_kind, node_ages, node_comm)``.

    The per-node attribute arrays are aligned row-for-row with ``active_nodes`` =
    [sorted U ids, sorted V ids].
    """
    adj, adj_type, active_nodes, node_types = _build_epidemic_snapshot(
        edges_u=snap["edges_u"], edges_v=snap["edges_v"],
        edges_type=snap["edges_type"],
        active_U=snap["active_U"], active_V=snap["active_V"],
        v_offset=v_offset)
    au, av = snap["active_U"], snap["active_V"]
    u_ids = sorted(au.keys())
    v_ids = sorted(av.keys())
    n = len(u_ids) + len(v_ids)
    entry_kind = np.empty(n, dtype=np.int8)
    node_ages = np.empty(n, dtype=float)
    node_comm = np.empty(n, dtype=np.int16)
    for i, u in enumerate(u_ids):
        info = au[u]
        entry_kind[i] = info.get("entry", ENTRY_INIT)
        node_ages[i] = info.get("age", np.nan)
        node_comm[i] = info.get("comm", -1)
    for j, v in enumerate(v_ids):
        info = av[v]
        k = len(u_ids) + j
        entry_kind[k] = info.get("entry", ENTRY_INIT)
        node_ages[k] = info.get("age", np.nan)
        node_comm[k] = info.get("comm", -1)
    return adj, adj_type, active_nodes, node_types, entry_kind, node_ages, node_comm


# =====================================================================
# Parameter conversion: interpretable statistics -> runtime parameters
# =====================================================================

def interpretable_to_params(user_params, nU, nV=None):
    """
    Convert interpretable survey statistics into runtime parameters.

    Recognised keys (besides the age/community block below) match the simple
    toy: ``mean_partners_per_year``, ``gamma_shape`` (+ per-side
    ``gamma_shape_U`` / ``gamma_shape_V``), ``D_mean`` (+ ``D_mean_short`` /
    ``D_mean_long``), ``pi_long`` (a.k.a. ``frac_long``). **No** ``tau`` — this
    model has no natural demography.

    Age / community keys (all optional; sensible defaults supplied):
        ``n_communities``      : number of communities (default 4).
        ``age_band_edges``     : upper-exclusive band boundaries.
        ``age_range``          : (min, max) supported age, for defaults/init.
        ``age_mixing``         : (n_bands x n_bands) matrix A[band_man,band_woman].
        ``community_mixing``   : (n_comm x n_comm) matrix C.
        ``age_sigma`` / ``age_male_older`` : shape the default age kernel.
        ``community_off_diag`` : off-diagonal weight of the default C.
        ``init_age_range``     : (min,max) for sampling INITIAL ages (default age_range).
        ``community_probs``    : community proportions for the INITIAL population.
    """
    if nV is None:
        nV = nU
    NU = int(nU)
    NV = int(nV)
    if NU <= 0 or NV <= 0:
        raise ValueError("nU and nV must be > 0")

    # ---- propensity heterogeneity ----
    shape_default = float(user_params.get("gamma_shape", 1.0))
    shape_U = float(user_params.get("gamma_shape_U", shape_default))
    shape_V = float(user_params.get("gamma_shape_V", shape_default))
    if shape_U <= 0 or shape_V <= 0:
        raise ValueError("gamma_shape (U and V) must be > 0")
    # propensity floor: theta = floor + (1-floor)*Gamma  (zero mass below floor)
    theta_floor = float(user_params.get("theta_floor", 0.0))
    if not (0.0 <= theta_floor < 1.0):
        raise ValueError("theta_floor must be in [0, 1)")

    # ---- durations & two-type split ----
    D_default = float(user_params.get("D_mean", 12.0))
    D_short = float(user_params.get("D_mean_short", D_default))
    D_long = float(user_params.get("D_mean_long", D_default))
    if D_short <= 1 or D_long <= 1:
        raise ValueError("D_mean_short and D_mean_long must be > 1 month")

    frac_long = float(user_params.get("frac_long", user_params.get("pi_long", 0.0)))
    if not (0.0 <= frac_long <= 1.0):
        raise ValueError("frac_long (pi_long) must be in [0, 1]")
    if frac_long <= 0.0:
        p_form_long = 0.0
    elif frac_long >= 1.0:
        p_form_long = 1.0
    else:
        p_form_long = (frac_long * D_short) / (
            (1.0 - frac_long) * D_long + frac_long * D_short)

    q_short = 1.0 / D_short
    q_long = 1.0 / D_long
    D_mean_eff = (1.0 - p_form_long) * D_short + p_form_long * D_long
    if D_mean_eff <= 1:
        raise ValueError(f"effective mean duration {D_mean_eff:.3f} <= 1")
    q_eff = 1.0 / D_mean_eff

    k_year = float(user_params.get("mean_partners_per_year",
                                   user_params.get("k_mean", 0.0)))
    if k_year <= 0:
        raise ValueError("mean_partners_per_year must be > 0")

    scale_obs = 1.0 + _OBS_WINDOW_MONTHS * q_eff
    k_snap = k_year / scale_obs

    N_eff = float(np.sqrt(NU * NV))
    rho = k_snap / N_eff

    # ---- age / community structure ----
    n_comm = int(user_params.get("n_communities", DEFAULT_N_COMMUNITIES))
    if n_comm < 1:
        raise ValueError("n_communities must be >= 1")
    band_edges = tuple(float(x) for x in
                       user_params.get("age_band_edges", DEFAULT_AGE_BAND_EDGES))
    age_range = tuple(float(x) for x in
                      user_params.get("age_range", DEFAULT_AGE_RANGE))
    n_bands = len(band_edges) + 1

    A = user_params.get("age_mixing")
    if A is None:
        A = default_age_mixing(band_edges, age_range,
                               sigma=float(user_params.get("age_sigma", 8.0)),
                               male_older=float(user_params.get("age_male_older", 2.0)))
    A = np.asarray(A, dtype=float)
    if A.shape != (n_bands, n_bands):
        raise ValueError(f"age_mixing must be {n_bands}x{n_bands}, got {A.shape}")

    C = user_params.get("community_mixing")
    if C is None:
        C = default_community_mixing(
            n_comm, off_diagonal=float(user_params.get("community_off_diag", 0.1)))
    C = np.asarray(C, dtype=float)
    if C.shape != (n_comm, n_comm):
        raise ValueError(f"community_mixing must be {n_comm}x{n_comm}, got {C.shape}")
    if np.any(A < 0) or np.any(C < 0):
        raise ValueError("mixing matrices must be non-negative")

    W_block = _build_W_block(C, A)
    n_blocks = n_comm * n_bands

    init_age_range = tuple(float(x) for x in
                           user_params.get("init_age_range", age_range))
    comm_probs = user_params.get("community_probs")
    if comm_probs is None:
        comm_probs = np.full(n_comm, 1.0 / n_comm)
    comm_probs = np.asarray(comm_probs, dtype=float)
    if comm_probs.shape != (n_comm,):
        raise ValueError(f"community_probs must have length {n_comm}")
    comm_probs = comm_probs / comm_probs.sum()

    print("Interpretable → runtime parameters  [age+community bipartite dcSBM]")
    print(f"  N_U={NU}  N_V={NV}")
    print(f"  partners/yr = {k_year}  →  momentary mean degree k_snap = {k_snap:.4f}")
    print(f"  Gamma shape: U={shape_U}  V={shape_V}   theta_floor={theta_floor}"
          f"  (CV_U={(1-theta_floor)/np.sqrt(shape_U):.2f})")
    print(f"  durations: short={D_short} (q={q_short:.4f}), "
          f"long={D_long} (q={q_long:.4f}); standing long-frac target={frac_long}")
    print(f"  communities={n_comm}, age bands={n_bands} (edges={band_edges}); "
          f"blocks={n_blocks}")
    print(f"  ρ={rho:.6e}  (should be ≪ 1)")

    return {
        # dynamical knobs
        "rho":           rho,
        "gamma_shape_U": shape_U,
        "gamma_shape_V": shape_V,
        "theta_floor":   theta_floor,
        "q":             q_eff,
        "q_short":       q_short,
        "q_long":        q_long,
        "p_form_long":   p_form_long,
        "frac_long_target": frac_long,
        # age / community
        "n_communities": n_comm,
        "n_bands":       n_bands,
        "n_blocks":      n_blocks,
        "age_band_edges": np.asarray(band_edges, dtype=float),
        "age_range":     np.asarray(age_range, dtype=float),
        "A_age":         A,
        "C_comm":        C,
        "W_block":       W_block,
        "init_age_range": np.asarray(init_age_range, dtype=float),
        "community_probs": comm_probs,
        # provenance
        "D_mean_short":  D_short,
        "D_mean_long":   D_long,
        "D_mean_eff":    D_mean_eff,
        "mean_partners_per_year": k_year,
        "k_snap":        k_snap,
    }


def build_model(nU, nV=None):
    """Static structural descriptor: the two population sizes + output cadence."""
    if nV is None:
        nV = nU
    NU = int(nU)
    NV = int(nV)
    if NU <= 0 or NV <= 0:
        raise ValueError("nU and nV must be > 0")
    return {"nU_init": NU, "nV_init": NV, "months_per_step": MONTHS_PER_STEP}


# =====================================================================
# State initialisation
# =====================================================================

def _sample_init_attributes(n, params, rng):
    """Sample (age, community) for the INITIAL population of one side."""
    lo, hi = params["init_age_range"]
    ages = rng.uniform(lo, hi, size=n)
    comm = rng.choice(params["n_communities"], size=n, p=params["community_probs"])
    return ages.astype(float), comm.astype(np.int16)


def init_network_state(model, params, rng):
    """Build the t=0 network state: propensities, ages, communities, initial edges.

    Optional per-side initial attributes may be supplied in ``params`` as
    ``init_ages_U`` / ``init_comm_U`` (and ``..._V``); otherwise they are sampled
    from ``init_age_range`` and ``community_probs``.
    """
    NU = model["nU_init"]
    NV = model["nV_init"]
    band_edges = params["age_band_edges"]
    n_bands = params["n_bands"]

    floor = params.get("theta_floor", 0.0)
    theta_U = _sample_side_theta(NU, params["gamma_shape_U"], rng,
                                 floor=floor, exact_mean_one=True)
    theta_V = _sample_side_theta(NV, params["gamma_shape_V"], rng,
                                 floor=floor, exact_mean_one=True)

    if params.get("init_ages_U") is not None and params.get("init_comm_U") is not None:
        age_U = np.asarray(params["init_ages_U"], float)
        comm_U = np.asarray(params["init_comm_U"], np.int16)
    else:
        age_U, comm_U = _sample_init_attributes(NU, params, rng)
    if params.get("init_ages_V") is not None and params.get("init_comm_V") is not None:
        age_V = np.asarray(params["init_ages_V"], float)
        comm_V = np.asarray(params["init_comm_V"], np.int16)
    else:
        age_V, comm_V = _sample_init_attributes(NV, params, rng)

    band_U = _age_to_band(age_U, band_edges)
    band_V = _age_to_band(age_V, band_edges)

    state = {
        "u_uid":      np.arange(NU, dtype=np.int64),
        "u_theta":    np.ascontiguousarray(theta_U, dtype=float),
        "u_entry":    np.full(NU, ENTRY_INIT, dtype=np.int8),
        "u_age":      np.ascontiguousarray(age_U, dtype=float),
        "u_comm":     np.ascontiguousarray(comm_U, dtype=np.int16),
        "u_band":     band_U.astype(np.int64),
        "u_block":    _block_of(comm_U, band_U, n_bands),
        "v_uid":      np.arange(NV, dtype=np.int64),
        "v_theta":    np.ascontiguousarray(theta_V, dtype=float),
        "v_entry":    np.full(NV, ENTRY_INIT, dtype=np.int8),
        "v_age":      np.ascontiguousarray(age_V, dtype=float),
        "v_comm":     np.ascontiguousarray(comm_V, dtype=np.int16),
        "v_band":     band_V.astype(np.int64),
        "v_block":    _block_of(comm_V, band_V, n_bands),
        "edges_u":    np.empty(0, dtype=np.int64),
        "edges_v":    np.empty(0, dtype=np.int64),
        "edges_type": np.empty(0, dtype=np.int8),
        "edge_birth": np.empty(0, dtype=np.int64),
        "next_uid_U": int(NU),
        "next_uid_V": int(NV),
    }

    new_u, new_v, new_t = _sample_edges(params, state, scale=params["rho"], rng=rng)
    state["edges_u"] = new_u
    state["edges_v"] = new_v
    state["edges_type"] = new_t
    state["edge_birth"] = np.zeros(new_u.size, dtype=np.int64)
    return state


# =====================================================================
# Edge sampler  (degree-corrected stochastic block model, sparse)
# =====================================================================

def _draw_within_blocks(cand_block, uid, theta, block, n_blocks, rng):
    """For each candidate (given its block id), draw a node ∝ theta within that block.

    Vectorised: one argsort of the side by block, then a loop over the few blocks
    actually present among the candidates (each iteration fully vectorised).
    """
    m = cand_block.size
    out = np.empty(m, dtype=np.int64)
    if m == 0:
        return out
    # group the side's nodes by block
    order = np.argsort(block, kind="stable")
    blk_s = block[order]
    uid_s = uid[order]
    th_s = theta[order]
    bounds = np.searchsorted(blk_s, np.arange(n_blocks + 1))
    # group the candidates by block
    cand_order = np.argsort(cand_block, kind="stable")
    cb_s = cand_block[cand_order]
    present = np.unique(cb_s)
    starts = np.searchsorted(cb_s, present, side="left")
    ends = np.r_[starts[1:], m]
    for k in range(present.size):
        g = int(present[k])
        lo, hi = int(bounds[g]), int(bounds[g + 1])
        members = uid_s[lo:hi]
        th = th_s[lo:hi]
        cdf = np.cumsum(th)
        tot = float(cdf[-1])
        sel = cand_order[starts[k]:ends[k]]
        u = rng.random(sel.size) * tot
        picks = np.clip(np.searchsorted(cdf, u, side="right"), 0, members.size - 1)
        out[sel] = members[picks]
    return out


def _sample_edges(params, state, scale, rng):
    """Sample new unique U–V edges from the sparse degree-corrected block model.

    Per-dyad intensity ≈ ``scale · θ_u · φ_v · W[block_u, block_v]``. The total
    number of candidates is ``Poisson(scale · S_U · S_V)`` (independent of the
    mixing, so the mean degree is preserved); candidates are split across block
    pairs ∝ ``W·S_U^g·S_V^h`` and each endpoint drawn ∝ propensity within its
    block. Within-batch duplicates and existing ties are dropped; bipartiteness
    is structural (endpoints from disjoint sets).
    """
    theta = state["u_theta"]
    phi = state["v_theta"]
    u_uid = state["u_uid"]
    v_uid = state["v_uid"]
    u_block = state["u_block"]
    v_block = state["v_block"]
    G = int(params["n_blocks"])
    W = params["W_block"]
    empty = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
             np.empty(0, dtype=np.int8))
    if u_uid.size == 0 or v_uid.size == 0 or scale <= 0.0:
        return empty

    S_U = float(theta.sum())
    S_V = float(phi.sum())
    if S_U <= 0.0 or S_V <= 0.0:
        return empty

    m = int(rng.poisson(scale * S_U * S_V))
    if m == 0:
        return empty

    # block propensity sums, and per-block-pair intensities
    SUg = np.bincount(u_block, weights=theta, minlength=G)
    SVh = np.bincount(v_block, weights=phi, minlength=G)
    R = W * np.outer(SUg, SVh)
    Rtot = float(R.sum())
    if Rtot <= 0.0:
        return empty

    counts = rng.multinomial(m, (R / Rtot).ravel()).reshape(G, G)
    nz = np.argwhere(counts > 0)
    if nz.shape[0] == 0:
        return empty
    gs = nz[:, 0]
    hs = nz[:, 1]
    cs = counts[gs, hs]
    man_block = np.repeat(gs, cs)
    woman_block = np.repeat(hs, cs)

    a = _draw_within_blocks(man_block, u_uid, theta, u_block, G, rng)
    b = _draw_within_blocks(woman_block, v_uid, phi, v_block, G, rng)

    cand_keys = a * _KEY + b
    cand_keys, idx = np.unique(cand_keys, return_index=True)
    a, b = a[idx], b[idx]

    if state["edges_u"].size:
        existing = state["edges_u"] * _KEY + state["edges_v"]
        fresh = ~np.isin(cand_keys, existing)
        a, b = a[fresh], b[fresh]

    p_form_long = float(params.get("p_form_long", 0.0))
    types = (rng.random(a.size) < p_form_long).astype(np.int8)   # 1=long, 0=short
    return a.astype(np.int64), b.astype(np.int64), types


# =====================================================================
# Duration bookkeeping
# =====================================================================

def _record_durations(durations, durations_type, t, edge_birth, edge_type, mask):
    """Append (t - birth) and type for the edges selected by ``mask``."""
    if durations is None:
        return
    ages = (t - edge_birth[mask]).tolist()
    durations.extend(ages)
    if durations_type is not None:
        durations_type.extend(edge_type[mask].tolist())


# =====================================================================
# External demography: deaths (supplied) then births (supplied)
# =====================================================================

_U_KEYS = ("u_uid", "u_theta", "u_entry", "u_age", "u_comm", "u_band", "u_block")
_V_KEYS = ("v_uid", "v_theta", "v_entry", "v_age", "v_comm", "v_band", "v_block")


def _drop_side_nodes(state, side, dead, node_events, t):
    """Remove ``dead`` UIDs (and mark events) from one side; edges handled by caller."""
    keys = _U_KEYS if side == "U" else _V_KEYS
    uid_key = keys[0]
    keep = ~np.isin(state[uid_key], dead)
    for k in keys:
        state[k] = state[k][keep]
    for uid in np.asarray(dead, dtype=np.int64).tolist():
        node_events.append((t, int(uid), side, "extra_death"))


def _append_side_births(state, params, side, births, t, node_events, rng):
    """Append externally-supplied entrants to one side.

    ``births`` is a dict with arrays ``"age"`` and ``"comm"`` (required) and an
    optional ``"theta"`` (else sampled Gamma, mean 1). Fresh UIDs are assigned.
    """
    if births is None:
        return
    age = np.asarray(births["age"], dtype=float)
    comm = np.asarray(births["comm"], dtype=np.int16)
    n = int(age.size)
    if n == 0:
        return
    if comm.size != n:
        raise ValueError("births 'age' and 'comm' must have equal length")
    n_comm = int(params["n_communities"])
    if np.any(comm < 0) or np.any(comm >= n_comm):
        raise ValueError(f"births 'comm' must be in [0,{n_comm})")

    if side == "U":
        shape = params["gamma_shape_U"]
        uid_key, theta_key, entry_key = "u_uid", "u_theta", "u_entry"
        age_key, comm_key, band_key, block_key = "u_age", "u_comm", "u_band", "u_block"
        next_key = "next_uid_U"
    else:
        shape = params["gamma_shape_V"]
        uid_key, theta_key, entry_key = "v_uid", "v_theta", "v_entry"
        age_key, comm_key, band_key, block_key = "v_age", "v_comm", "v_band", "v_block"
        next_key = "next_uid_V"

    theta = births.get("theta")
    theta = (_sample_side_theta(n, shape, rng, floor=params.get("theta_floor", 0.0))
             if theta is None else np.asarray(theta, dtype=float))
    band = _age_to_band(age, params["age_band_edges"])
    block = _block_of(comm, band, params["n_bands"])

    start = state[next_key]
    born = np.arange(start, start + n, dtype=np.int64)
    state[next_key] += n

    state[uid_key]   = np.concatenate([state[uid_key], born])
    state[theta_key] = np.concatenate([state[theta_key], theta])
    state[entry_key] = np.concatenate([state[entry_key],
                                       np.full(n, ENTRY_BIRTH, dtype=np.int8)])
    state[age_key]   = np.concatenate([state[age_key], age])
    state[comm_key]  = np.concatenate([state[comm_key], comm])
    state[band_key]  = np.concatenate([state[band_key], band])
    state[block_key] = np.concatenate([state[block_key], block])
    for uid in born.tolist():
        node_events.append((t, int(uid), side, "birth"))


def _apply_external_turnover(state, params, rng, t, node_events,
                             track_durations, durations, durations_type,
                             *, extra_deaths_U, extra_deaths_V,
                             births_U, births_V):
    """One round of external deaths (both sides) then external births (both sides)."""
    def _alive(dead, uid):
        if dead is None:
            return np.empty(0, dtype=np.int64)
        dead = np.asarray(dead, dtype=np.int64)
        return dead[np.isin(dead, uid)]

    dead_U = _alive(extra_deaths_U, state["u_uid"])
    dead_V = _alive(extra_deaths_V, state["v_uid"])

    # remove edges incident to any dead node
    if dead_U.size or dead_V.size:
        eu, ev = state["edges_u"], state["edges_v"]
        if eu.size:
            incident = np.zeros(eu.size, dtype=bool)
            if dead_U.size:
                incident |= np.isin(eu, dead_U)
            if dead_V.size:
                incident |= np.isin(ev, dead_V)
            if track_durations and incident.any():
                _record_durations(durations, durations_type, t,
                                  state["edge_birth"], state["edges_type"], incident)
            keep = ~incident
            state["edges_u"]    = eu[keep]
            state["edges_v"]    = ev[keep]
            state["edges_type"] = state["edges_type"][keep]
            state["edge_birth"] = state["edge_birth"][keep]

    if dead_U.size:
        _drop_side_nodes(state, "U", dead_U, node_events, t)
    if dead_V.size:
        _drop_side_nodes(state, "V", dead_V, node_events, t)

    _append_side_births(state, params, "U", births_U, t, node_events, rng)
    _append_side_births(state, params, "V", births_V, t, node_events, rng)


# =====================================================================
# Snapshot
# =====================================================================

def build_snapshot(state, model, *, edges_u=None, edges_v=None, edges_type=None):
    """
    Raw snapshot of the current state for the epidemic model, carrying per-node
    age & community. Supplying edge arrays overrides the live edges (used for the
    integrated 3-month union). Edges to inactive nodes are dropped.
    """
    u_uid, v_uid = state["u_uid"], state["v_uid"]

    active_U = {int(u_uid[i]): {"entry": int(state["u_entry"][i]),
                                "age": float(state["u_age"][i]),
                                "comm": int(state["u_comm"][i])}
                for i in range(u_uid.size)}
    active_V = {int(v_uid[i]): {"entry": int(state["v_entry"][i]),
                                "age": float(state["v_age"][i]),
                                "comm": int(state["v_comm"][i])}
                for i in range(v_uid.size)}

    if edges_u is None:
        edges_u = state["edges_u"]
        edges_v = state["edges_v"]
        edges_type = state["edges_type"]
    eu = np.asarray(edges_u, dtype=np.int64)
    ev = np.asarray(edges_v, dtype=np.int64)
    et = np.asarray(edges_type, dtype=np.int8)
    if eu.size:
        alive = np.isin(eu, u_uid) & np.isin(ev, v_uid)
        eu, ev, et = eu[alive], ev[alive], et[alive]

    return {
        "edges_u":    eu.copy(),
        "edges_v":    ev.copy(),
        "edges_type": et.copy(),
        "active_U":   active_U,
        "active_V":   active_V,
    }


# =====================================================================
# One monthly timestep
# =====================================================================

def network_step(state, model, params, rng, t, *,
                 extra_deaths_U=None, extra_deaths_V=None,
                 births_U=None, births_V=None,
                 track_durations=False,
                 durations=None, durations_type=None, node_events=None):
    """
    Advance the network by one **monthly** substep in place.

    Order: external turnover (supplied deaths, then supplied births) → per-type
    edge dissolution → degree-corrected block formation. There is no natural
    demography: with no ``extra_deaths_*`` / ``births_*`` the population is fixed.
    """
    q_short = params["q_short"]
    q_long = params["q_long"]

    if node_events is None:
        node_events = []
    if durations is None:
        durations = []

    # --- 1. external turnover (deaths then births) ---
    has_turnover = (extra_deaths_U is not None or extra_deaths_V is not None
                    or births_U is not None or births_V is not None)
    if has_turnover:
        _apply_external_turnover(
            state, params, rng, t, node_events,
            track_durations, durations, durations_type,
            extra_deaths_U=extra_deaths_U, extra_deaths_V=extra_deaths_V,
            births_U=births_U, births_V=births_V)

    # --- 2. edge dissolution (per-type monthly probability) ---
    eu, ev = state["edges_u"], state["edges_v"]
    if eu.size > 0:
        et = state["edges_type"]
        q_edge = np.where(et == EDGE_LONG, q_long, q_short)
        drop = rng.random(eu.size) < q_edge
        if track_durations and drop.any():
            _record_durations(durations, durations_type, t,
                              state["edge_birth"], et, drop)
        keep = ~drop
        state["edges_u"]    = eu[keep]
        state["edges_v"]    = ev[keep]
        state["edges_type"] = et[keep]
        state["edge_birth"] = state["edge_birth"][keep]

    # --- 3. edge formation ---
    # Total formation scale = q_eff · rho keeps the equilibrium mean degree at
    # k_snap; the population-size correction keeps the per-node rate constant as
    # external demography changes N.
    nU_now = int(state["u_uid"].size)
    nV_now = int(state["v_uid"].size)
    N_eff_now = float(np.sqrt(max(nU_now, 1) * max(nV_now, 1)))
    N_eff_init = float(np.sqrt(model["nU_init"] * model["nV_init"]))
    eff_rho = params["rho"] * N_eff_init / max(N_eff_now, 1.0)
    nu, nv, nt = _sample_edges(params, state, scale=params["q"] * eff_rho, rng=rng)
    if nu.size:
        state["edges_u"]    = np.concatenate([state["edges_u"], nu])
        state["edges_v"]    = np.concatenate([state["edges_v"], nv])
        state["edges_type"] = np.concatenate([state["edges_type"], nt])
        state["edge_birth"] = np.concatenate(
            [state["edge_birth"], np.full(nu.size, t, dtype=np.int64)])


# =====================================================================
# Main simulation loop
# =====================================================================

def _default_burn_months(params):
    """Enough months to equilibrate the long ties (~ several × D_mean_long)."""
    return int(max(5.0 * params.get("D_mean_long", params.get("D_mean_eff", 12.0)),
                   120))


def _persist_snapshot(path, edges_u, edges_v, edges_type, active_u, active_v, t):
    np.savez_compressed(
        path,
        edges_u=np.asarray(edges_u, dtype=np.int64),
        edges_v=np.asarray(edges_v, dtype=np.int64),
        edges_type=np.asarray(edges_type, dtype=np.int8),
        active_u=np.asarray(active_u, dtype=np.int64),
        active_v=np.asarray(active_v, dtype=np.int64),
        t=np.int64(t))


def _union_3month(sub_keys, sub_types):
    """Collapse the substep edge lists into the quarter union (long wins on ties)."""
    if not sub_keys:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8))
    keys = np.concatenate(sub_keys)
    types = np.concatenate(sub_types)
    if keys.size == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int8))
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    types = types[order]
    uniq, start = np.unique(keys, return_index=True)
    # max type within each run of equal keys (long=1 beats short=0)
    maxtype = np.maximum.reduceat(types, start).astype(np.int8)
    iu = (uniq // _KEY).astype(np.int64)
    iv = (uniq % _KEY).astype(np.int64)
    return iu, iv, maxtype


def simulate(model, params, T, *,
             seed=1,
             burn_in_months=None,
             save_edge_snapshots=(),
             track_durations=True,
             base_dir=None,
             return_raw_snapshots=True,
             extra_deaths_U_by_month=None,
             extra_deaths_V_by_month=None,
             extra_births_U_by_month=None,
             extra_births_V_by_month=None,
             **kwargs):
    """
    Run the network for ``T`` output steps (quarters); emit the integrated
    3-month union each quarter.

    Demography is external. ``extra_deaths_*_by_month`` map ``rel_month -> UID
    array``; ``extra_births_*_by_month`` map ``rel_month -> {"age":..., "comm":...,
    "theta"?:...}``. ``rel_month`` is 1-based from the first emitted month
    (post-burn-in). The burn-in runs with **no** demography.

    Returns the epidemic-model inputs, including the new per-node attribute
    arrays ``node_ages_list`` and ``node_comm_list`` (aligned with
    ``active_nodes_list``).
    """
    if not (0 < params["q_short"] < 1) or not (0 < params["q_long"] < 1):
        raise ValueError("q_short and q_long must be in (0, 1)")

    months_per_step = int(model.get("months_per_step", MONTHS_PER_STEP))
    if base_dir is None:
        base_dir = os.path.expanduser("~/Downloads/age_community_bipartite_network")
    os.makedirs(base_dir, exist_ok=True)

    rng = np.random.default_rng(seed)
    state = init_network_state(model, params, rng)

    # silent burn-in to stationarity (no demography, no tracking)
    if burn_in_months is None:
        burn_in_months = _default_burn_months(params)
    for bt in range(1, int(burn_in_months) + 1):
        network_step(state, model, params, rng, bt)
    t0 = int(burn_in_months)

    node_events = []
    durations = [] if track_durations else None
    durations_type = [] if track_durations else None

    snap_times_set = {int(s) for s in save_edge_snapshots if 0 <= int(s) <= T}
    saved_paths = []
    v_offset = int(max(state["next_uid_U"], model["nU_init"]) + 1000)

    def _save(t_snap, snap):
        path = os.path.join(base_dir, f"edges_q{t_snap}.npz")
        _persist_snapshot(path, snap["edges_u"], snap["edges_v"], snap["edges_type"],
                          sorted(snap["active_U"].keys()),
                          sorted(snap["active_V"].keys()), t_snap)
        saved_paths.append(path)

    keep_raw = bool(return_raw_snapshots)
    raw_snapshots = [] if keep_raw else None
    adj_list, adj_type_list = [], []
    active_nodes_list, node_types_list, entry_kind_list = [], [], []
    node_ages_list, node_comm_list = [], []

    def _ingest(snap):
        adj, adj_type, an, nt, ek, ages, comm = _epi_snapshot_tuple(snap, v_offset)
        adj_list.append(adj)
        adj_type_list.append(adj_type)
        active_nodes_list.append(an)
        node_types_list.append(nt)
        entry_kind_list.append(ek)
        node_ages_list.append(ages)
        node_comm_list.append(comm)
        if keep_raw:
            raw_snapshots.append(snap)

    # Quarter 0: instantaneous baseline network.
    if T > 0:
        snap0 = build_snapshot(state, model)
        _ingest(snap0)
        if 0 in snap_times_set:
            _save(0, snap0)

    for quarter in range(1, T + 1):
        sub_keys, sub_types = [], []
        for sub in range(months_per_step):
            rel_month = (quarter - 1) * months_per_step + sub + 1
            month = t0 + rel_month
            edU = (extra_deaths_U_by_month.get(rel_month)
                   if extra_deaths_U_by_month else None)
            edV = (extra_deaths_V_by_month.get(rel_month)
                   if extra_deaths_V_by_month else None)
            bU = (extra_births_U_by_month.get(rel_month)
                  if extra_births_U_by_month else None)
            bV = (extra_births_V_by_month.get(rel_month)
                  if extra_births_V_by_month else None)
            network_step(
                state, model, params, rng, month,
                extra_deaths_U=edU, extra_deaths_V=edV,
                births_U=bU, births_V=bV,
                track_durations=track_durations,
                durations=durations, durations_type=durations_type,
                node_events=node_events)
            eu, ev, et = state["edges_u"], state["edges_v"], state["edges_type"]
            if eu.size:
                sub_keys.append(eu * _KEY + ev)
                sub_types.append(et.copy())

        iu, iv, tarr = _union_3month(sub_keys, sub_types)
        snap = build_snapshot(state, model, edges_u=iu, edges_v=iv, edges_type=tarr)
        if quarter < T:
            _ingest(snap)
        if quarter in snap_times_set:
            _save(quarter, snap)

    final_month = t0 + T * months_per_step
    censored_ages = ((final_month - state["edge_birth"]).tolist()
                     if track_durations else None)

    return {
        "T":                 T,
        "months_per_step":   months_per_step,
        "state":             state,
        "raw_snapshots":     raw_snapshots,
        "node_events":       node_events,
        "durations":         durations,
        "durations_type":    durations_type,
        "censored_ages":     censored_ages,
        "snapshots":         saved_paths,
        "base_dir":          base_dir,
        "adj_list":          adj_list,
        "adj_type_list":     adj_type_list,
        "active_nodes_list": active_nodes_list,
        "node_types_list":   node_types_list,
        "entry_kind_list":   entry_kind_list,
        "node_ages_list":    node_ages_list,
        "node_comm_list":    node_comm_list,
        "change_times":      np.arange(T + 1, dtype=float),
        "epi_v_offset":      v_offset,
    }


# =====================================================================
# Empirical calibration (burn-in) — unchanged logic
# =====================================================================

def _measure_burnin(cal_model, params, burn_months, window, seed):
    """Burn in, then measure yearly distinct partners and standing long-fraction."""
    rng = np.random.default_rng(seed)
    state = init_network_state(cal_model, params, rng)
    t = 0
    for _ in range(burn_months):
        t += 1
        network_step(state, cal_model, params, rng, t)

    pm, pw = {}, {}
    am, aw = set(), set()
    long_frac = []
    for _ in range(window):
        t += 1
        network_step(state, cal_model, params, rng, t)
        am.update(state["u_uid"].tolist())
        aw.update(state["v_uid"].tolist())
        for a, b in zip(state["edges_u"].tolist(), state["edges_v"].tolist()):
            pm.setdefault(a, set()).add(b)
            pw.setdefault(b, set()).add(a)
        et = state["edges_type"]
        long_frac.append(float((et == EDGE_LONG).mean()) if et.size else 0.0)

    counts = ([len(pm.get(u, ())) for u in am] + [len(pw.get(v, ())) for v in aw])
    k_year = float(np.mean(counts)) if counts else 0.0
    f_long = float(np.mean(long_frac)) if long_frac else 0.0
    return k_year, f_long


def calibrate(model, params, *, n_cal=2500, burn_months=None, window=12,
              max_iters=6, tol=0.03, seed=12345, verbose=True):
    """
    Tune ``rho`` and ``p_form_long`` so the realised network matches the inputs.

    The mixing structure preserves the mean degree, so the same rescaling of
    ``rho`` (to the yearly-partner target) and odds-ratio nudge of
    ``p_form_long`` (to the standing long-fraction) used by the simple toy still
    apply. Returns a new params dict.
    """
    target_k = params["mean_partners_per_year"]
    target_f = params.get("frac_long_target", 0.0)
    k_snap = params["k_snap"]

    ratio = model["nV_init"] / model["nU_init"]
    nU_cal = int(min(model["nU_init"], n_cal))
    nV_cal = int(max(1, round(nU_cal * ratio)))
    cal_model = build_model(nU_cal, nV_cal)
    rho_analytic_cal = k_snap / np.sqrt(nU_cal * nV_cal)

    if burn_months is None:
        burn_months = int(max(5.0 * params["D_mean_long"], 120))

    cp = dict(params)
    cp["rho"] = rho_analytic_cal

    if verbose:
        print("─" * 64)
        print(f"calibrate: size {nU_cal}+{nV_cal}, burn={burn_months} mo, "
              f"targets: partners/yr={target_k}, long-frac={target_f}")

    k_real = f_real = None
    for it in range(max_iters):
        k_real, f_real = _measure_burnin(cal_model, cp, burn_months, window, seed + it)
        cp["rho"] *= target_k / max(k_real, 1e-9)
        if 0.0 < target_f < 1.0:
            p = min(max(cp["p_form_long"], 1e-6), 1.0 - 1e-6)
            odds = (p / (1.0 - p)) * ((target_f / (1.0 - target_f)) /
                                      max(f_real / (1.0 - f_real), 1e-9))
            cp["p_form_long"] = odds / (1.0 + odds)
        ok_k = abs(k_real - target_k) <= tol * target_k
        ok_f = (target_f <= 0.0 or target_f >= 1.0 or
                abs(f_real - target_f) <= tol)
        if verbose:
            print(f"  iter {it}: partners/yr={k_real:.3f}, long-frac={f_real:.3f}"
                  f"  →  p_form_long={cp['p_form_long']:.3f}")
        if ok_k and ok_f:
            break

    c = cp["rho"] / rho_analytic_cal
    out = dict(params)
    out["rho"] = params["rho"] * c
    out["p_form_long"] = cp["p_form_long"]
    out["rho_correction_factor"] = float(c)
    out["calibrated"] = True
    if verbose:
        print(f"  → rho correction factor = {c:.3f}; "
              f"final p_form_long = {out['p_form_long']:.3f}")
        print("─" * 64)
    return out


__all__ = [
    "interpretable_to_params", "build_model", "init_network_state",
    "build_snapshot", "network_step", "simulate", "calibrate",
    "sample_gamma", "gamma_mean",
    "default_age_mixing", "default_community_mixing",
    "MONTHS_PER_STEP", "ENTRY_INIT", "ENTRY_BIRTH", "EDGE_SHORT", "EDGE_LONG",
]


# =====================================================================
# Tiny self-test / usage example
# =====================================================================

if __name__ == "__main__":
    user_params = {
        "mean_partners_per_year": 2.0,
        "gamma_shape":            1.5,
        "D_mean_short":           2.0,
        "D_mean_long":            36.0,
        "pi_long":                0.4,
        "n_communities":          4,
        # default age bands + assortative age & community mixing
    }
    nU = nV = 2000
    params = interpretable_to_params(user_params, nU, nV)
    model = build_model(nU, nV)
    params = calibrate(model, params, n_cal=1500, verbose=True)

    # External demography demo: kill a few known initial nodes at rel_month 1,
    # and add entrants at rel_month 2 carrying explicit age & community.
    rng0 = np.random.default_rng(0)
    births = {"age": rng0.uniform(16, 19, size=40),
              "comm": rng0.integers(0, 4, size=40)}
    out = simulate(model, params, T=8, seed=0,
                   extra_deaths_U_by_month={1: np.arange(10)},
                   extra_births_U_by_month={2: births})

    q = 0
    adj = out["adj_list"][q]
    nt = out["node_types_list"][q]
    comm = out["node_comm_list"][q]
    ages = out["node_ages_list"][q]

    coo = adj.tocoo()
    up = coo.row < coo.col
    i, j = coo.row[up], coo.col[up]
    same_comm = float(np.mean(comm[i] == comm[j])) if i.size else float("nan")
    age_gap = float(np.mean(np.abs(ages[i] - ages[j]))) if i.size else float("nan")

    print(f"\nEmitted {len(out['adj_list'])} quarterly snapshots.")
    print(f"Quarter-0: {nt.size} nodes ({int((nt==0).sum())} men, "
          f"{int((nt==1).sum())} women), {i.size} edges.")
    print(f"  within-community edge fraction = {same_comm:.2f} "
          f"(vs ~0.25 if mixing were random over 4 communities)")
    print(f"  mean |age_man - age_woman| on edges = {age_gap:.1f} years")
    print(f"  births recorded: "
          f"{sum(1 for e in out['node_events'] if e[3]=='birth')}, "
          f"extra_deaths: {sum(1 for e in out['node_events'] if e[3]=='extra_death')}")

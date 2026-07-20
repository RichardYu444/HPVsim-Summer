"""
make_francesco_network(pars, popdict)
    Build model + state at t=0. Returns a dict with keys:
    'model', 'params', 'state', 'layer', 'uid_to_idx', 'idx_to_uid'

step_francesco_network(f_net, people, pars, t)
    Advance one monthly timestep in place and push updated edges back
    into people.contacts['f'].

print_network_stats(f_net, people, t)
    Print mean degree, edge count, population size etc.
    Safe to call at any timestep.
"""

import numpy as np

# --- standalone network module (same package) ---
try:
    from . import simple_network_model as snm
except ImportError:
    import simple_network_model as snm  # fallback for direct script runs


# =====================================================================
# Internal helpers
# =====================================================================

def _make_layer(p1, p2, acts):
    """
    Return a plain dict satisfying HPVsim's Layer contract.

    Parameters
    ----------
    p1, p2 : int arrays
        Person *indices* (not UIDs) into the people array.
    acts : float array or scalar
        Number of sexual acts per partnership per timestep.
    """
    p1 = np.asarray(p1, dtype=np.int64)
    p2 = np.asarray(p2, dtype=np.int64)
    n  = p1.size
    if np.isscalar(acts):
        acts = np.full(n, acts, dtype=float)
    else:
        acts = np.asarray(acts, dtype=float)
    return dict(p1=p1, p2=p2, acts=acts)


def _uid_edges_to_indices(edges_u, edges_v, uid_to_idx):
    """
    Convert UID-based edge arrays to people-array index pairs.
    Edges whose endpoints are not in uid_to_idx are silently dropped
    (can happen transiently during turnover).
    """
    eu = np.asarray(edges_u, dtype=np.int64)
    ev = np.asarray(edges_v, dtype=np.int64)
    if eu.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    # vectorised lookup via a pre-built array
    # uid_to_idx is a numpy array indexed by UID
    max_uid = uid_to_idx.size - 1
    in_range = (eu <= max_uid) & (ev <= max_uid)
    eu, ev = eu[in_range], ev[in_range]
    if eu.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

    idx_u = uid_to_idx[eu]
    idx_v = uid_to_idx[ev]
    valid = (idx_u >= 0) & (idx_v >= 0)
    return idx_u[valid], idx_v[valid]


def _build_uid_to_idx(uid_array, max_uid):
    """
    Build a lookup array of shape (max_uid+1,) where
    lookup[uid] = position of uid in uid_array, or -1 if absent.
    """
    lookup = np.full(max_uid + 1, -1, dtype=np.int64)
    lookup[uid_array] = np.arange(uid_array.size, dtype=np.int64)
    return lookup


def _compute_acts(pars, n_edges):
    """
    Compute acts per partnership using HPVsim's built-in acts
    parameters from the 'f' layer if defined, otherwise fall back
    to sensible defaults.

    HPVsim stores acts as a lognormal: mean = f_acts_mean,
    std = f_acts_std (both per timestep).
    """
    acts_mean = float(pars.get('f_acts_mean', 1.0))
    acts_std  = float(pars.get('f_acts_std',  0.5))
    if n_edges == 0:
        return np.empty(0, dtype=float)
    # lognormal parametrisation matching HPVsim convention
    sigma2 = np.log1p((acts_std / max(acts_mean, 1e-9)) ** 2)
    mu     = np.log(max(acts_mean, 1e-9)) - 0.5 * sigma2
    return np.random.lognormal(mu, np.sqrt(sigma2), size=n_edges)


# =====================================================================
# Initialisation
# =====================================================================

def make_francesco_network(pars, popdict):
    """
    Build the Francesco network at t=0.

    Parameters
    ----------
    pars    : HPVsim parameter dict  (must contain the 'f_*' keys)
    popdict : HPVsim population dict with keys 'uid', 'sex'
              sex == 0 → female (nodes_U), sex == 1 → male (nodes_V)

    Returns
    -------
    f_net : dict
        Runtime container passed to every subsequent call.
        Keys: 'model', 'params', 'state', 'uid_to_idx',
              'female_uids', 'male_uids', 'all_uids'
    """
    uid = np.asarray(popdict['uid'],  dtype=np.int64)
    sex = np.asarray(popdict['sex'],  dtype=np.int8)

    female_uids = uid[sex == 0]
    male_uids   = uid[sex == 1]
    nU          = int(female_uids.size)   # females → nodes_U
    nV          = int(male_uids.size)     # males   → nodes_V

    if nU == 0:
        raise ValueError("make_francesco_network: no female agents found.")
    if nV == 0:
        raise ValueError("make_francesco_network: no male agents found.")

    # --- build interpretable user_params from HPVsim pars ---
    user_params = {
        'mean_partners_per_year': float(pars.get('f_mean_partners_per_year', 2.0)),
        'exponent':               float(pars.get('f_exponent',               2.5)),
        'kappa':                  float(pars.get('f_kappa',                  10.0)),
        'D_mean':                 float(pars.get('f_D_mean',                 12.0)),
        'tau':                    float(pars.get('f_tau',                    360.0)),
        'epsilon':                float(pars.get('f_epsilon',                0.0)),
        'xmin':                   float(pars.get('f_xmin',                   1.0)),
    }

    # single unipartite pool: combine both sexes
    # (simple_network_model is unipartite; we map all agents in)
    N_total       = nU + nV
    comm_sizes_U  = [N_total]

    net_params = snm.interpretable_to_params(
        user_params, N_total, comm_sizes_U, bipartite=False)
    net_params = snm.calibrate_rho(
        snm.build_model(N_total, comm_sizes_U, net_params),
        net_params)

    model = snm.build_model(N_total, comm_sizes_U, net_params)
    rng   = np.random.default_rng(int(pars.get('rand_seed', 1)))
    state = snm.init_network_state(model, net_params, rng)

    # --- re-label UIDs so they match HPVsim's uid array ---
    # simple_network_model initialises UIDs as 0..N-1;
    # we remap them to HPVsim UIDs (female first, then male)
    hpvsim_uids         = np.concatenate([female_uids, male_uids])
    state['u_uid']      = hpvsim_uids.copy()
    state['next_uid']   = int(uid.max()) + 1

    # fix edge UIDs: edges were sampled against 0..N-1, remap
    old_to_new = hpvsim_uids  # position i had old UID i
    if state['edges_u'].size:
        state['edges_u'] = old_to_new[state['edges_u']]
        state['edges_v'] = old_to_new[state['edges_v']]

    # --- UID → people-array index lookup ---
    max_uid     = int(uid.max())
    uid_to_idx  = _build_uid_to_idx(uid, max_uid)

    f_net = {
        'model':        model,
        'params':       net_params,
        'state':        state,
        'uid_to_idx':   uid_to_idx,
        'female_uids':  female_uids,
        'male_uids':    male_uids,
        'all_uids':     hpvsim_uids,
        'rng':          rng,
        'max_uid':      max_uid,
        # statistics accumulators
        '_mean_degree_history': [],
        '_edge_count_history':  [],
        '_pop_size_history':    [],
        '_timestep_history':    [],
    }
    return f_net


# =====================================================================
# Per-timestep update
# =====================================================================

def step_francesco_network(f_net, people, pars, t):
    """
    Advance the Francesco network by one timestep and update
    people.contacts['f'].

    Parameters
    ----------
    f_net  : dict returned by make_francesco_network
    people : HPVsim People object
    pars   : HPVsim parameter dict
    t      : current integer timestep
    """
    state     = f_net['state']
    model     = f_net['model']
    net_params = f_net['params']
    rng       = f_net['rng']

    # advance the network
    snm.network_step(
        state, model, net_params, rng, t,
        extra_deaths_U=None,
        track_durations=False)

    # rebuild UID→idx lookup (population size may have drifted)
    uid        = state['u_uid']
    max_uid    = int(uid.max()) if uid.size else 0
    uid_to_idx = _build_uid_to_idx(uid, max(max_uid, f_net['max_uid']))
    f_net['uid_to_idx'] = uid_to_idx
    f_net['max_uid']    = max(max_uid, f_net['max_uid'])

    # convert UID edges → people-index edges
    idx_u, idx_v = _uid_edges_to_indices(
        state['edges_u'], state['edges_v'], uid_to_idx)

    # enforce sex constraint: keep only edges where one end is female
    # and the other is male (heterosexual contact)
    if idx_u.size > 0:
        hpv_uid_u = uid[idx_u] if idx_u.size else np.empty(0, dtype=np.int64)
        hpv_uid_v = uid[idx_v] if idx_v.size else np.empty(0, dtype=np.int64)

        # rebuild people uid→sex map
        p_sex = np.asarray(people.sex, dtype=np.int8)  # 0=F 1=M
        p_uid = np.asarray(people.uid, dtype=np.int64)
        p_max = int(p_uid.max())
        sex_lookup = np.full(p_max + 1, -1, dtype=np.int8)
        sex_lookup[p_uid] = p_sex

        in_range = (hpv_uid_u <= p_max) & (hpv_uid_v <= p_max)
        hpv_uid_u = hpv_uid_u[in_range]
        hpv_uid_v = hpv_uid_v[in_range]
        idx_u     = idx_u[in_range]
        idx_v     = idx_v[in_range]

        sex_u = sex_lookup[hpv_uid_u]
        sex_v = sex_lookup[hpv_uid_v]
        hetero = (sex_u != sex_v) & (sex_u >= 0) & (sex_v >= 0)
        idx_u  = idx_u[hetero]
        idx_v  = idx_v[hetero]

    acts = _compute_acts(pars, idx_u.size)
    people.contacts['f'] = _make_layer(idx_u, idx_v, acts)

    # --- accumulate stats ---
    n_nodes = int(uid.size)
    n_edges = int(state['edges_u'].size)
    mean_deg = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0
    f_net['_mean_degree_history'].append(mean_deg)
    f_net['_edge_count_history'].append(n_edges)
    f_net['_pop_size_history'].append(n_nodes)
    f_net['_timestep_history'].append(t)


# =====================================================================
# Diagnostics
# =====================================================================

def print_network_stats(f_net, t=None, verbose=True):
    """
    Print a summary of current network statistics.

    Can be called at any timestep.  When ``verbose=True`` (default)
    also prints the per-year mean degree history.

    Parameters
    ----------
    f_net   : dict returned by make_francesco_network
    t       : current timestep (optional, for display only)
    verbose : bool — if True, also print the year-by-year history
    """
    state  = f_net['state']
    w      = 60

    n_nodes = int(state['u_uid'].size)
    n_edges = int(state['edges_u'].size)
    mean_deg = (2.0 * n_edges / n_nodes) if n_nodes > 0 else 0.0

    label = f" t={t}" if t is not None else ""
    print("─" * w)
    print(f"Francesco network stats{label}")
    print("─" * w)
    print(f"  Active nodes : {n_nodes}")
    print(f"  Active edges : {n_edges}")
    print(f"  Mean degree  : {mean_deg:.4f}")

    hist_t   = f_net['_timestep_history']
    hist_deg = f_net['_mean_degree_history']
    hist_n   = f_net['_pop_size_history']

    if verbose and hist_t:
        print()
        print("  Year-by-year mean degree (monthly → annual mean):")
        print(f"  {'Year':>6}  {'Mean deg':>10}  {'Pop size':>10}")
        print("  " + "-" * 30)
        # group by year (12 steps per year)
        hist_t   = np.asarray(hist_t,   dtype=float)
        hist_deg = np.asarray(hist_deg, dtype=float)
        hist_n   = np.asarray(hist_n,   dtype=float)
        years    = np.unique((hist_t / 12).astype(int))
        for yr in years:
            mask = ((hist_t / 12).astype(int) == yr)
            print(f"  {yr:>6}  {hist_deg[mask].mean():>10.4f}"
                  f"  {hist_n[mask].mean():>10.1f}")
    print("─" * w)


def plot_network_stats(f_net):
    """
    Plot mean degree and population size over time.
    Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    hist_t   = np.asarray(f_net['_timestep_history'],   dtype=float)
    hist_deg = np.asarray(f_net['_mean_degree_history'], dtype=float)
    hist_n   = np.asarray(f_net['_pop_size_history'],    dtype=float)

    if hist_t.size == 0:
        print("No history recorded yet.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(hist_t / 12, hist_deg, lw=1.5, color='steelblue')
    ax1.set_ylabel("Mean degree")
    ax1.set_title("Francesco network — mean degree over time")
    ax1.grid(True, alpha=0.3)

    ax2.plot(hist_t / 12, hist_n, lw=1.5, color='darkorange')
    ax2.set_ylabel("Population size")
    ax2.set_xlabel("Year")
    ax2.set_title("Population size over time")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

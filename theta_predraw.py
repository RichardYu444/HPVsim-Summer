"""
theta_predraw.py
================

Make the community network's latent partner-formation propensity ``theta`` knowable BEFORE an
agent's sexual debut, so that an intervention can target the "core group" at vaccination age.

Why this file exists
--------------------
``theta`` is what defines the core group (see hpvsim_working/analysis.py's
``core_group_attribution``), but the network only draws it when an agent becomes sexually active
-- ``CommunityNetworkBackend._inject_arrivals()``, fired the timestep ``people.is_active`` flips
at ``age > debut``, with debut ~ N(16, 3.1). The NHS vaccination programme vaccinates at age
12-13. At that age essentially nobody has a theta yet, so nothing can identify the core group at
the moment of vaccination.

This module makes theta a deterministic function of ``(rand_seed, side, agent index)``, fixed
from birth and readable at any age, while the network still consumes exactly that same value at
debut. That turns "vaccinate the core group at 12-13" into a well-posed PERFECT-FORESIGHT
counterfactual: an upper bound on what targeting could achieve if the top decile could be
identified in advance.

IMPORTANT -- both arms of any comparison must import this module. The patch changes the theta
VALUES the network sees (though not its rng consumption -- see below), so a run with the patch is
not comparable against a run without it. Re-run the NHS_Vacc baseline under the patch rather than
reusing older outputs.

Installation idiom
------------------
``install()`` is called at MODULE IMPORT, exactly as powerlaw.py does with
``_install_powerlaw_theta()``. This is load-bearing, not stylistic: sciris parallelize spawns
workers on Windows that re-import ``__main__``, so the patch has to land via a module-level
import in the run script. A call inside ``main()`` would leave the workers unpatched and they
would silently draw a different theta from the parent.

What the patch does
-------------------
Two methods on ``CommunityNetworkBackend`` are wrapped:

* ``initialize(sim)`` -- stashes ``self._predraw_seed = sim['rand_seed']``. Needed because
  ``_inject_arrivals`` has no access to ``sim`` and so cannot reach the seed itself.
* ``_inject_arrivals(new_female, new_male, people)`` -- runs the original, then OVERWRITES the
  theta it just drew with the deterministic per-uid values, in both ``_theta_true_u/_v`` (the
  source of truth ``_refresh_annual_gate()`` rebuilds ``state['u_theta']`` from) and the tail of
  ``state['u_theta']/['v_theta']`` (each side's new rows are appended at the end of its own
  array, so ``[-n:]`` is exactly them). Injection happens before the monthly loop's
  ``_refresh_annual_gate()``, so writing raw un-gated theta here is correct -- new debuts are
  meant to stay ungated until the next 12-month boundary.

The original's own ``_sample_side_theta`` draws still happen and are discarded. That is
deliberate: rng CONSUMPTION is left identical to the unpatched backend, so only the theta values
move, not the rest of the network's random stream.

The t=0 population is not touched -- its theta comes from ``init_network_state()``, not from
``_inject_arrivals``. That cohort is irrelevant to vaccination (the youngest of them is over 40
by the time the programme starts in 2008), and ``theta_for_people()`` prefers the backend's
recorded value anyway, so nothing is inconsistent.
"""

import numpy as np

from hpvsim_working import community_network as hpcn


# Distinguishes this module's random stream from every other consumer of the same seed.
_SALT = np.uint64(0x9E3779B97F4A7C15)

SIDE_U = 0  # female side
SIDE_V = 1  # male side

_installed = False
_orig_initialize = None
_orig_inject = None


# =====================================================================
# 1. Deterministic per-agent theta
# =====================================================================

def _splitmix64(x):
    '''
    Vectorised splitmix64 finaliser: uint64 array -> well-mixed uint64 array. The wraparound on
    the multiplies IS the algorithm, so numpy's overflow warning is muted rather than avoided.
    '''
    with np.errstate(over='ignore'):
        z = np.asarray(x, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _uniforms(uids, side, seed):
    '''
    Deterministic U(0,1] independent of everything except (seed, side, uid). Counter-based, so
    an agent's draw does not depend on when or with whom it is requested -- which is the whole
    point: the value peeked at age 12 must equal the value the network stores at debut.
    '''
    key = _splitmix64(np.uint64(int(seed) * 2 + int(side)) ^ _SALT)
    h = _splitmix64(np.asarray(uids, dtype=np.uint64) ^ key)
    u = (h >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)  # [0, 1)
    return np.where(u <= 0.0, 2.0 ** -53, u)                    # -> (0, 1], safe for u**(-1/alpha)


def _is_pareto_sampler():
    ''' True if powerlaw.py's Pareto sampler is the one currently installed. '''
    return getattr(hpcn._sample_side_theta, '__name__', '') == '_sample_side_theta_powerlaw'


def theta_for(uids, side, shape, seed, floor=0.0):
    '''
    Theta for the given agent indices, deterministic in (seed, side, uid).

    Args:
        uids  (array): agent indices (== HPVsim uids)
        side    (int): SIDE_U for women, SIDE_V for men
        shape (float): the sampler's shape parameter -- Pareto tail index alpha under powerlaw.py,
                       Gamma shape under the stock sampler
        seed    (int): the sim's rand_seed
        floor (float): community_pars['theta_floor'], applied as floor + (1-floor)*raw

    Two paths. Under powerlaw.py's Pareto sampler the draw is inverted analytically and
    vectorised, which is ~1000x faster than building one Generator per agent and matters at
    200k agents x 50 runs. Under any other sampler it falls back to seeding a Generator per uid
    and calling whatever ``hpcn._sample_side_theta`` currently is, which is correct for every
    sampler but costs ~50 us/agent (~30 s per sim). The lookup is by attribute at call time, so
    powerlaw.py's monkeypatch is picked up whether it was installed before or after this module.
    '''
    uids = np.asarray(uids, dtype=np.int64)
    if uids.size == 0:
        return np.empty(0, dtype=float)
    floor = float(floor)

    if _is_pareto_sampler():
        # _sample_side_theta_powerlaw draws 1 + Lomax(alpha), i.e. classical Pareto(alpha, x_m=1)
        # with survival x**-alpha, so the inverse CDF is u**(-1/alpha) for u ~ U(0,1].
        alpha = float(shape)
        if alpha <= 2:
            raise ValueError('shape (Pareto tail index alpha) must be > 2 for finite variance')
        raw = _uniforms(uids, side, seed) ** (-1.0 / alpha)
    else:
        raw = np.empty(uids.size, dtype=float)
        for i, uid in enumerate(uids.tolist()):
            rng = np.random.default_rng([int(seed), int(_SALT), int(side), int(uid)])
            raw[i] = hpcn._sample_side_theta(1, float(shape), rng, floor=0.0)[0]

    if floor:
        raw = floor + (1.0 - floor) * raw
    return raw


def _backend_bits(sim):
    ''' (backend, seed, floor) -- the pieces theta_for() needs, pulled off the running sim. '''
    backend = sim.network_backend
    if backend is None or not hasattr(backend, '_theta_true_u'):
        bname = type(backend).__name__ if backend is not None else None
        errormsg = ("theta_predraw needs the 'community' network backend, which is what carries "
                    f"theta; got network={sim['network']!r} (backend={bname}).")
        raise ValueError(errormsg)
    seed = getattr(backend, '_predraw_seed', None)
    if seed is None:
        errormsg = ("This backend has no _predraw_seed -- theta_predraw.install() had not run by "
                    "the time the network was initialized. Import theta_predraw at MODULE level "
                    "in the run script (see this module's docstring).")
        raise RuntimeError(errormsg)
    return backend, seed, float(backend._params.get('theta_floor', 0.0))


def theta_for_people(sim, inds):
    '''
    Theta for arbitrary agents, at any age. Prefers the backend's recorded value when the agent
    has already debuted (which covers the t=0 cohort, whose theta came from init_network_state()
    rather than from the patched _inject_arrivals), and falls back to the predraw otherwise.
    For everyone injected at debut the two agree by construction.
    '''
    backend, seed, floor = _backend_bits(sim)
    inds = np.asarray(inds, dtype=np.int64)
    out = np.full(inds.size, np.nan, dtype=float)
    if inds.size == 0:
        return out

    female = sim.people.is_female[inds].astype(bool)
    for mask, side, dct, shape_key in (
            (female,  SIDE_U, backend._theta_true_u, 'gamma_shape_U'),
            (~female, SIDE_V, backend._theta_true_v, 'gamma_shape_V')):
        sel = inds[mask]
        if sel.size == 0:
            continue
        vals = theta_for(sel, side, backend._params[shape_key], seed, floor)
        # Already-debuted agents: use what the network actually holds
        for j, uid in enumerate(sel.tolist()):
            known = dct.get(uid)
            if known is not None:
                vals[j] = known
        out[mask] = vals
    return out


def core_cut(sim, frac):
    '''
    The theta threshold above which an agent is in the top ``frac`` of the population.

    Monte-Carlo'd off whichever sampler is installed rather than hardcoded, so this file does not
    have to know the distribution. Costs milliseconds. For Pareto alpha=3 and frac=0.10 it must
    land on 10**(1/3) = 2.1544, which is also where the 50-run empirical cuts in
    csvs/powerlaw_alpha3_200k_50runs_curves.npz sit (median 2.1542).
    '''
    backend, seed, floor = _backend_bits(sim)
    shape = backend._params['gamma_shape_U']
    rng = np.random.default_rng(20240905)  # fixed: the cut is a property of the model, not the run
    draws = hpcn._sample_side_theta(200_000, float(shape), rng, floor=floor)
    return float(np.quantile(np.asarray(draws, dtype=np.float64), 1.0 - float(frac)))


# =====================================================================
# 2. The patch
# =====================================================================

def _initialize_predraw(self, sim):
    self._predraw_seed = int(sim['rand_seed'])
    return _orig_initialize(self, sim)


def _inject_arrivals_predraw(self, new_female, new_male, people):
    _orig_inject(self, new_female, new_male, people)
    seed = getattr(self, '_predraw_seed', None)
    if seed is None:  # initialize() ran before the patch was installed; nothing sane to do
        return
    floor = float(self._params.get('theta_floor', 0.0))
    for uids, side, shape_key, theta_key, dct in (
            (new_female, SIDE_U, 'gamma_shape_U', 'u_theta', self._theta_true_u),
            (new_male,   SIDE_V, 'gamma_shape_V', 'v_theta', self._theta_true_v)):
        uids = np.asarray(uids)
        if uids.size == 0:
            continue
        th = theta_for(uids, side, self._params[shape_key], seed, floor)
        for uid, t in zip(uids.tolist(), th.tolist()):
            dct[uid] = t
        # Each side's new rows were just appended to the end of its own array by the original
        self._state[theta_key][-uids.size:] = th
    return


def install():
    ''' Idempotent. Called at import -- see the module docstring for why that matters. '''
    global _installed, _orig_initialize, _orig_inject
    if _installed:
        return
    _orig_initialize = hpcn.CommunityNetworkBackend.initialize
    _orig_inject = hpcn.CommunityNetworkBackend._inject_arrivals
    hpcn.CommunityNetworkBackend.initialize = _initialize_predraw
    hpcn.CommunityNetworkBackend._inject_arrivals = _inject_arrivals_predraw
    _installed = True
    return


install()


if __name__ == '__main__':
    # Standalone proof the deterministic draw has the right distribution and is stable per uid,
    # independent of any sim. Run under powerlaw.py so the Pareto path is the one exercised.
    import powerlaw  # noqa: F401  -- installs the Pareto sampler

    alpha = 3.0
    uids = np.arange(200_000)
    th = theta_for(uids, SIDE_U, alpha, seed=0)
    print(f'Pareto fast path, alpha={alpha}, n={uids.size}:')
    print(f'  min={th.min():.4f} (should be >= 1)  mean={th.mean():.4f} '
          f'(expected {alpha/(alpha-1):.4f})  max={th.max():.1f}')
    for q in (0.01, 0.05, 0.10, 0.20):
        print(f'  top {q:>5.1%} cut: empirical {np.quantile(th, 1-q):.4f}  '
              f'analytic {q**(-1/alpha):.4f}')

    # Same uid, same value, regardless of how it is asked for
    a = theta_for([7, 99, 4321], SIDE_U, alpha, seed=0)
    b = theta_for([4321], SIDE_U, alpha, seed=0)
    print(f'\n  stable per uid: {a[2]:.10f} == {b[0]:.10f} -> {a[2] == b[0]}')
    print(f'  side matters:   {theta_for([7], SIDE_V, alpha, seed=0)[0]:.4f} vs {a[0]:.4f}')
    print(f'  seed matters:   {theta_for([7], SIDE_U, alpha, seed=1)[0]:.4f} vs {a[0]:.4f}')

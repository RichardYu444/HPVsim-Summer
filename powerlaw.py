'''
powerlaw.py
===========

External sandbox for experimenting with the age+community bipartite network model
(hpvsim_working/age_community_bipartite_network_model.py, wired into HPVsim via
hpvsim_working/community_network.py's CommunityNetworkBackend) WITHOUT touching the
hpvsim_working package itself. Sibling of recreation.py (which does the same swap but to
Poisson) -- see that file's docstring for the general rationale; only the differences are
covered below.

Four changes relative to the package's own 'community' defaults:

1. Partner-propensity distribution: Gamma -> power-law (Pareto). The package draws each
   node's partner-formation propensity theta from a Gamma(shape, 1/shape) (mean 1,
   CV = 1/sqrt(shape)) via age_community_bipartite_network_model._sample_side_theta().
   That function is looked up as a plain module-global at every call site --
   init_network_state() and _append_side_births() inside the model module itself, and
   CommunityNetworkBackend._inject_arrivals() inside community_network.py (which imports
   the name directly at module load time: `from .age_community_bipartite_network_model
   import _sample_side_theta`). Monkeypatching BOTH modules' `_sample_side_theta`
   attribute below (_install_powerlaw_theta()) redirects every one of those call sites to
   a power-law (Pareto Type I, x_m=1) sampler with the same mean-1 contract -- no edits to
   hpvsim_working needed, since Python resolves module-global names through each module's
   own __dict__ at call time, not at `from ... import` time.

   Concretely, ``shape`` (still passed in as community_pars['gamma_shape'] -- the vendored
   model's parameter name is unchanged, it's just read as a different distribution's shape
   now) is the Pareto tail index alpha. Drawing L ~ Lomax(alpha) (numpy's rng.pareto) gives
   X = 1 + L ~ classical Pareto(alpha, x_m=1), with analytic mean alpha/(alpha-1) (for
   alpha > 1) and CV = 1/sqrt(alpha*(alpha-2)) (for alpha > 2, the regime this sampler
   requires so the propensity distribution has finite variance). Dividing by the analytic
   mean gives mean 1, the same contract the Gamma/Poisson samplers satisfy -- but with a
   genuinely heavy (power-law) right tail instead of Gamma's or Poisson's thin one, i.e. a
   small number of nodes end up with disproportionately many partners, closer to
   empirically-observed "core group" sexual-network structure than the package default.
   Smaller alpha (closer to 2) means a heavier tail / more extreme heterogeneity; larger
   alpha means it behaves more like the Gamma/Poisson case it replaces.

2. n_communities=1. With a single community, community_mixing collapses to the 1x1
   matrix [[1.0]] and every node lands in the same community -- i.e. the community layer
   is present but inert, equivalent to having no community concept at all.

3. Mirroring 'default' network's per-layer settings onto 's'/'l' (see
   _default_network_layer_pars() / MIRRORED_LAYER_PARS below) -- acts/age_act_pars/condoms
   and (approximately) partnership durations/mixing, exactly as recreation.py does. See
   recreation.py's docstring point 3 for the full rationale and caveats (duration shape,
   single mixing kernel, cross-layer coupling not reproduced).

4. m_partners/f_partners/dur_pship/layer_probs are forced to None before building the sim
   (see make_sim() below), NOT merely left unset. basePars.py's own base_pars now carries
   real 'c'/'m'-keyed values for these four (added when the *default* network's own poisson
   calibration was wired in -- see calibrate_default_poisson.py); those are placeholders
   for 'network'=='community' (never consumed by CommunityNetworkBackend -- see
   parameters.py's layer_defaults['community'] comment) but reset_layer_pars() merges
   *all* layer keys it finds (community's own 's'/'l' defaults plus whatever's already
   sitting in pars[pkey]), so leaving basePars.py's 'c'/'m' entries in place would silently
   produce 4-key m_partners/f_partners/dur_pship/layer_probs dicts (s, l, c, m) instead of
   the intended 2-key (s, l) ones. Passing None for all four lets reset_layer_pars() fall
   back cleanly to the 'community' layer_defaults.

The sim itself runs on the project's own basePars.base_pars (basePars.py, at the repo
root) as the base configuration -- only 'network', 'community_pars', the mirrored
'mixing'/'acts'/'age_act_pars'/'condoms', and the four nulled-out placeholders above are
overridden.
'''

import numpy as np
import sciris as sc

import hpvsim_working as hpv
import hpvsim_working.parameters as hpp
from hpvsim_working import age_community_bipartite_network_model as acbnm
from hpvsim_working import community_network as hpcn

from basePars import base_pars


# =====================================================================
# 1. Gamma -> power-law (Pareto) propensity swap
# =====================================================================

def _sample_side_theta_powerlaw(n, shape, rng, floor=0.0, exact_mean_one=False):
    '''
    Drop-in replacement for age_community_bipartite_network_model._sample_side_theta:
    same signature/contract (mean-1 propensities, optional floor, optional exact
    renormalisation) as the Gamma original and recreation.py's Poisson swap, but drawn
    from a classical Pareto(alpha, x_m=1) distribution instead -- a genuine power-law
    right tail (P(theta > x) ~ x^-alpha for large x) rather than Gamma's/Poisson's thin
    tail. ``shape`` is read as the Pareto tail index alpha; must be > 2 for the
    distribution to have finite variance (CV = 1/sqrt(alpha*(alpha-2))).
    '''
    if n <= 0:
        return np.empty(0, dtype=float)
    alpha = float(shape)
    if alpha <= 2:
        raise ValueError("shape (Pareto tail index alpha) must be > 2 for finite variance")
    # rng.pareto(alpha) draws from the Lomax (Pareto Type II) distribution; 1 + Lomax(alpha)
    # is the classical Pareto(alpha, x_m=1), with analytic mean alpha/(alpha-1).
    raw = 1.0 + rng.pareto(alpha, size=int(n))
    raw = raw * (alpha - 1.0) / alpha
    if exact_mean_one:
        mu = float(raw.mean())
        if mu > 0:
            raw = raw / mu
    if floor:
        raw = float(floor) + (1.0 - float(floor)) * raw
    return raw


def _install_powerlaw_theta():
    ''' Monkeypatch both modules' bound name for _sample_side_theta (see module docstring). '''
    acbnm._sample_side_theta = _sample_side_theta_powerlaw
    hpcn._sample_side_theta = _sample_side_theta_powerlaw
    return


_install_powerlaw_theta()


# =====================================================================
# 2. Pull 'default' network's own layer defaults straight from parameters.py, so
#    nothing here drifts out of sync if those defaults ever change.
# =====================================================================

def _default_network_layer_pars():
    ''' {'acts':, 'age_act_pars':, 'condoms':, 'dur_pship':, 'mixing':} for network='default' '''
    p = dict(network='default')
    hpp.reset_layer_pars(p, force=True)
    return dict(acts=p['acts'], age_act_pars=p['age_act_pars'], condoms=p['condoms'],
                dur_pship=p['dur_pship'], mixing=p['mixing'])


_DEFAULT = _default_network_layer_pars()

# Layer-key mapping used throughout: community's 's' (short) <-> default's 'c' (casual),
# community's 'l' (long) <-> default's 'm' (marital) -- both networks already document this
# same short/casual vs long/marital correspondence (see parameters.py's layer_defaults['community']
# comments).
MIRRORED_LAYER_PARS = dict(
    acts=dict(s=sc.dcp(_DEFAULT['acts']['c']), l=sc.dcp(_DEFAULT['acts']['m'])),
    age_act_pars=dict(s=sc.dcp(_DEFAULT['age_act_pars']['c']), l=sc.dcp(_DEFAULT['age_act_pars']['m'])),
    condoms=dict(s=_DEFAULT['condoms']['c'], l=_DEFAULT['condoms']['m']),
    # Single-kernel limitation (see recreation.py's docstring) -- both layers get default's
    # marital matrix.
    mixing=dict(s=_DEFAULT['mixing']['m'].copy(), l=_DEFAULT['mixing']['m'].copy()),
)

# hpu.sample()'s 'lognormal' and 'neg_binomial' both have mean == par1 (see utils.py's sample()
# docstring), and dur_pship's par1 is in years (matching network.py's `tind = sim.yearvec[t] -
# sim['start']` bookkeeping) while community_pars' D_mean_short/D_mean_long are in months (see
# parameters.py's community_pars docstring) -- hence the *12 conversion.
D_MEAN_SHORT_MONTHS = float(_DEFAULT['dur_pship']['c']['par1']) * 12.0
D_MEAN_LONG_MONTHS = float(_DEFAULT['dur_pship']['m']['par1']) * 12.0


# =====================================================================
# 3. Community network pars -- one community; duration means mirrored from 'default'
#    (see D_MEAN_SHORT_MONTHS/D_MEAN_LONG_MONTHS above). mean_partners_per_year/frac_long/
#    gamma_shape (now the Pareto alpha) have no 'default'-network equivalent to derive and
#    are left as tunable knobs -- see calibrate_community_powerlaw.py alongside this file.
# =====================================================================

COMMUNITY_PARS = dict(
    mean_partners_per_year=3.0,
    gamma_shape=3,          # now read as the Pareto tail index alpha -- see _sample_side_theta_powerlaw
    D_mean_short=D_MEAN_SHORT_MONTHS,
    D_mean_long=D_MEAN_LONG_MONTHS,
    frac_long=0.5,
    n_communities=1,
)


def make_sim(**overrides):
    pars = sc.mergedicts(base_pars, dict(
        network='community',
        community_pars=COMMUNITY_PARS,
        mixing=MIRRORED_LAYER_PARS['mixing'],
        acts=MIRRORED_LAYER_PARS['acts'],
        age_act_pars=MIRRORED_LAYER_PARS['age_act_pars'],
        condoms=MIRRORED_LAYER_PARS['condoms'],
        # Placeholders for the *default* network's own poisson calibration -- never consumed
        # by CommunityNetworkBackend, but must be nulled out here (not merely left unset) so
        # reset_layer_pars() doesn't merge basePars.py's leftover 'c'/'m' keys onto community's
        # own 's'/'l' defaults (see module docstring point 4).
        m_partners=None,
        f_partners=None,
        dur_pship=None,
        layer_probs=None,
    ), overrides)
    return hpv.Sim(pars)


if __name__ == '__main__':
    # Quick proof the power-law swap took effect, independent of the sim: draw directly from
    # the now-patched sampler and show the values are heavy-tailed (max >> mean).
    rng_check = np.random.default_rng(0)
    shape = COMMUNITY_PARS['gamma_shape']
    theta_check = acbnm._sample_side_theta(2000, shape, rng_check, exact_mean_one=True)
    print(f"sample thetas from the patched sampler (alpha={shape}), n=2000:")
    print(f"  mean={theta_check.mean():.3f}  max={theta_check.max():.3f}  "
          f"frac>3x mean={float((theta_check > 3).mean()):.3f}")

    print(f"\nmirrored from 'default': acts={MIRRORED_LAYER_PARS['acts']}")
    print(f"                          age_act_pars={MIRRORED_LAYER_PARS['age_act_pars']}")
    print(f"                          condoms={MIRRORED_LAYER_PARS['condoms']}")
    print(f"                          D_mean_short={D_MEAN_SHORT_MONTHS} months, "
          f"D_mean_long={D_MEAN_LONG_MONTHS} months")

    sim = make_sim(verbose=0.1)
    sim.run()
    sim.plot()

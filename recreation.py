'''
recreation.py
==============

External sandbox for experimenting with the age+community bipartite network model
(hpvsim_working/age_community_bipartite_network_model.py, wired into HPVsim via
hpvsim_working/community_network.py's CommunityNetworkBackend) WITHOUT touching the
hpvsim_working package itself.

Three changes relative to the package's own 'community' defaults:

1. Partner-propensity distribution: Gamma -> Poisson. The package draws each node's
   partner-formation propensity theta from a Gamma(shape, 1/shape) (mean 1,
   CV = 1/sqrt(shape)) via age_community_bipartite_network_model._sample_side_theta().
   That function is looked up as a plain module-global at every call site --
   init_network_state() and _append_side_births() inside the model module itself, and
   CommunityNetworkBackend._inject_arrivals() inside community_network.py (which
   imports the name directly at module load time: `from
   .age_community_bipartite_network_model import _sample_side_theta`). Monkeypatching
   BOTH modules' `_sample_side_theta` attribute below (_install_poisson_theta())
   redirects every one of those call sites to a Poisson(lambda=shape)/shape sampler
   (mean 1, same CV = 1/sqrt(shape) functional form as the Gamma it replaces) -- no
   edits to hpvsim_working needed, since Python resolves module-global names through
   each module's own __dict__ at call time, not at `from ... import` time.

2. n_communities=1. With a single community, community_mixing collapses to the 1x1
   matrix [[1.0]] and every node lands in the same community -- i.e. the community
   layer is present but inert, equivalent to having no community concept at all.

3. Mirroring 'default' network's per-layer settings onto 's'/'l' (see
   _default_network_layer_pars() / MIRRORED_LAYER_PARS below). CommunityNetworkBackend
   forms/dissolves partnerships itself (community_pars), but acts/age_act_pars/condoms
   are read directly off sim['acts']/sim['age_act_pars']/sim['condoms'] by the
   transmission loop (sim.py) and _build_contacts_dict() (community_network.py)
   regardless of which backend created the contact -- those three are genuinely
   shared and are mirrored exactly, layer-for-layer ('c'->'s', 'm'->'l'). Two things
   can only be *approximated*, not mirrored exactly, because of structural
   differences between the two backends' partnership-formation mechanisms:

   * Duration. 'default' draws each partnership's duration from a full distribution
     (dur_pship: lognormal for casual, neg_binomial for marital) each with its own
     shape; CommunityNetworkBackend instead enforces a single memoryless-exponential
     hazard per type (q_short/q_long, i.e. D_mean_short/D_mean_long in community_pars,
     in months). Only the *mean* can be matched, not the shape/skew. Both
     hpu.sample()'s 'lognormal' and 'neg_binomial' have mean == par1 (see
     utils.sample()'s docstring), so D_mean_short/D_mean_long here are just
     dur_pship['c'/'m']['par1'] converted from years (dur_pship's units, matching
     'default's own tind/end-date bookkeeping in network.py) to months (community's
     unit, per community_pars' own docstring in parameters.py).

   * Mixing. CommunityNetworkBackend reads only sim['mixing']['s'] and applies that
     SAME kernel to both short and long partnerships (see community_network.py's
     module docstring point 1) -- there is no way to give it two different kernels
     the way 'default' has separate 'm' and 'c' matrices. We mirror 'default's 'm'
     (marital/assortative) matrix, since that's already CommunityNetworkBackend's own
     built-in fallback (get_mixing()'s 'community' branch) -- so this is made
     explicit here rather than actually changing behaviour.

   mean_partners_per_year, frac_long and the propensity-heterogeneity parameters
   (gamma_shape/theta_floor) are NOT derived from 'default' here: unlike
   acts/age_act_pars/condoms/duration, there's no direct read of an equivalent
   'default' parameter to convert -- they're emergent outcomes of 'default's
   layer_probs/m_partners/f_partners/cross_layer machinery, which would require
   actually running a reference 'default' sim and measuring realised degree/duration
   statistics to calibrate against (out of scope here; left as tunable knobs).

   Cross-layer coupling (f_cross_layer/m_cross_layer) is NOT mirrored -- it has no
   equivalent in CommunityNetworkBackend's continuous propensity-driven formation
   (see community_network.py's module docstring point 1); reproducing it would
   require adding actual gating logic to community_network.py itself, not just
   passing arguments.

The sim itself runs on the project's own basePars.base_pars (basePars.py, at the repo
root) as the base configuration -- only 'network', 'community_pars', and the mirrored
'mixing'/'acts'/'age_act_pars'/'condoms' are overridden.
'''

import numpy as np
import sciris as sc

import hpvsim_working as hpv
import hpvsim_working.parameters as hpp
from hpvsim_working import age_community_bipartite_network_model as acbnm
from hpvsim_working import community_network as hpcn

from basePars import base_pars


# =====================================================================
# 1. Gamma -> Poisson propensity swap
# =====================================================================

def _sample_side_theta_poisson(n, shape, rng, floor=0.0, exact_mean_one=False):
    '''
    Drop-in replacement for age_community_bipartite_network_model._sample_side_theta:
    same signature/contract (mean-1 propensities, optional floor, optional exact
    renormalisation), but drawn from Poisson(lambda=shape)/shape instead of
    Gamma(shape, 1/shape). Poisson(lambda) has mean=lambda and variance=lambda, so
    dividing by shape=lambda gives mean 1 and CV = 1/sqrt(shape) -- the same
    functional form as the Gamma shape parameter it replaces, just discrete instead
    of continuous.
    '''
    if n <= 0:
        return np.empty(0, dtype=float)
    lam = float(shape)
    if lam <= 0:
        raise ValueError("shape (Poisson lambda) must be > 0")
    raw = rng.poisson(lam, size=int(n)).astype(float) / lam
    if exact_mean_one:
        mu = float(raw.mean())
        if mu > 0:
            raw = raw / mu
    if floor:
        raw = float(floor) + (1.0 - float(floor)) * raw
    return raw


def _install_poisson_theta():
    ''' Monkeypatch both modules' bound name for _sample_side_theta (see module docstring). '''
    acbnm._sample_side_theta = _sample_side_theta_poisson
    hpcn._sample_side_theta = _sample_side_theta_poisson
    return


_install_poisson_theta()


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
    # Single-kernel limitation (see module docstring) -- both layers get default's marital matrix.
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
#    gamma_shape have no 'default'-network equivalent to derive (see module docstring) and
#    are left as tunable knobs.
# =====================================================================

COMMUNITY_PARS = dict(
    mean_partners_per_year=3.0,
    gamma_shape=3,          # now read as the Poisson lambda -- see _sample_side_theta_poisson
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
    ), overrides)
    return hpv.Sim(pars)


if __name__ == '__main__':
    # Quick proof the Poisson swap took effect, independent of the sim: draw directly from the
    # now-patched sampler and show the values sit on the expected 1/shape grid.
    rng_check = np.random.default_rng(0)
    shape = COMMUNITY_PARS['gamma_shape']
    theta_check = acbnm._sample_side_theta(10, shape, rng_check)
    print(f"sample thetas from the patched sampler (shape={shape}): {np.round(theta_check, 3)}")
    print(f"  -> values * shape are integers: {np.allclose(theta_check * shape, np.round(theta_check * shape))}")

    print(f"\nmirrored from 'default': acts={MIRRORED_LAYER_PARS['acts']}")
    print(f"                          age_act_pars={MIRRORED_LAYER_PARS['age_act_pars']}")
    print(f"                          condoms={MIRRORED_LAYER_PARS['condoms']}")
    print(f"                          D_mean_short={D_MEAN_SHORT_MONTHS} months, "
          f"D_mean_long={D_MEAN_LONG_MONTHS} months")

    sim = make_sim(verbose=0.1)
    sim.run()
    sim.plot()

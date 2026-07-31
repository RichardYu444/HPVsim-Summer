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
   theta = 1 + L ~ classical Pareto(alpha, x_m=1): a hard FLOOR of exactly 1 (nobody has
   below-average propensity purely from an unlucky draw) and analytic mean alpha/(alpha-1)
   (for alpha > 1), CV = 1/sqrt(alpha*(alpha-2)) about that mean (for alpha > 2, the regime
   this sampler requires so the propensity distribution has finite variance).
   Deliberately NOT rescaled to force mean 1 (unlike the Gamma/Poisson samplers it
   replaces). IMPORTANT -- this is a presentational choice ONLY, with no effect on the
   simulated network: theta's absolute scale is invisible to the model. _sample_edges draws
   its total edge count as Poisson(scale * sum(theta_u) * sum(theta_v)) and allocates
   endpoints in strict proportion to theta, and acbnm.calibrate() then empirically rescales
   rho until realised degree hits target -- so multiplying every theta by a constant is
   exactly absorbed by rho shrinking by that constant squared. This was verified on
   2026-07-30 by rescaling theta with a compensating rho and obtaining BIT-IDENTICAL edge
   sets. `1 + Lomax(alpha)` and `(1 + Lomax(alpha)) * (alpha-1)/alpha` are the same
   distribution up to scale, so switching between them changes nothing; the floor-of-1 form
   is simply easier to reason about. Only theta's SHAPE (set by alpha, or by theta_floor,
   which is affine and therefore genuinely not scale-invariant) affects the degree
   distribution. Singleness is NOT controlled here at all -- it is set by p_single_annual
   plus the guaranteed-pairing pass (see community_pars below and
   CommunityNetworkBackend._force_pair_ungated() in community_network.py). Smaller alpha
   (closer to 2) means a heavier tail and more extreme heterogeneity; larger alpha means
   less heterogeneity.

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

from basePars import base_pars, married_matrix, casual_matrix


# =====================================================================
# 1. Gamma -> power-law (Pareto) propensity swap
# =====================================================================

def _sample_side_theta_powerlaw(n, shape, rng, floor=0.0, exact_mean_one=False):
    '''
    Drop-in replacement for age_community_bipartite_network_model._sample_side_theta, same
    signature as the Gamma original and recreation.py's Poisson swap, but "power law + 1":
    theta = 1 + Lomax(alpha), i.e. classical Pareto(alpha, x_m=1) -- a hard floor of exactly
    1 and analytic mean alpha/(alpha-1) > 1, CV = 1/sqrt(alpha*(alpha-2)) about that mean.
    ``shape`` is read as the Pareto tail index alpha; must be > 2 for finite variance.

    Deliberately NOT renormalised to mean 1 (unlike the Gamma/Poisson samplers it replaces),
    but note this is presentational only -- theta's absolute scale is invisible to the model
    and is absorbed by acbnm.calibrate()'s rho rescaling. See the module docstring's section
    1 for the full argument and the bit-identical-edge-set verification.

    IMPORTANT: ``exact_mean_one`` is accepted only for signature compatibility with the
    vendored _sample_side_theta contract -- init_network_state() calls this with
    exact_mean_one=True for the t=0 population, while CommunityNetworkBackend's
    _inject_arrivals() calls without it for later debuts. It is DELIBERATELY IGNORED here:
    honouring it would silently renormalise the t=0 cohort back toward mean 1 while later
    debuts stayed unnormalised -- exactly the inconsistency this "+1" design must avoid. Do
    NOT reintroduce a ``raw = raw / raw.mean()`` step here regardless of what's passed.
    '''
    if n <= 0:
        return np.empty(0, dtype=float)
    alpha = float(shape)
    if alpha <= 2:
        raise ValueError("shape (Pareto tail index alpha) must be > 2 for finite variance")
    # rng.pareto(alpha) draws from the Lomax (Pareto Type II) distribution; 1 + Lomax(alpha)
    # is the classical Pareto(alpha, x_m=1): floor exactly 1, analytic mean alpha/(alpha-1).
    raw = 1.0 + rng.pareto(alpha, size=int(n))
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
# 2b. Age-mixing kernel, supplied EXPLICITLY rather than left to the backend's default of
#     reading sim['mixing']['s'].
#
#     Two problems with the default path, both found on 2026-07-30:
#
#     (i) Structural exclusion of the young. CommunityNetworkBackend supports only ONE
#         age-mixing kernel and reads it from the 's' slot, which basePars assigns
#         married_matrix. That matrix's 15-20 and 20-25 bands are near-zero, so those
#         cohorts were left 100% / 98% permanently single -- the worst possible age group
#         to disconnect in an HPV model, since that's where incidence peaks.
#         casual_matrix, by contrast, has strong young-age coverage.
#     (ii) Calibration/validation mismatch. make_sim() below overrides 'mixing' with
#         MIRRORED_LAYER_PARS, i.e. HPVsim's PACKAGE-default marital matrix, whereas
#         basePars_community_powerlaw.py supplies the project's own married_matrix -- so
#         knobs were being calibrated against a different network than they were validated
#         on. Supplying age_mixing/age_band_edges through community_pars (which the backend
#         honours as an override when BOTH keys are present) makes sim['mixing'] irrelevant
#         to the network and so removes the mismatch at the source.
#
#     Since only one kernel is supported but the network carries both casual-like ('s') and
#     marital-like ('l') ties, blend the two matrices. The weight is a pragmatic fixed
#     constant rather than something derived from frac_long/p_form_long: those are
#     calibration knobs, and deriving the kernel from them would make the mixing structure a
#     moving target between calibration iterations.
# =====================================================================

MARRIED_BLEND_WEIGHT = 0.5  # 1.0 = married_matrix only (excludes the young), 0.0 = casual only

AGE_BAND_EDGES, _A_MARRIED = hpcn._age_mixing_from_hpvsim_matrix(married_matrix)
_, _A_CASUAL = hpcn._age_mixing_from_hpvsim_matrix(casual_matrix)
AGE_MIXING = MARRIED_BLEND_WEIGHT * _A_MARRIED + (1.0 - MARRIED_BLEND_WEIGHT) * _A_CASUAL


def check_age_mixing_coverage(A=AGE_MIXING, band_edges=AGE_BAND_EDGES, min_age=16.0, tol=1e-6):
    '''
    Fail loudly if any age band that real sexually-active agents actually occupy has
    effectively no mixing weight -- such a band can never be selected as an edge endpoint,
    so everyone in it is permanently single by construction rather than by chance. Returns
    the list of offending (band_index, lower_edge) pairs; raises if any are found.

    Bands entirely below `min_age` are skipped: nobody is sexually active there, so their
    zero weight is correct rather than a defect.
    '''
    A = np.asarray(A, dtype=float)
    lower = np.concatenate([[0.0], np.asarray(band_edges, dtype=float)])
    weight = A.sum(axis=1) + A.sum(axis=0)  # Row (as one sex) + column (as the other)
    upper = np.concatenate([np.asarray(band_edges, dtype=float), [np.inf]])
    offenders = [(int(i), float(lower[i])) for i in range(A.shape[0])
                 if upper[i] > min_age and weight[i] <= tol]
    if offenders:
        errormsg = (f"age-mixing kernel leaves sexually-active age band(s) "
                    f"{offenders} with no partnering weight -- everyone in them would be "
                    f"structurally single. Check MARRIED_BLEND_WEIGHT / the source matrices.")
        raise ValueError(errormsg)
    return offenders


check_age_mixing_coverage()


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
    # Annual singleness control -- see community_network.py's module docstring point 4 and
    # _force_pair_ungated(). Each year an independent 20% are blocked from forming new ties
    # and the remaining 80% who hold none are force-paired, so singleness is a controlled
    # input rather than a Poisson residual. NOTE realised p_single comes out somewhat BELOW
    # this value, because gated people who already hold a partnership keep it (the gate is
    # formation-only by design) and so don't count as single. Set to 0.0 to disable.
    p_single_annual=0.20,
    # Explicit age-mixing kernel (see section 2b) -- both keys must be present for the
    # backend to use them instead of reading sim['mixing']['s'].
    age_mixing=AGE_MIXING,
    age_band_edges=AGE_BAND_EDGES,
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
    expected_mean = shape / (shape - 1.0)
    print(f"sample thetas from the patched sampler (alpha={shape}), n=2000:")
    print(f"  min={theta_check.min():.3f} (should be >= 1.0)  "
          f"mean={theta_check.mean():.3f} (expected ~{expected_mean:.3f})  "
          f"max={theta_check.max():.3f}  frac>3x mean={float((theta_check > 3).mean()):.3f}")

    print(f"\nmirrored from 'default': acts={MIRRORED_LAYER_PARS['acts']}")
    print(f"                          age_act_pars={MIRRORED_LAYER_PARS['age_act_pars']}")
    print(f"                          condoms={MIRRORED_LAYER_PARS['condoms']}")
    print(f"                          D_mean_short={D_MEAN_SHORT_MONTHS} months, "
          f"D_mean_long={D_MEAN_LONG_MONTHS} months")

    sim = make_sim(verbose=0.1)
    sim.run()
    sim.plot()

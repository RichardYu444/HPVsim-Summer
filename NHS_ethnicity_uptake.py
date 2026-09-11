"""
NHS_ethnicity_uptake.py
=======================

NHS_Vacc.py + NHS_2025_lambdamu.py, with a PER-ETHNICITY UPTAKE SETTING applied to every
vaccination and screening acceptance probability.

    import NHS_ethnicity_uptake as eth        # at MODULE level -- see "Installation idiom"
    eth.set_multipliers(vaccination=dict(White=1.0, Asian=0.85, Black=0.80, Chinese=0.95),
                        screening=dict(White=1.0, Asian=0.75, Black=0.85, Chinese=0.70))
    sim = hpv.Sim(base_pars, interventions=eth.get_interventions(l=1, m=1))

``get_interventions()`` returns the screening pathway AND the vaccination programme in one list,
so it is a drop-in replacement for basePars_community.py's

    NHS_2025_lambdamu.get_interventions(l=1, m=1) + NHS_Vacc.vaccinations

Nothing about the two programmes is re-specified here. Every intervention is CLONED from the
objects in those two files -- same products, years, ages, sexes, eligibility functions, labels and
probabilities -- and only the acceptance step is intercepted. So NHS_Vacc.py and
NHS_2025_lambdamu.py stay the single source of truth for the programmes themselves: change a
coverage figure there and it changes here too.


What "uptake by ethnicity" means here
-------------------------------------
Ethnicity is the community network's community tag (basePars_community.ETHNICITIES =
['White', 'Asian', 'Black', 'Chinese'], drawn from community_probs and fixed for life -- see
hpvsim_working/community_network.py's module docstring, point 2). It already drives who partners
whom; this file makes it drive who accepts an offer as well.

The setting is a MULTIPLIER per ethnicity, not an absolute uptake, because the underlying
probabilities are not single numbers: first-dose coverage moves year by year through the
programme (0.86 in 2008 ... 0.565 in 2020 ... 0.80 from 2022), and primary screening uptake
differs by age band. An agent of ethnicity e offered a service with England-wide probability p
accepts with probability

    clip(p * m_e, 0, 1)

so m_e = 1 reproduces NHS_Vacc/NHS_2025 exactly -- verified, not asserted: check_null_equivalence()
runs both and compares, and they come out identical to the last bit -- and the relative gradient
between groups is held fixed across every programme year. Both multiplier sets default to 1.0 for
every ethnicity: the defaults in this file are a NULL, and the numbers to put in them are yours to
supply. The values in the __main__ demo are made up to exercise the machinery and are not
estimates of anything.

p above is the probability ON THE SCALE IT WAS QUOTED ON: the ANNUAL first-dose coverage for
vaccination, the per-offer probability for screening. HPVsim has already converted the annual one
to a per-timestep 1-(1-p)**dt by the time an intervention runs, and scaling that instead quietly
flattens the gradient (0.75 x the per-timestep form of 0.80 annualises back to 0.68, not 0.60),
so _per_agent_probs() undoes the conversion, applies the multiplier, and redoes it.
check_annual_scale() pins this down exactly, with no simulation involved.

RENORMALISE (below) optionally rescales the multipliers so that the population-weighted mean is
1, which turns the setting into a pure redistribution that leaves overall national uptake at the
calibrated level -- the analogue of core_vacc.py holding the total dose count fixed. Off by
default, since the natural reading of "Black uptake is 0.8" is 0.8 of the national rate, not 0.8
after everyone else has been scaled up to compensate.

Which probabilities it touches, and which it does not:

    vaccination     first doses (all four programme eras)          always
                    second/third doses                             VACC_APPLY_TO_LATER_DOSES
    screening       routine_screening_under50 / _50andover         always
                    second/third_consecutive_screening             APPLY_TO_FOLLOWUP_SCREENS
                    cytology triage / colposcopy attendance        APPLY_TO_TRIAGE
    treatment       ablation, general cancer treatment             never (out of scope)

Second and third doses are conditional continuation probabilities, not fresh decisions to enter
the programme, so by default the differential applies once at entry rather than compounding over
the course (a group at m=0.8 would otherwise end up at ~0.8**2.4 of full-course coverage). Turn
VACC_APPLY_TO_LATER_DOSES on if you mean to model differential completion as well.


Ethnicity before sexual debut (the patch)
-----------------------------------------
This is the same obstacle core_vacc.py hit with theta, and it is solved the same way.

The network only assigns a community when an agent enters it, i.e. at sexual debut
(CommunityNetworkBackend._inject_arrivals(), debut ~ N(16, 3.1)). Vaccination happens at age
12-13, when essentially every agent still has people.community == -1. So on the stock backend a
vaccination intervention cannot see ethnicity at all.

Importing this module patches the backend so that community is a deterministic function of
(rand_seed, agent index), fixed from birth and readable at any age, with the network then
consuming exactly that value at debut. The marginal distribution is unchanged (the same
community_probs, inverted analytically instead of sampled), and the original rng.choice() draw
still happens and is discarded, so the backend's random stream is consumed identically -- but the
community VALUES the network sees do change, and with them which blocks of the mixing matrix a
given agent lands in.

    ==> BOTH ARMS OF ANY COMPARISON MUST IMPORT THIS MODULE. <==

A run under the patch is not comparable against an older NHS_Vacc/NHS_2025 output produced
without it. The control arm is this module with all multipliers left at 1.0, which is
byte-for-byte the same programme on the same patched network -- that is exactly why the defaults
are 1.0 rather than an illustrative gradient.

The t=0 population is not touched (its communities come from init_network_state(), and the
lookup below prefers the backend's recorded value for anyone who has already debuted, so nothing
is inconsistent). Screening never needs the predraw at all -- everyone eligible for screening is
past debut and so already carries a real community -- but the patch is installed unconditionally
anyway, so that a screening-only run and a vaccination run share one network realisation.

On a sim with no community network (basePars.py's default network), there are no communities to
read; the lookup falls back to assigning ethnicity deterministically from POP_SHARES, so the file
still runs and the uptake gradient still applies. Nothing else in such a sim knows about
ethnicity, so it is a uptake-only counterfactual there.

Installation idiom
------------------
install() runs at MODULE IMPORT, exactly as theta_predraw.py and powerlaw.py do, and for the same
reason: sciris parallelize spawns workers on Windows that re-import __main__, so a patch applied
inside main() would leave every worker unpatched and silently drawing different communities from
the parent. Import this module at module level in the run script, and set the multipliers either
from the environment (below) or with a module-level set_multipliers() call.

    ETH_VACC_MULT=1,0.85,0.80,0.95 ETH_SCREEN_MULT=1,0.75,0.85,0.70 python run_my_sweep.py

is the safest route for a sweep: the environment survives process spawn where a value assigned in
main() would not.

A caveat worth carrying into the write-up: community_probs in basePars_community.py are
PARTNERSHIP-END shares (91.2 / 4.5 / 3.6 / 0.7), not census population shares (86.0 / 9.1 / 4.2 /
0.7) -- see the comment above eth_partner_end_shares there. Agents are tagged from those, so the
minority groups here are smaller than England's, and a coverage gap in one of them moves the
national total less than it really would.
"""

import os

import numpy as np
import sciris as sc

import hpvsim_working as hpv
from hpvsim_working import community_network as hpcn
from hpvsim_working import interventions as hpi

import basePars_community            # ETHNICITIES and the network's own community_probs
import NHS_Vacc                      # the vaccination programme cloned below
import NHS_2025_lambdamu as NHS      # the screening pathway cloned below


# =====================================================================
# 1. Settings
# =====================================================================

# Ethnicity order is the network's community order -- community 0 is ETHNICITIES[0] and so on.
# Taken from basePars_community rather than restated, so the two cannot drift apart.
ETHNICITIES = list(basePars_community.ETHNICITIES)
N_ETH = len(ETHNICITIES)

# Population weights, used only by RENORMALISE and by the fallback assignment on a sim with no
# community network. These are the network's community_probs, i.e. partnership-end shares -- see
# the caveat at the end of the module docstring. Census shares for the same four groups,
# renormalised, are [0.860, 0.091, 0.042, 0.007].
POP_SHARES = np.asarray(basePars_community.community_probs, dtype=float)
POP_SHARES = POP_SHARES / POP_SHARES.sum()


def _mult_from_env(varname):
    '''
    Multipliers from an environment variable, as N_ETH comma-separated numbers in ETHNICITIES
    order, e.g. ETH_VACC_MULT=1,0.85,0.80,0.95. Environment rather than a constant edited here
    because sciris parallelize re-imports this module in each spawned Windows worker: the
    environment is inherited, an assignment made in a run script's main() is not (same reasoning
    as core_vacc.CORE_FRAC). Unset => no differentiation.
    '''
    raw = os.environ.get(varname)
    if not raw:
        return {e: 1.0 for e in ETHNICITIES}
    vals = [float(x) for x in raw.replace(';', ',').split(',') if x.strip() != '']
    if len(vals) != N_ETH:
        errormsg = (f'{varname} must give {N_ETH} numbers in the order {ETHNICITIES}, '
                    f'got {len(vals)}: {raw!r}')
        raise ValueError(errormsg)
    return dict(zip(ETHNICITIES, vals))


# THE SETTING. Relative uptake by ethnicity; 1.0 everywhere = the NHS_Vacc/NHS_2025 programmes
# unchanged. Deliberately left as a null default -- put your own estimates in here, or set them
# from the environment / set_multipliers() rather than editing this file, so that one checkout
# can run every arm of a sweep.
VACCINATION_MULTIPLIERS = _mult_from_env('ETH_VACC_MULT')
SCREENING_MULTIPLIERS   = _mult_from_env('ETH_SCREEN_MULT')

# Rescale the multipliers so sum_e POP_SHARES[e] * m_e == 1, i.e. hold national uptake at the
# calibrated level and make the setting a pure redistribution between groups. Off by default:
# with it off, m_e = 0.8 means "80% of the national rate", which is how uptake ratios are usually
# quoted. With it on, the same input means "80% of the national rate RELATIVE to the others",
# and the majority group is scaled up to pay for it.
RENORMALISE = bool(int(os.environ.get('ETH_RENORMALISE', 0)))

# Which stages the differential reaches. See the table in the module docstring.
VACC_APPLY_TO_LATER_DOSES = bool(int(os.environ.get('ETH_LATER_DOSES', 0)))
APPLY_TO_FOLLOWUP_SCREENS = bool(int(os.environ.get('ETH_FOLLOWUP_SCREENS', 1)))
APPLY_TO_TRIAGE           = bool(int(os.environ.get('ETH_TRIAGE', 0)))

# Labels of the screening-pathway interventions in NHS_2025_lambdamu.py, split by stage. These
# are matched against the labels of the objects that file's get_interventions() hands back, so
# they must stay in step with it.
PRIMARY_SCREEN_LABELS  = ('routine_screening_under50', 'routine_screening_50andover')
FOLLOWUP_SCREEN_LABELS = ('second_consecutive_screening', 'third_consecutive_screening')
TRIAGE_LABELS          = ('first_cytology', 'second_cytology', 'third_cytology', 'colposcopy')


# =====================================================================
# 2. Ethnicity known from birth (the backend patch)
# =====================================================================

# Distinguishes this module's random stream from every other consumer of the same seed. The
# splitmix helpers below are duplicated from theta_predraw.py rather than imported: importing
# that module would install ITS patch too, silently changing theta as well as ethnicity.
_SALT = np.uint64(0x243F6A8885A308D3)

_installed = False
_orig_initialize = None
_orig_inject = None


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


def _uniforms(uids, seed):
    '''
    Deterministic U(0,1] independent of everything except (seed, uid). Counter-based, so an
    agent's draw does not depend on when or with whom it is requested -- which is the whole point:
    the ethnicity read off at age 12 must equal the one the network stores at debut.
    '''
    key = _splitmix64(np.uint64(int(seed)) ^ _SALT)
    h = _splitmix64(np.asarray(uids, dtype=np.uint64) ^ key)
    u = (h >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)  # [0, 1)
    return np.where(u <= 0.0, 2.0 ** -53, u)                    # -> (0, 1]


def ethnicity_for(uids, seed, probs):
    '''
    Community index for the given agent indices, deterministic in (seed, uid), distributed
    exactly as ``probs``. Inverse-CDF on a counter-based uniform, so it is the same answer at any
    age and in any order -- the analytic equivalent of the backend's rng.choice(n_comm, p=probs).
    '''
    uids = np.asarray(uids, dtype=np.int64)
    if uids.size == 0:
        return np.empty(0, dtype=np.int64)
    cum = np.cumsum(np.asarray(probs, dtype=float))
    cum[-1] = 1.0  # guard against float drift leaving the last bin unreachable
    out = np.searchsorted(cum, _uniforms(uids, seed), side='left')
    return np.clip(out, 0, len(cum) - 1).astype(np.int64)


def _seed_of(sim):
    seed = sim['rand_seed']
    return 0 if seed is None else int(seed)


def _community_probs_of(sim):
    '''
    The community_probs actually in force, taken from the running backend where there is one (it
    validates/normalises them) and from POP_SHARES otherwise.
    '''
    backend = getattr(sim, 'network_backend', None)
    params = getattr(backend, '_params', None)
    if params is not None and params.get('community_probs') is not None:
        return np.asarray(params['community_probs'], dtype=float)
    return POP_SHARES


def ethnicity_of(sim, inds):
    '''
    Ethnicity (as a community index) for arbitrary agents, at any age.

    Prefers what the sim actually holds -- people.community, set at network entry -- and falls
    back to the deterministic predraw for anyone who has not entered the network yet, which at
    the 12-13 vaccination window is almost everyone. For agents injected at debut under the patch
    the two agree by construction; the agents where they differ are the t=0 cohort, and multiscale
    cancer agents, which are clones of an existing agent and inherit that agent's ethnicity along
    with the rest of its state. Both are already network members, so both take the first branch
    and keep the ethnicity they actually have.
    '''
    inds = np.asarray(inds, dtype=np.int64)
    if inds.size == 0:
        return np.empty(0, dtype=np.int64)
    probs = _community_probs_of(sim)
    if len(probs) != N_ETH:
        errormsg = (f"This sim has {len(probs)} communities but ETHNICITIES has {N_ETH} entries "
                    f"({ETHNICITIES}). The uptake multipliers are indexed by community, so the "
                    f"two must match -- check community_pars['n_communities'].")
        raise ValueError(errormsg)

    comm = np.asarray(sim.people.community[inds], dtype=np.int64)
    unknown = comm < 0
    if unknown.any():
        comm[unknown] = ethnicity_for(inds[unknown], _seed_of(sim), probs)
    return comm


def _initialize_predraw(self, sim):
    self._eth_predraw_seed = _seed_of(sim)
    return _orig_initialize(self, sim)


def _inject_arrivals_predraw(self, new_female, new_male, people):
    '''
    Run the original injection, then overwrite the community tag it just drew with the
    deterministic one -- in people.community, in the backend's own state, and in the derived
    age/community block. The block matters: _refresh_bands() recomputes it from state[*_comm] on
    every later step, but it runs BEFORE injection within a step, so this step's partnership
    formation would otherwise use the discarded community's block.
    '''
    _orig_inject(self, new_female, new_male, people)
    seed = getattr(self, '_eth_predraw_seed', None)
    if seed is None:  # initialize() ran before the patch was installed; nothing sane to do
        return
    state, params = self._state, self._params
    probs, n_bands = params['community_probs'], params['n_bands']
    for uids, comm_key, band_key, block_key in (
            (new_female, 'u_comm', 'u_band', 'u_block'),
            (new_male,   'v_comm', 'v_band', 'v_block')):
        uids = np.asarray(uids)
        if uids.size == 0:
            continue
        comm = ethnicity_for(uids, seed, probs).astype(state[comm_key].dtype)
        people.community[uids] = comm
        # Each side's new rows were just appended to the end of its own arrays by the original
        state[comm_key][-uids.size:] = comm
        state[block_key][-uids.size:] = hpcn._block_of(comm, state[band_key][-uids.size:], n_bands)
    return


def install():
    ''' Idempotent. Called at import -- see "Installation idiom" in the module docstring. '''
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


# =====================================================================
# 3. Turning a national probability into a per-agent one
# =====================================================================

class EthnicityUptake:
    '''
    Resolves a national acceptance probability into one probability per agent, by ethnicity.

    A single instance is shared by every intervention it applies to, so set_multipliers() reaches
    interventions that were already built. MultiSim deep-copies the sim per run, so each run gets
    its own copy and there is no cross-run state.
    '''

    def __init__(self, multipliers, name, renormalise=None):
        self.name = name
        self.renormalise = RENORMALISE if renormalise is None else bool(renormalise)
        self._warned = False
        self.set(multipliers)
        return

    def set(self, multipliers):
        ''' Accepts {ethnicity: multiplier} or a sequence in ETHNICITIES order. '''
        if isinstance(multipliers, dict):
            missing = [e for e in ETHNICITIES if e not in multipliers]
            extra = [k for k in multipliers if k not in ETHNICITIES]
            if missing or extra:
                errormsg = (f'{self.name} multipliers must have exactly the keys {ETHNICITIES}; '
                            f'missing {missing}, unexpected {extra}.')
                raise ValueError(errormsg)
            vals = [multipliers[e] for e in ETHNICITIES]
        else:
            vals = list(multipliers)
            if len(vals) != N_ETH:
                errormsg = (f'{self.name} multipliers must have {N_ETH} entries in the order '
                            f'{ETHNICITIES}, got {len(vals)}.')
                raise ValueError(errormsg)
        mult = np.asarray(vals, dtype=float)
        if np.any(mult < 0):
            raise ValueError(f'{self.name} multipliers must be non-negative, got {mult.tolist()}.')
        self.raw = mult
        if self.renormalise:
            mean = float(POP_SHARES @ mult)
            if mean <= 0:
                raise ValueError(f'{self.name} multipliers are zero for every group; nothing to '
                                 f'renormalise against.')
            mult = mult / mean
        self.mult = mult
        self._warned = False
        return self

    def as_dict(self):
        return {e: float(m) for e, m in zip(ETHNICITIES, self.mult)}

    @property
    def is_null(self):
        ''' True if this makes no difference to anything, i.e. every multiplier is exactly 1. '''
        return bool(np.all(self.mult == 1.0))

    def probs(self, sim, inds, base):
        '''
        Per-agent acceptance probability for the agents in ``inds``, given the national
        probability ``base`` for this timestep.
        '''
        p = float(base) * self.mult[ethnicity_of(sim, inds)]
        if not self._warned and p.size and p.max() > 1.0:
            over = [ETHNICITIES[i] for i in range(N_ETH) if float(base) * self.mult[i] > 1.0]
            print(f'Note: {self.name} uptake for {over} exceeds 1 at a base probability of '
                  f'{float(base):.3f} and is clipped to 1. The realised gradient is therefore '
                  f'flatter than the multipliers imply wherever the base probability is high.')
            self._warned = True
        return np.clip(p, 0.0, 1.0)


VACC_UPTAKE = EthnicityUptake(VACCINATION_MULTIPLIERS, 'vaccination')
SCREEN_UPTAKE = EthnicityUptake(SCREENING_MULTIPLIERS, 'screening')


class UptakeAudit:
    '''
    Optional tally of offers and acceptances by ethnicity, so delivered uptake can be checked
    rather than assumed -- the counterpart of core_vacc.py measuring its dose neutrality instead
    of trusting the algebra.

        eth.AUDIT = eth.UptakeAudit()
        sim.run()
        print(eth.AUDIT.report())

    Off (None) by default and skipped entirely when off. Single-run diagnostic: it is a
    module-level object, so an in-process MultiSim would pool every run's offers into it, and a
    spawned worker would fill its own copy that the parent never sees.

    Counts offers, not people: an agent who declines a screening invitation is invited again and
    counted again, which is what makes the ratio here a clean read of the multiplier.
    '''

    def __init__(self):
        self.offered = {}
        self.accepted = {}
        self.per_year = {}
        return

    @staticmethod
    def _key(intervention):
        label = intervention.label or type(intervention).__name__
        if isinstance(intervention, hpv.BaseVaccination):  # vx labels are None in NHS_Vacc.py
            name = intervention.product.genotype_pars['name'].values[0]
            label = f'vx {name} {intervention.start_year:.0f}-'
        return f'{intervention.uptake.name}: {label}'

    def record(self, intervention, sim, eligible, accept):
        key = self._key(intervention)
        for store, inds in ((self.offered, eligible), (self.accepted, accept)):
            row = store.setdefault(key, np.zeros(N_ETH, dtype=np.int64))
            if len(inds):
                row += np.bincount(ethnicity_of(sim, inds), minlength=N_ETH)
        # Timesteps per year, or 1 where the probability was per-offer to begin with -- needed to
        # put the reported rate back on the scale the multiplier was quoted on
        self.per_year[key] = round(1 / sim['dt']) if intervention.annual_prob else 1
        return

    def report(self):
        '''
        Offers, acceptances and the realised acceptance rate per ethnicity, with each group's rate
        as a ratio to ETHNICITIES[0]'s. The ratio is the number to check: it is what the multiplier
        sets, provided the probability being scaled stays below 1 (clipping flattens it) and the
        mix of programme years behind the offers is similar between groups.

        Vaccination offers are per timestep, so their rate is annualised first -- comparing the
        raw per-timestep rates would understate the gradient in exactly the way _per_agent_probs()
        exists to avoid.
        '''
        lines = []
        for key in sorted(self.offered):
            off = self.offered[key].astype(float)
            acc = self.accepted.get(key, np.zeros(N_ETH)).astype(float)
            rate = np.divide(acc, off, out=np.full(N_ETH, np.nan), where=off > 0)
            n = self.per_year.get(key, 1)
            annual = 1 - (1 - rate) ** n
            ref = annual[0]
            scale = 'per year' if n > 1 else 'per offer'
            lines.append(f'{key}')
            lines.append(f"  {'':<10}{'offers':>10}{'accepts':>10}{scale:>11}"
                         f"{'ratio to ' + ETHNICITIES[0]:>18}")
            for i, name in enumerate(ETHNICITIES):
                ratio = annual[i] / ref if ref else np.nan
                lines.append(f'  {name:<10}{off[i]:>10,.0f}{acc[i]:>10,.0f}'
                             f'{annual[i]:>11.3f}{ratio:>18.3f}')
        return '\n'.join(lines)


# Set to an UptakeAudit() to tally offers and acceptances by ethnicity; None = no overhead.
AUDIT = None


def set_multipliers(vaccination=None, screening=None):
    '''
    Change the setting after import. Interventions hold a reference to the shared uptake objects,
    so this reaches lists already built by get_interventions() -- but only in THIS process: call
    it at module level in a run script, not inside main(), or spawned workers will not see it
    (use the ETH_VACC_MULT / ETH_SCREEN_MULT environment variables instead).
    '''
    if vaccination is not None:
        VACC_UPTAKE.set(vaccination)
    if screening is not None:
        SCREEN_UPTAKE.set(screening)
    return


# ---------------------------------------------------------------------
# Observed uptake for this study
# ---------------------------------------------------------------------
# Uptake by ethnic group, as percentages, supplied for this project. The fourth community stands
# for "Other" here (the source's Other category includes Chinese), which is why it is not the
# Chinese-specific figure the community label suggests.
OBSERVED_VACCINATION_PCT = dict(White=74.1, Asian=63.8, Black=70.1, Chinese=66.4)   # Chinese = Other
OBSERVED_SCREENING_PCT   = dict(White=74.4, Asian=54.9, Black=49.2, Chinese=36.0)   # Chinese = Other


def multipliers_from_uptake(uptake_pct, reference='White', national=None):
    '''
    Turn measured uptake PERCENTAGES into the multipliers this file applies. Exactly one of
    ``reference`` or ``national``:

    reference='White'   divide by that group's uptake. The reference group keeps the model's own
                        calibrated probability (0.80 first-dose coverage, 0.68/0.76 screening) and
                        everyone else is placed relative to it, so only the GRADIENT comes from
                        the data and the national level stays where the calibration put it.
    national=0.80       divide by an absolute probability instead, taking the quoted percentages
                        as the acceptance probabilities themselves. Reproduces the numbers
                        literally, at the cost of moving the overall level away from the
                        calibrated one.

    The reference form is the default, for two reasons. The reference group is 91% of the modelled
    population, so the national aggregate barely moves. And a measured "74.4% screened" is
    coverage -- the share of women screened within the last 3-5 years, accumulated over repeated
    invitations -- whereas the model's 0.68 is the probability of accepting ONE invitation. The two
    are different quantities and should not be equated; their RATIO between groups is the part that
    carries over.
    '''
    if (reference is None) == (national is None):
        raise ValueError('Provide exactly one of reference or national.')
    pct = {k: float(v) for k, v in uptake_pct.items()}
    denom = pct[reference] if national is None else float(national) * 100
    return {k: v / denom for k, v in pct.items()}


def use_observed(reference='White', national_vacc=None, national_screen=None):
    '''
    Apply OBSERVED_*_PCT. See multipliers_from_uptake() for what the two conversions mean; pass
    national_vacc/national_screen to use the absolute form instead of the reference form.
    '''
    kw_v = dict(national=national_vacc) if national_vacc else dict(reference=reference)
    kw_s = dict(national=national_screen) if national_screen else dict(reference=reference)
    set_multipliers(vaccination=multipliers_from_uptake(OBSERVED_VACCINATION_PCT, **kw_v),
                    screening=multipliers_from_uptake(OBSERVED_SCREENING_PCT, **kw_s))
    return


# Applied at import when set, so spawned workers get it too (an assignment inside a run script's
# main() would not survive process spawn):  ETH_OBSERVED=1 python run_ethnicity_uptake.py
if os.environ.get('ETH_OBSERVED', '0') != '0':
    use_observed()


# =====================================================================
# 4. Ethnicity-aware intervention classes
# =====================================================================

class _EthnicityUptakeMixin:
    '''
    Makes an HPVsim routine intervention accept people at a rate that depends on their ethnicity.

    HPVsim decides who accepts in ONE line, interventions.select_people(eligible_inds, prob=p),
    with a single scalar p for the whole cohort. Rather than reimplement each apply() -- which
    would mean duplicating the dose/screen counters, the immunity bookkeeping and the results
    updates, and keeping the copies in step with the vendored HPVsim -- this draws the acceptance
    itself and then hands the accepted set straight back to the stock apply() as "everyone
    eligible, probability 1". All the bookkeeping stays in vendored code.

    The real eligibility function is evaluated exactly once per timestep despite apply() asking
    for it again, which matters: NHS_2025_lambdamu's routine-screening eligibility loops over
    every agent in the sim.
    '''

    def __init__(self, *args, uptake=None, **kwargs):
        self.uptake = uptake
        super().__init__(*args, **kwargs)
        return

    def _replacement_eligibility(self, accept, n):
        ''' Boolean mask -- what BaseVaccination/BaseScreening.check_eligibility() expects. '''
        mask = np.zeros(n, dtype=bool)
        mask[accept] = True
        return lambda sim: mask

    def _per_agent_probs(self, sim, inds, ti):
        '''
        Per-agent acceptance probability for this timestep, with the multiplier applied ON THE
        SCALE THE PROBABILITY WAS QUOTED ON.

        This is not a detail. RoutineDelivery.initialize() has already replaced an annual
        probability p with the per-timestep 1-(1-p)**dt by the time apply() runs, and scaling
        THAT compresses the gradient: at dt=0.25, an annual 0.80 becomes 0.3313 per timestep, and
        0.75 x 0.3313 annualises back to 0.68, not the 0.60 the multiplier was meant to express.
        So the conversion is inverted, the multiplier applied to the annual figure (which is also
        where clipping at 1 belongs), and the conversion redone. The inversion is exact: it undoes
        arithmetic this same code did at initialization.

        Screening and triage pass annual_prob=None/False in NHS_2025_lambdamu.py -- their
        probabilities are per-offer already -- so there is nothing to undo and the multiplier
        applies directly.
        '''
        base = float(self.prob[ti])
        if not self.annual_prob:
            return self.uptake.probs(sim, inds, base)
        dt = sim['dt']
        annual = 1 - (1 - base) ** (1 / dt)
        return 1 - (1 - self.uptake.probs(sim, inds, annual)) ** dt

    def apply(self, sim):
        if self.uptake is None or self.uptake.is_null or sim.t not in self.timepoints:
            return super().apply(sim)  # nothing to change: stock behaviour, including the no-op path

        eligible = self.check_eligibility(sim)
        if len(eligible) == 0:
            accept = np.empty(0, dtype=hpv.default_int)
        else:
            ti = sc.findinds(self.timepoints, sim.t)[0]
            accept = hpi.select_people(eligible, prob=self._per_agent_probs(sim, eligible, ti))
        if AUDIT is not None:
            AUDIT.record(self, sim, eligible, accept)

        saved_elig, saved_prob = self.eligibility, self.prob
        self.eligibility = self._replacement_eligibility(accept, len(sim.people))
        self.prob = np.ones_like(saved_prob)
        try:
            return super().apply(sim)
        finally:
            self.eligibility, self.prob = saved_elig, saved_prob


class eth_routine_vx(_EthnicityUptakeMixin, hpv.routine_vx):
    ''' hpv.routine_vx with a per-ethnicity uptake multiplier. '''
    pass


class eth_routine_screening(_EthnicityUptakeMixin, hpv.routine_screening):
    ''' hpv.routine_screening with a per-ethnicity uptake multiplier. '''
    pass


class eth_routine_triage(_EthnicityUptakeMixin, hpv.routine_triage):
    '''
    hpv.routine_triage with a per-ethnicity uptake multiplier. BaseTriage.check_eligibility()
    returns INDICES (its eligibility functions hand back other interventions' outcome lists),
    not the boolean mask the screening/vaccination classes use, so the replacement has to match.
    '''

    def _replacement_eligibility(self, accept, n):
        return lambda sim: accept


# =====================================================================
# 5. The programmes, cloned from NHS_Vacc.py and NHS_2025_lambdamu.py
# =====================================================================

def _vx_product_name(prod):
    '''
    Recover the product string ('bivalent', 'bivalent2', ...) from a vx object. Products are
    passed by name rather than by reference so each intervention gets its own instance, exactly
    as NHS_Vacc.py's routine_vx(product='bivalent') calls do -- vx.imm_source is assigned per-sim
    during immunity initialization ("Warning, fragile!!!", interventions.py), and sharing one
    product object across two sims' intervention lists would have them fighting over it.

    default_vx() builds each base name plus a '2' (imm_boost 1.2) and '3' (imm_boost 1.1)
    variant, so the suffix is recoverable from imm_boost. The round-trip is checked rather than
    assumed.
    '''
    base = str(prod.genotype_pars['name'].values[0])
    if prod.imm_boost is None:
        name = base
    elif np.isclose(prod.imm_boost, 1.2):
        name = base + '2'
    elif np.isclose(prod.imm_boost, 1.1):
        name = base + '3'
    else:
        errormsg = (f'Vaccine product {base!r} has imm_boost={prod.imm_boost}, which does not '
                    f'match any of default_vx()\'s dose variants, so it cannot be cloned by name.')
        raise ValueError(errormsg)
    check = hpi.default_vx(prod_name=name)
    if check.imm_boost != prod.imm_boost or (check.imm_init is None) != (prod.imm_init is None):
        raise ValueError(f'Product name {name!r} does not round-trip back to the same product.')
    return name


def _dx_product_name(prod):
    ''' Recover the diagnostic string ('hpv', 'lbc', 'colposcopy') from a dx object. '''
    names = prod.df.name.unique()
    if len(names) != 1:
        raise ValueError(f'Expected one product name, got {names.tolist()}.')
    return str(names[0])


def _clone_vx(src, uptake):
    ''' NHS_Vacc.py's routine_vx, rebuilt as its ethnicity-aware equivalent. '''
    return eth_routine_vx(product=_vx_product_name(src.product),
                          prob=np.array(src.prob, dtype=float),  # copied: initialize() rewrites it
                          start_year=src.start_year,
                          end_year=src.end_year,
                          sex=src.sex,
                          age_range=src.age_range,
                          eligibility=src.eligibility,
                          label=src.input_args.get('label'),
                          uptake=uptake)


def _clone_test(src, uptake):
    ''' NHS_2025_lambdamu.py's routine_screening / routine_triage, rebuilt the same way. '''
    cls = eth_routine_screening if isinstance(src, hpv.BaseScreening) else eth_routine_triage
    return cls(product=_dx_product_name(src.product),
               prob=np.array(src.prob, dtype=float),
               start_year=src.start_year,
               end_year=src.end_year,
               eligibility=src.eligibility,
               age_range=src.age_range,
               annual_prob=src.annual_prob,
               label=src.label,
               uptake=uptake)


def _check_imm_source_ordering(interventions):
    '''
    Copied from core_vacc.py, because the booby trap is in the vendored HPVsim rather than in
    either file: immunity.init_immunity() assigns each vaccine product's imm_source from the
    POSITION OF ITS FIRST OCCURRENCE in the intervention list, but sizes people.peak_imm as
    n_genotypes + n_unique_products. Those only agree if the first nv entries introduce nv
    distinct products; otherwise a later product indexes off the end of the array and the run
    dies mid-flight with an IndexError. Cloning preserves NHS_Vacc.py's order, which satisfies
    this -- the check is here so that an edit which breaks it fails at import rather than 30
    minutes into a run.
    '''
    names = [iv.product.genotype_pars['name'].values[0] for iv in interventions]
    _, first = np.unique(names, return_index=True)
    if sorted(first.tolist()) != list(range(len(first))):
        errormsg = (
            f'Vaccination list ordering would break HPVsim immunity indexing. Distinct products '
            f'first appear at positions {sorted(first.tolist())}, but they must appear at '
            f'{list(range(len(first)))}. Current order: {names}')
        raise ValueError(errormsg)
    return


def get_vaccinations():
    '''
    NHS_Vacc.vaccinations with the ethnicity multiplier applied. Order, products, years, ages,
    sexes and probabilities are all NHS_Vacc.py's; first doses are identified by their
    eligibility function rather than by name, so adding an era there needs no edit here.
    '''
    out = []
    for src in NHS_Vacc.vaccinations:
        first_dose = src.eligibility is NHS_Vacc.eligible_first_dose
        uptake = VACC_UPTAKE if (first_dose or VACC_APPLY_TO_LATER_DOSES) else None
        out.append(_clone_vx(src, uptake))
    _check_imm_source_ordering(out)
    return out


def get_screening(l, m, end_screening_at_switch_year=False):
    '''
    NHS_2025_lambdamu.get_interventions(l, m) with the ethnicity multiplier applied to the
    acceptance step of the screening (and optionally triage) interventions.

    The list is walked rather than rebuilt, so the pathway -- the trackers, the eligibility
    functions, the order they run in, the treatments at the end -- stays entirely that file's.
    Labels are preserved, which is load-bearing: its trackers and eligibility functions find each
    other with sim.get_intervention('routine_screening_under50') and friends.
    '''
    out = []
    for src in NHS.get_interventions(l, m, end_screening_at_switch_year=end_screening_at_switch_year):
        if isinstance(src, hpv.BaseScreening):
            differentiate = (src.label in PRIMARY_SCREEN_LABELS
                             or (APPLY_TO_FOLLOWUP_SCREENS and src.label in FOLLOWUP_SCREEN_LABELS))
            out.append(_clone_test(src, SCREEN_UPTAKE if differentiate else None))
        elif isinstance(src, hpv.BaseTriage):
            differentiate = APPLY_TO_TRIAGE and src.label in TRIAGE_LABELS
            out.append(_clone_test(src, SCREEN_UPTAKE if differentiate else None))
        else:
            out.append(src)  # trackers (plain functions) and the two treatments, used as-is
    return out


def get_interventions(l=1, m=1, end_screening_at_switch_year=False):
    '''
    The whole thing: screening pathway then vaccination programme, ready to hand to hpv.Sim().
    Drop-in for NHS_2025_lambdamu.get_interventions(l, m) + NHS_Vacc.vaccinations.

    l, m are NHS_2025_lambdamu's post-switch_year screening-interval scale factors for
    unvaccinated and vaccinated agents respectively; l = m = 1 keeps the current 5-year interval
    for everyone.
    '''
    return get_screening(l, m, end_screening_at_switch_year=end_screening_at_switch_year) + get_vaccinations()


# =====================================================================
# 6. Demo / self-check
# =====================================================================

def print_uptake_table(vacc=None, screen=None):
    ''' What the current setting does to each headline probability, with no simulation involved. '''
    vacc = vacc or VACC_UPTAKE
    screen = screen or SCREEN_UPTAKE
    import GlobalScreeningParameters as GSP

    rows = [
        ('vaccination, 1st dose 2008-2011', vacc, 0.86),
        ('vaccination, 1st dose 2019',      vacc, 0.85),
        ('vaccination, 1st dose 2020',      vacc, (0.60 + 0.53) / 2),
        ('vaccination, 1st dose 2022-',     vacc, GSP.projected_teen_vaccination_uptake),
        ('screening, primary <50',          screen, GSP.primary_screen_prob_under50),
        ('screening, primary 50+',          screen, GSP.primary_screen_prob_50andover),
        ('screening, 2nd consecutive',      screen, GSP.secondary_screen_prob),
        ('screening, 3rd consecutive',      screen, GSP.third_screen_prob),
    ]

    print(f'Ethnicities (community order): {ETHNICITIES}')
    print(f'Population weights:            {np.round(POP_SHARES, 4).tolist()}')
    print(f'Vaccination multipliers:       {vacc.as_dict()}'
          + ('  [renormalised]' if vacc.renormalise else ''))
    print(f'Screening multipliers:         {screen.as_dict()}'
          + ('  [renormalised]' if screen.renormalise else ''))
    print(f'Later doses differentiated: {VACC_APPLY_TO_LATER_DOSES}   '
          f'Follow-up screens: {APPLY_TO_FOLLOWUP_SCREENS}   Triage: {APPLY_TO_TRIAGE}\n')

    head = f"{'stage':<34}{'national':>10}" + ''.join(f'{e:>10}' for e in ETHNICITIES) + f"{'wtd mean':>10}"
    print(head)
    print('-' * len(head))
    for name, uptake, base in rows:
        p = np.clip(base * uptake.mult, 0, 1)
        print(f'{name:<34}{base:>10.4f}' + ''.join(f'{x:>10.4f}' for x in p)
              + f'{float(POP_SHARES @ p):>10.4f}')
    return


def check_annual_scale(dt=0.25, annual=0.8, seed=1, verbose=True):
    '''
    Exact check of _per_agent_probs(), with no simulation run and no Monte Carlo: an agent of
    ethnicity e offered a service whose ANNUAL probability is p must accept, in a single dt
    timestep, with probability 1-(1-p*m_e)**dt -- so that a full year of offers delivers exactly
    p*m_e.

    This is the step that is easy to get wrong (scaling HPVsim's already-converted per-timestep
    probability instead delivers 0.68 where the multiplier asked for 0.60 at p=0.8, m=0.75), and
    a sim big enough to catch that difference statistically takes minutes. This takes a second.
    '''
    uptake = EthnicityUptake([1.0, 0.75, 0.60, 0.90][:N_ETH], 'annual-scale check', renormalise=False)
    iv = eth_routine_vx(product='nonavalent', prob=annual, start_year=2025, end_year=2026,
                        sex=['f', 'm'], age_range=[12, 13], uptake=uptake)
    sim = hpv.Sim(dict(n_agents=200, start=2020, end=2030, dt=dt, rand_seed=seed, verbose=0,
                       location='united kingdom'),
                  interventions=[iv])
    sim.initialize()
    # Sim copies its parameters, so the object that got initialized -- and whose prob has been
    # converted to per-timestep -- is the sim's, not the one built above.
    iv = sim['interventions'][0]

    inds = np.arange(len(sim.people))
    got = iv._per_agent_probs(sim, inds, 0)
    want = 1 - (1 - np.clip(annual * uptake.mult[ethnicity_of(sim, inds)], 0, 1)) ** dt
    worst = float(np.abs(got - want).max())
    delivered = 1 - (1 - got) ** (1 / dt)  # what a full year of offers comes to, per agent

    if verbose:
        print(f'Annual-scale check: annual p={annual}, dt={dt}, multipliers {uptake.mult.tolist()}')
        print(f"  {'':<10}{'per timestep':>14}{'over a year':>14}{'target':>10}")
        for i, name in enumerate(ETHNICITIES):
            sel = ethnicity_of(sim, inds) == i
            if not sel.any():
                continue
            print(f'  {name:<10}{got[sel][0]:>14.6f}{delivered[sel][0]:>14.6f}'
                  f'{min(1.0, annual * uptake.mult[i]):>10.6f}')
        print(f'  largest deviation from target: {worst:.2e}  ({"OK" if worst < 1e-12 else "FAIL"})')
    if worst >= 1e-12:
        raise AssertionError(f'per-agent probabilities are off by up to {worst:.3e}')
    return worst


def check_predraw(n_agents=2_000, end=2005, seed=3, verbose=True):
    '''
    The central claim of the patch, checked on a real run: the ethnicity that can be read off an
    agent at age 12 is the one the network actually gives them at debut, and the distribution of
    those tags is still community_probs.

    Every agent who joined the network after t=0 must satisfy
    people.community[uid] == ethnicity_for(uid, seed, community_probs). The t=0 cohort is
    deliberately exempt (see the module docstring) and is excluded by age.
    '''
    pars = sc.mergedicts(basePars_community.base_pars_geno, dict(
        n_agents=n_agents, end=end, rand_seed=seed, verbose=0, analyzers=[], interventions=[]))
    sim = hpv.Sim(pars)
    sim.initialize()
    # People.add_births() only ever appends (_grow), so the t=0 cohort keeps positions [0, n0)
    # for the whole run and everyone born into the sim sits above it. Age will not do as the test:
    # a dead agent's age stops advancing, so a t=0 agent who died at 20 still reads as 20 decades
    # later and would pass for a recent birth.
    n0 = len(sim.people)
    sim.run()

    probs = _community_probs_of(sim)
    born_into_sim = np.arange(len(sim.people)) >= n0
    # Multiscale cancer agents (level1) are clones: People.set_severity() copies every state from
    # the agent that spawned them, ethnicity included. Carrying the source agent's ethnicity is
    # exactly right, but it is by definition not the predraw for their own slot, so they are not
    # part of this test.
    inds = np.where((sim.people.community >= 0) & born_into_sim & sim.people.level0)[0]
    if inds.size == 0:
        raise AssertionError('no post-start network members to check -- run for longer')
    want = ethnicity_for(inds, seed, probs)
    got = np.asarray(sim.people.community[inds], dtype=np.int64)
    mismatch = int((want != got).sum())

    # And the marginal, over far more agents than one small sim provides
    draw = ethnicity_for(np.arange(200_000), seed, probs)
    share = np.bincount(draw, minlength=N_ETH) / draw.size

    if verbose:
        print(f'Predraw check: {inds.size:,} agents entered the network after t=0')
        print(f'  mismatches between predraw and the tag the network used: {mismatch}  '
              f'({"OK" if mismatch == 0 else "FAIL"})')
        print(f"  {'':<10}{'target':>10}{'predrawn':>10}   (200k draws)")
        for i, name in enumerate(ETHNICITIES):
            print(f'  {name:<10}{probs[i]:>10.4f}{share[i]:>10.4f}')
    if mismatch:
        raise AssertionError(f'{mismatch} of {inds.size} network members carry an ethnicity other '
                             f'than the one readable before their debut')
    return mismatch


def check_null_equivalence(n_agents=3_000, end=2030, seed=1, verbose=True):
    '''
    With every multiplier at 1, this file must reproduce NHS_2025_lambdamu + NHS_Vacc exactly --
    that is the claim the whole comparison rests on, since the control arm is this module rather
    than those files (the network patch is installed either way, so the two sims here differ only
    in the intervention objects).

    Runs the same sim twice and compares the results. Slow-ish for a check, so it is opt-in:

        EQUIV=1 python NHS_ethnicity_uptake.py

    Measured at 3,000 agents to 2030: every compared result identical to the last bit.
    '''
    saved = (VACC_UPTAKE.mult.copy(), SCREEN_UPTAKE.mult.copy())
    set_multipliers(vaccination=[1.0] * N_ETH, screening=[1.0] * N_ETH)
    try:
        base = dict(n_agents=n_agents, end=end, rand_seed=seed, verbose=0, analyzers=[])
        arms = {
            'this file (all multipliers 1)': get_interventions(l=1, m=1),
            'NHS_2025_lambdamu + NHS_Vacc': (NHS.get_interventions(l=1, m=1) + NHS_Vacc.vaccinations),
        }
        out = {}
        for name, interventions in arms.items():
            if verbose:
                print(f'  running: {name} ...')
            sim = hpv.Sim(sc.mergedicts(basePars_community.base_pars_geno,
                                        base, dict(interventions=interventions)))
            sim.run()
            out[name] = sim
    finally:
        VACC_UPTAKE.set(saved[0])
        SCREEN_UPTAKE.set(saved[1])

    # Intervention delivery first, then the epi outcomes downstream of it. The epi keys are flows
    # (this HPVsim has no cum_infections), so they are summed for the printed total but compared
    # timestep by timestep.
    keys = ['cum_doses', 'cum_vaccinated', 'cum_screens', 'infections', 'cancers', 'cancer_deaths']
    a, b = out.values()
    worst, bad = 0.0, []
    for key in keys:
        d = float(np.abs(np.asarray(a.results[key][:]) - np.asarray(b.results[key][:])).max())
        worst = max(worst, d)
        if d:
            bad.append(f'{key} (max abs diff {d:g})')
        if verbose:
            total_a, total_b = (np.asarray(s.results[key][:]).sum() if key.startswith(('inf', 'can'))
                                else s.results[key][-1] for s in (a, b))
            print(f'  {key:<20} total {total_a:>14,.0f} vs {total_b:>14,.0f}'
                  f'   max abs diff {d:g}')
    if bad:
        raise AssertionError('null setting is not equivalent to the stock programmes: '
                             + '; '.join(bad))
    if verbose:
        print('  identical  (OK)')
    return worst


def smoke_test(n_agents=5_000, end=2036, seed=1):
    """
    Run a small community-network sim under a deliberately strong (and entirely made-up) gradient
    and check that what came out is what the setting asked for.

    The check is on OFFERS, not on lifetime coverage, via UptakeAudit: acceptance per offer is
    exactly what the multiplier sets, whereas lifetime coverage is confounded by who was present
    during the vaccination window and, for screening, by everyone eventually accepting after
    enough repeat invitations. Both are printed -- the first as the test, the second because it
    is the quantity that actually matters epidemiologically, and the gap between them is the
    point.

    A shakedown, not a result: a few thousand agents leaves the smallest ethnicity with a few
    hundred offers, so its rate is noisy.

    NB HPVsim emits "invalid value encountered in power" while initializing this (and any other)
    intervention list built from NHS_Vacc.py -- its vx_1921_d2 probability works out to 1.13,
    which is the TODO already flagged in that file. Pre-existing and inherited unchanged.
    """
    global AUDIT
    demo_vacc = dict(White=1.00, Asian=0.75, Black=0.60, Chinese=0.90)
    demo_screen = dict(White=1.00, Asian=0.70, Black=0.55, Chinese=0.85)
    print('\n' + '=' * 78)
    print('SMOKE TEST -- demo multipliers, invented to exercise the machinery, not estimates')
    print('=' * 78)
    set_multipliers(vaccination=demo_vacc, screening=demo_screen)
    print_uptake_table()

    # verbose=0 rather than basePars_community's -1: at -1 HPVsim prints its one-line brief()
    # summary at the end, which contains a character a cp1252 Windows console cannot encode.
    pars = sc.mergedicts(basePars_community.base_pars_geno, dict(
        n_agents=n_agents, end=end, rand_seed=seed, verbose=0, analyzers=[],
        interventions=get_interventions(l=1, m=1),
    ))
    sim = hpv.Sim(pars)
    AUDIT = UptakeAudit()
    print(f'Running {n_agents:,} agents, {pars["start"]}-{end}, seed {seed} ...')
    sim.run()

    print('\nACCEPTANCE PER OFFER (the direct test -- ratios should sit on the multipliers)')
    print(f'  multipliers: vaccination {[f"{m:.2f}" for m in VACC_UPTAKE.mult]}, '
          f'screening {[f"{m:.2f}" for m in SCREEN_UPTAKE.mult]}\n')
    print(AUDIT.report())

    people = sim.people
    year = sim.yearvec[-1]
    # Everyone who passed through the 12-13 window during the 2022- era (nominal first-dose
    # coverage 0.80, sex-neutral), i.e. currently aged 13 to (year-2022)+13.
    age_hi = min(30, (year - 2022) + 13)
    vx_sel = people.alive & (people.age >= 13) & (people.age <= age_hi)
    sc_sel = (people.alive & people.is_female & (people.age >= 30) & (people.age <= 64)
              & (people.age > people.debut))

    print('\nLIFETIME OUTCOMES (confounded, as above -- shown for contrast, not as the test)')
    print(f"{'':<34}" + ''.join(f'{e:>10}' for e in ETHNICITIES))
    for title, sel, flag in ((f'  ever vaccinated, aged 13-{age_hi:.0f}', vx_sel, people.vaccinated),
                             ('  ever screened, women 30-64', sc_sel, people.screened),
                             ('  mean screens, women 30-64', sc_sel, people.screens)):
        inds = np.where(sel)[0]
        eth = ethnicity_of(sim, inds)
        vals = np.asarray(flag[inds], dtype=float)
        cells = ''.join(f'{vals[eth == i].mean() if (eth == i).any() else np.nan:>10.3f}'
                        for i in range(N_ETH))
        print(f'{title:<34}{cells}')

    AUDIT = None
    return sim


if __name__ == '__main__':
    print_uptake_table()
    print()
    check_annual_scale()
    print()
    check_predraw()
    if os.environ.get('EQUIV', '0') != '0':
        print('\nNull-equivalence check (all multipliers 1 == the stock programmes):')
        check_null_equivalence()
    if os.environ.get('SMOKE', '1') != '0':
        smoke_test(n_agents=int(os.environ.get('N_AGENTS', 5_000)),
                   end=int(os.environ.get('END', 2036)))

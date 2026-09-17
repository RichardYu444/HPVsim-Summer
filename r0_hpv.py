"""
r0_hpv.py
=========

Basic reproduction number R0 of HPV in this project's HPVsim models: the 'default' network
(basePars.py) and the age+community 'community' network with Gamma (basePars_community.py) or
power-law (basePars_community_powerlaw.py) partner propensities.

R0 here is the intrinsic quantity: a fully susceptible population, no vaccination, no screening.
Each genotype has its own R0 (its own beta and infection duration). It counts the DISTINCT people
a case infects; see "Reinfection" at the end for what that leaves out.


1. Per-partnership transmission
-------------------------------
A partnership transmits with probability T = 1 - exp(-lambda * overlap), where overlap is how
long the partnership and the infectious period coincide, and lambda (per year) is

    lambda = -a * ln(1 - p)         a = acts per year in that partnership  (~ a*p for small p)
    p      = beta * rel_beta_g * trans_dir * (1 - condoms_layer * eff_condoms)
    trans_dir = transf2m (1.0) female->male,  transm2f (3.69) male->female

With partnerships of type tau (short/long, casual/marital) ending at rate q_tau (exponential
durations, so a current partnership's remaining length is also Exp(q_tau)), and kappa = lambda+q:

    to a current partner:          T_cur(D) = (lambda/kappa) (1 - e^{-kappa D})
    to partners formed during D:   T_new(D) = (lambda/kappa) (D - (1 - e^{-kappa D})/kappa)
                                   (per unit formation rate, i.e. per new partner per year)

In these models p is large (0.2-0.3 female->male, 0.7-1.1 male->female for HPV16) and there are
10-20 acts per quarter, so T_cur ~ 1 and T_new ~ D: transmission SATURATES, and R0 is set by how
many partners someone has "in play" during an infectious period, not by beta or acts.


2. Who is newly infected: two types per sex
-------------------------------------------
A newly infected person is not a typical person. There are two ways to get infected:

  A  by an EXISTING partner, who became infected during the partnership. People are drawn in
     proportion to their number of current partners n_i (size-biasing).
  B  at the START of a new partnership with someone already infected. People are drawn in
     proportion to how often they form partnerships, c_i.

Either way the infector cannot be reinfected. A type-X case of sex s then infects

    K[A <- X_s] = sum_tau S^X_tau * E[T_cur]      (their OTHER current partners -> type A cases)
    K[B <- X_s] = sum_tau C^X_tau * E[T_new]      (partners formed during D    -> type B cases)

    S^A_tau = E[(n-1) n_tau] / E[n]     C^A_tau = E[n c_tau] / E[n]      (weighted by n)
    S^B_tau = E[c n_tau] / E[c]         C^B_tau = E[c c_tau] / E[c]      (weighted by c)

(n_tau = current type-tau partners, c_tau = new type-tau partners per year, for sex s; the B
averages are taken at the moment of forming the infecting partnership, excluding it.)

Heterosexual transmission makes K a 4x4 matrix over (A_m, B_m, A_f, B_f) with only
male->female and female->male blocks, and

    R0 = spectral radius of K = sqrt( rho(K_{f<-m} K_{m<-f}) )

Special cases:
  * When formation is proportional to current degree -- the community model's dcSBM, where both
    are proportional to theta -- A and B cases behave the same, K collapses to one type per sex:
        R_s = sum_tau E[ S_tau T_cur + C_tau T_new ],   R0 = sqrt(R_m R_f)                 (*)
    and in the saturated limit, with h = E[theta^2]/E[theta]^2 = 1 + CV_theta^2,
        R_s = h (k_s + c_s E[D_s])
    k_s = mean current partners, c_s = mean new partners per year. (Gamma shape a: CV^2 = 1/a;
    Pareto alpha: CV^2 = 1/(alpha(alpha-2)), which diverges as alpha -> 2.)
  * Static network (no turnover): R_s = T_s (E[n^2] - E[n]) / E[n], the classic network formula.
    "Mean degree" enters as the mean EXCESS degree, not the mean degree.

The default network does not form partnerships in proportion to current degree (a "slot" model:
people refill concurrency slots as partnerships end), and neither does the community model once
you account for age (formation depends on current age, long partnerships on past age), so the
two-type version is what should be used in general.


3. Why community structure drops out
------------------------------------
Split cases by community c as well. The NGM entry from community c to c' is K(c) times
P(partner in c' | person in c), a row-stochastic matrix. If every community has the same degree
distribution and the same transmission parameters, K(c) = K for all c and the NGM is K (x) a
stochastic matrix, whose dominant eigenvalue is rho(K). basePars_community.py's IPF rescaling
guarantees exactly this (each community's mixing row sums to one, so theta alone sets degree),
as does run_sim_community_2iso.py's diagonal 1/p_c matrix. So community mixing changes WHERE
infection goes, not R0. It stops being true once degree, acts or condom use differ by community;
community-specific vaccination changes R_eff, not R0.

Age does NOT drop out the same way (act rates and partner numbers depend on age), and the
formulas above assume partners are drawn without further correlation beyond the A/B typing.


4. Reinfection
--------------
HPVsim lets a cleared partner be reinfected within the same partnership (men clear in ~1 year,
women's HPV16 infections last ~3 years on average and have a long tail). R0 here counts distinct
people; counting every infection a case causes roughly doubles the number. That "ping-pong" is a
threshold-relevant SIS effect that no individual-level R0 captures cleanly.


5. What this module provides
----------------------------
  R0Recorder                  analyzer: records partnership timelines from t0 (and optionally
                              runs a ground-truth one-generation outbreak, see validate_r0.py)
  estimate_r0(rec, sim)       two-type K from the recorded network, two ways:
                                'window' -- the formulas above evaluated on the ACTUAL partnership
                                            timelines (real durations, real formation after
                                            infection) with sampled infectious periods. Use this.
                                'closed' -- the formulas above with S, C and exponential q measured
                                            over one year and extrapolated at constant rate
  r0_from_parameters(sim)     (*) from community_pars alone (idealised dcSBM: no age effects,
                              singleness gate, forced pairing or mortality)

Usage on any sim (the network does not depend on the epidemic, so any run will do; the recording
should cover the longest infections, ~30 years, or pass t_end to truncate consistently):

    rec = r0_hpv.R0Recorder(t0_year=2000)
    sim = hpv.Sim(pars, analyzers=[rec]); sim.run()
    est, stats = r0_hpv.estimate_r0(sim.get_analyzer('R0Recorder'), sim)
    est['hpv16']['window']['R0']

Validation (validate_r0.py, HPV16, pooled over seeds; ground truth = one-generation outbreak):

    model                   ground truth        window   closed (const)   parameters only
    default                 1.48 +- 0.07        1.50     2.00             -
    gamma, 4 communities    2.53 +- 0.10        2.63     3.13             2.58
    gamma, 1 community      2.46 +- 0.11        2.62     3.13             2.58
    power law (200k)        5.59 +- 0.50 (*)    5.30     5.76             15.6

    (*) at 0.2% seeding; at 1% seeding hubs saturate and the outbreak under-counts (3.82 +- 0.16).
The constant-rate closed form overshoots because formation slows after a new partnership in the
default network and with age in all of them, which matters for women's long HPV16 infections.
"""
import numpy as np
import pandas as pd
import sciris as sc

import hpvsim_working as hpv
from hpvsim_working import utils as hpu
from hpvsim_working import parameters as hppar
from hpvsim_working import age_community_bipartite_network_model as acbnm

LAMBDA_INF = 1e9  # per-year hazard used when a single timestep transmits with certainty
TYPES = ('A_m', 'B_m', 'A_f', 'B_f')  # index order of the 4x4 next-generation matrices


# =====================================================================
# Transmission and infectious period, replicating Sim.step() / People.infect()
# =====================================================================

def genotype_key(sim, g):
    ''' Genotype name for index g. '''
    return sim['genotype_map'][g]


def per_act_prob(sim, g, lkey, direction):
    '''
    Per-act transmission probability p for genotype g in layer lkey, direction 'f2m' or 'm2f'.
    Not capped: values above 1 happen (beta=0.33 * transm2f=3.69 > 1), and then every timestep
    with at least one act transmits with certainty (see step_prob()).
    '''
    gpars = sim['genotype_pars'][genotype_key(sim, g)]
    trans = sim['transf2m'] if direction == 'f2m' else sim['transm2f']
    return sim['beta'] * gpars['rel_beta'] * trans * (1 - sim['condoms'][lkey] * sim['eff_condoms'])


def step_prob(p, acts_per_year, dt):
    ''' Per-timestep transmission probability of a partnership, exactly as Sim.step(), clipped. '''
    frac, whole = np.modf(np.asarray(acts_per_year, dtype=float) * dt)
    P = 1 - (1 - p) ** whole * (1 - frac * p)
    return np.clip(P, 0.0, 1.0)


def hazard_from_step_prob(P, dt):
    ''' Continuous per-year hazard equivalent to a per-timestep probability P. '''
    P = np.asarray(P, dtype=float)
    lam = np.full(P.shape, LAMBDA_INF)
    ok = P < 1
    lam[ok] = -np.log1p(-P[ok]) / dt
    return lam


def sample_transmitting_steps(sim, g, sex, n, ages=None):
    '''
    Number of timesteps on which a newly infected person can transmit, replicating HPVsim:
    infected on step t0, infectious from t0+1, cleared on step date_clearance (checked before
    transmission), so they transmit on (date_clearance - t0 - 1) steps.
      men:   date_clearance = t0 + ceil(D/dt),       D ~ dur_infection_male
      women: date_clearance = t0 + randround(D/dt),  D = dur_precin [+ dur_cin * age_mod if CIN]
    Progression to cancer is ignored (rare, and with ms_agent_ratio=1 negligible).

    Args:
        ages: ages of the women at infection (for dur_cin's age_risk multiplier); None => 1
    '''
    dt = sim['dt']
    n = int(n)
    if sex == 'm':
        D = hpu.sample(**sim['dur_infection_male'], size=n)
        steps = np.ceil(D / dt)
    else:
        gpars = sim['genotype_pars'][genotype_key(sim, g)]
        dur_precin = hpu.sample(**gpars['dur_precin'], size=n)
        rel_sev = hpu.sample(**sim['sev_dist'], size=n)
        cin_probs = hppar.compute_severity(dur_precin, rel_sev=rel_sev, pars=gpars['cin_fn'])
        cin = np.random.random(n) < cin_probs
        age_mod = np.ones(n)
        if ages is not None:
            age_mod[np.asarray(ages) >= sim['age_risk']['age']] = sim['age_risk']['risk']
        dur_cin = hpu.sample(**gpars['dur_cin'], size=n) * age_mod
        D = dur_precin + cin * dur_cin
        steps = np.floor(D / dt) + (np.random.random(n) < np.mod(D / dt, 1))  # sc.randround
    return np.maximum(steps - 1, 0).astype(np.int64)


def T_current(lam, q, D):
    ''' Transmission probability to a current partner (exponential remaining duration). '''
    kappa = lam + q
    return (lam / kappa) * -np.expm1(-kappa * D)


def T_new(lam, q, D):
    ''' Expected transmissions to partners formed during D, per unit formation rate. '''
    kappa = lam + q
    return (lam / kappa) * (D + np.expm1(-kappa * D) / kappa)


def spectral_radius(K):
    return float(np.max(np.abs(np.linalg.eigvals(K))))


# =====================================================================
# (*) from the community model's input parameters alone
# =====================================================================

def theta_h(community_pars, powerlaw=False):
    ''' h = E[theta^2]/E[theta]^2 = 1 + CV^2 of the partner propensity. '''
    a = float(community_pars.get('gamma_shape', 1.0))
    floor = float(community_pars.get('theta_floor', 0.0))
    if powerlaw:
        cv2 = 1.0 / (a * (a - 2.0)) if a > 2 else np.inf
    else:
        cv2 = 1.0 / a
    return 1.0 + cv2 * (1.0 - floor) ** 2  # theta = floor + (1-floor) X scales CV by (1-floor)


def r0_from_parameters(sim, powerlaw=False, n=200_000, age_act_factor=0.8):
    '''
    (*) using only community_pars. The dcSBM forms partnerships at rate proportional to theta,
    so per type k_tau = c_tau / q_tau, with the total set by mean_partners_per_year through
    interpretable_to_params(). Acts are drawn from sim['acts'] times a flat age factor.
    Returns ({genotype: {R_m, R_f, R0}}, info).
    '''
    cp = sim['community_pars']
    user = {k: v for k, v in cp.items() if k in ('mean_partners_per_year', 'gamma_shape',
            'D_mean_short', 'D_mean_long', 'frac_long')}
    with sc.capture():
        params = acbnm.interpretable_to_params(user, 1000)
    dt = sim['dt']
    pi = {'s': 1 - params['p_form_long'], 'l': params['p_form_long']}
    form_rate = params['q'] * params['k_snap'] * 12  # new partners per year at mean theta
    c = {k: form_rate * pi[k] for k in pi}
    q = {'s': 12 * params['q_short'], 'l': 12 * params['q_long']}
    h = theta_h(cp, powerlaw=powerlaw)

    res = {}
    for g in range(sim['n_genotypes']):
        R = {}
        for sex, direction in (('m', 'm2f'), ('f', 'f2m')):
            D = sample_transmitting_steps(sim, g, sex, n) * dt
            R[sex] = 0.0
            for lkey in ('s', 'l'):
                acts = hpu.sample(**sim['acts'][lkey], size=n) * age_act_factor
                lam = hazard_from_step_prob(step_prob(per_act_prob(sim, g, lkey, direction), acts, dt), dt)
                R[sex] += h * c[lkey] / q[lkey] * T_current(lam, q[lkey], D).mean()
                R[sex] += h * c[lkey] * T_new(lam, q[lkey], D).mean()
        res[genotype_key(sim, g)] = dict(R_m=R['m'], R_f=R['f'], R0=np.sqrt(R['m'] * R['f']))
    info = dict(h=h, k={k: c[k] / q[k] for k in c}, c=c, q=q)
    return res, info


# =====================================================================
# Recorder: partnership timelines (+ optional ground-truth outbreak)
# =====================================================================

class R0Recorder(hpv.Analyzer):
    '''
    Records every partnership present from step t0 onwards: endpoints, layer, acts, and the first
    and last step it was present. Analyzers run at the end of Sim.step(), after transmission, so
    "present at step t" means present during step t's transmission. Attach it to any sim (the
    network does not depend on the epidemic) and pass it to estimate_r0() afterwards.

    If ground_truth=True it also runs a one-generation outbreak for genotype index `genotype`:
      * at the end of step t0-1 it infects a random seed_frac of sexually active people
        (generation 0), who transmit from step t0;
      * everyone a seed infects is generation 1 -- a sample of newly infected people in an
        otherwise uninfected population;
      * anyone infected by anyone else gets rel_trans = 0 and never transmits; seeds stop
        transmitting after cohort_years;
      * every transmission from generations 0 and 1 is logged, so ground_truth() can type each
        case as A/B and build the empirical 4x4 next-generation matrix (ground_truth_ngm()).
    Use one genotype per run (rel_trans is per person, not per genotype) with no other
    infections (rel_init_prev=0, no interventions) and track_transmission=True. Keep seed_frac
    small enough that high-degree people are not likely to be seeded or infected by a seed --
    on the power-law network 1% was too much (hubs saturate and their partners are already
    infected), 0.2% was not.
    '''

    def __init__(self, t0_year, ground_truth=False, genotype=0, seed_frac=0.01, cohort_years=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.t0_year = t0_year
        self.ground_truth = ground_truth
        self.g = genotype
        self.seed_frac = seed_frac
        self.cohort_years = cohort_years
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.t0 = int(sc.findnearest(sim.yearvec, self.t0_year))
        self.lkeys = list(sim.people.contacts.keys())
        self.dt = sim['dt']
        self.eid = np.empty(0, dtype=np.int64)  # per-edge arrays, kept sorted by eid
        self.f = np.empty(0, dtype=np.int64)
        self.m = np.empty(0, dtype=np.int64)
        self.layer = np.empty(0, dtype=np.int8)
        self.acts = np.empty(0, dtype=float)
        self.first = np.empty(0, dtype=np.int64)
        self.last = np.empty(0, dtype=np.int64)
        self.snap = None
        if self.ground_truth:
            self.gen = np.full(len(sim.people), -1, dtype=np.int64)
            self.seeds = None
            self.tx = []  # (t, source, target, first_time) for sources in generations 0 and 1
            self.prev_at_end_cohort = None
        return

    def _grow_gen(self, n):
        if len(self.gen) < n:
            self.gen = np.concatenate([self.gen, np.full(n - len(self.gen), -1, dtype=np.int64)])

    def _record_edges(self, sim, t):
        eids, fs, ms, ls, acts = [], [], [], [], []
        for li, lkey in enumerate(self.lkeys):
            layer = sim.people.contacts[lkey]
            if len(layer) == 0:
                continue
            eids.append(np.asarray(layer['eid'], dtype=np.int64))
            fs.append(np.asarray(layer['f'], dtype=np.int64))
            ms.append(np.asarray(layer['m'], dtype=np.int64))
            ls.append(np.full(len(layer), li, dtype=np.int8))
            acts.append(np.asarray(layer['acts'], dtype=float))
        if not eids:
            return
        eids = np.concatenate(eids)
        if len(self.eid):
            pos = np.minimum(np.searchsorted(self.eid, eids), len(self.eid) - 1)
            known = self.eid[pos] == eids
            self.last[pos[known]] = t
        else:
            known = np.zeros(len(eids), bool)
        new = ~known
        if new.any():
            nn = int(new.sum())
            self.eid = np.concatenate([self.eid, eids[new]])
            self.f = np.concatenate([self.f, np.concatenate(fs)[new]])
            self.m = np.concatenate([self.m, np.concatenate(ms)[new]])
            self.layer = np.concatenate([self.layer, np.concatenate(ls)[new]])
            self.acts = np.concatenate([self.acts, np.concatenate(acts)[new]])
            self.first = np.concatenate([self.first, np.full(nn, t, dtype=np.int64)])
            self.last = np.concatenate([self.last, np.full(nn, t, dtype=np.int64)])
            order = np.argsort(self.eid, kind='stable')
            for key in ('eid', 'f', 'm', 'layer', 'acts', 'first', 'last'):
                setattr(self, key, getattr(self, key)[order])

    def apply(self, sim):
        t = sim.t
        people = sim.people
        if self.ground_truth:
            self._grow_gen(len(people))
            if t == self.t0 - 1:  # seed at the end of the step before t0
                cand = hpu.true(people.alive & people.is_active & people.susceptible[self.g])
                seeds = cand[np.random.random(len(cand)) < self.seed_frac]
                people.infect(inds=seeds, g=self.g, layer='seed_infection')
                self.seeds = seeds
                self.gen[seeds] = 0
                self.ever = np.zeros(len(people), bool)
                self.ever[seeds] = True
            if t >= self.t0:
                if len(self.ever) < len(people):
                    self.ever = np.concatenate([self.ever, np.zeros(len(people) - len(self.ever), bool)])
                for sources, targets, _scale, g, _lkey in people.new_transmissions:
                    if g != self.g or len(targets) == 0:
                        continue
                    src_gen = self.gen[sources]
                    first_time = ~self.ever[targets]
                    keep = (src_gen == 0) | (src_gen == 1)
                    if keep.any():
                        self.tx.append(np.stack([np.full(keep.sum(), t), sources[keep], targets[keep],
                                                 first_time[keep].astype(np.int64)], axis=1))
                    # Reinfections are always dead ends, so a generation-1 case is credited with the
                    # offspring of its FIRST infection only (otherwise a man who clears and is
                    # reinfected by his seed partner within the cohort window counts twice)
                    self.gen[targets] = np.where((src_gen >= 0) & first_time, src_gen + 1, 99)
                    self.ever[targets] = True
                    people.rel_trans[targets[self.gen[targets] >= 2]] = 0.0
                if t == self.t0 + int(round(self.cohort_years / self.dt)) - 1:
                    people.rel_trans[self.seeds] = 0.0  # seeds stop producing generation 1
                    self.prev_at_end_cohort = float(people.infectious[self.g][people.alive].mean())
        if t >= self.t0:
            if t == self.t0:
                self.snap = dict(age=people.age.copy(), alive=people.alive.copy())
            self._record_edges(sim, t)
        return

    def finalize(self, sim):
        super().finalize()
        self.t_end = sim.t
        n = len(sim.people)  # People only ever grows: pad the t0 snapshot to everyone
        n0 = len(self.snap['age'])
        self.is_female = np.asarray(sim.people.is_female).copy()
        self.age0 = np.concatenate([self.snap['age'], np.full(n - n0, np.nan)])
        self.alive0 = np.concatenate([self.snap['alive'], np.zeros(n - n0, bool)])
        if self.ground_truth:
            self.tx = np.concatenate(self.tx) if self.tx else np.empty((0, 4), dtype=np.int64)
        return

    # ------------------------------------------------------------------
    def ground_truth_ngm(self, sim, n_boot=500):
        '''
        Empirical next-generation matrix from the generation-1 cohort. A case is type A if the
        partnership it was infected through was already present when its infector became
        infectious (t0 for seeds), else type B. K[Y, X] = mean number of DISTINCT type-Y people a
        type-X generation-1 case went on to infect (reinfections of someone already infected
        once are excluded; K_all counts them). R0 = spectral radius, SE by bootstrapping cases.
        '''
        tx = self.tx
        t, src, tgt, first = tx[:, 0], tx[:, 1], tx[:, 2], tx[:, 3].astype(bool)
        from_seed = np.isin(src, self.seeds)
        # generation-1 cases: people whose first-ever infection came from a seed
        s_idx = np.flatnonzero(from_seed & first)
        order = s_idx[np.argsort(t[s_idx], kind='stable')]
        coh, first_pos = np.unique(tgt[order], return_index=True)
        coh_t = t[order][first_pos]
        coh_src = src[order][first_pos]
        not_seed = ~np.isin(coh, self.seeds)
        coh, coh_t, coh_src = coh[not_seed], coh_t[not_seed], coh_src[not_seed]
        coh_type_A = _pair_present(self, coh_src, coh, np.full(len(coh), self.t0))
        coh_female = self.is_female[coh]
        tinf = dict(zip(coh.tolist(), coh_t.tolist()))

        # offspring of generation-1 cases (all during their first infection: any later
        # infection of theirs is a dead end with rel_trans = 0)
        g1 = np.isin(src, coh) & ~from_seed
        o_src, o_tgt, o_t, o_first = src[g1], tgt[g1], t[g1], first[g1]
        o_tinf = np.array([tinf[s] for s in o_src.tolist()], dtype=np.int64)
        o_A = _pair_present(self, o_src, o_tgt, o_tinf)
        pos = np.searchsorted(coh, o_src)

        n_c = len(coh)
        offA = np.bincount(pos[o_first & o_A], minlength=n_c)
        offB = np.bincount(pos[o_first & ~o_A], minlength=n_c)
        allA = np.bincount(pos[o_A], minlength=n_c)
        allB = np.bincount(pos[~o_A], minlength=n_c)
        cls = np.where(coh_female, 2, 0) + np.where(coh_type_A, 0, 1)  # index into TYPES

        def build(sel, a, b):
            K = np.zeros((4, 4))
            n = np.zeros(4)
            for X in range(4):
                m = sel[cls[sel] == X]
                n[X] = len(m)
                if len(m) == 0:
                    continue
                rows = (2, 3) if X < 2 else (0, 1)  # male -> female types, female -> male types
                K[rows[0], X] = a[m].mean()
                K[rows[1], X] = b[m].mean()
            return K, n

        allsel = np.arange(n_c)
        K, n_by_type = build(allsel, offA, offB)
        K_all, _ = build(allsel, allA, allB)
        boot = []
        rng = np.random.default_rng(0)
        for _ in range(n_boot):
            sel = np.concatenate([rng.choice(np.flatnonzero(cls == X), (cls == X).sum())
                                  for X in range(4) if (cls == X).any()])
            boot.append(spectral_radius(build(sel, offA, offB)[0]))
        R_sex = {s: (offA + offB)[coh_female == (s == 'f')].mean() for s in ('m', 'f')}
        return dict(K=K, R0=spectral_radius(K), R0_se=float(np.std(boot)),
                    R0_ci=np.percentile(boot, [2.5, 97.5]).tolist(),
                    K_all=K_all, R0_all=spectral_radius(K_all), n_by_type=n_by_type,
                    R_m_cohort=R_sex['m'], R_f_cohort=R_sex['f'],
                    n_seeds=len(self.seeds), prev_at_end_cohort=self.prev_at_end_cohort,
                    cases=dict(person=coh, infector=coh_src, t=coh_t, type=cls, offA=offA, offB=offB))


def _pair_present(rec, a, b, t):
    ''' For each (person a_i, person b_i, step t_i): did the pair have a partnership at step t_i? '''
    a, b, t = np.asarray(a), np.asarray(b), np.asarray(t)
    npp = len(rec.is_female) + 1
    fa = rec.is_female[a]
    f = np.where(fa, a, b)
    m = np.where(fa, b, a)
    q = pd.DataFrame(dict(i=np.arange(len(a)), key=f * npp + m, t=t))
    E = pd.DataFrame(dict(key=rec.f * npp + rec.m, first=rec.first, last=rec.last))
    X = q.merge(E, on='key', how='left')
    X['hit'] = (X['first'] <= X['t']) & (X['last'] >= X['t'])
    return X.groupby('i')['hit'].any().reindex(np.arange(len(a)), fill_value=False).to_numpy()


# =====================================================================
# Two-type estimators from recorded timelines
# =====================================================================

def _events(rec, sex, window_steps):
    '''
    Candidate index cases of sex `sex`:
      A: every (person, partner) pair current at t0, infected at t0 through that partnership
      B: every pair whose first partnership starts in (t0, t0+window], infected at its start
    Returns dict of arrays: person, partner, s (infection step), type ('A'/'B').
    '''
    t0 = rec.t0
    npp = len(rec.is_female) + 1
    side = rec.f if sex == 'f' else rec.m
    other = rec.m if sex == 'f' else rec.f
    key = side * npp + other
    cur_keys = np.unique(key[(rec.first <= t0) & (rec.last >= t0)])
    df = pd.DataFrame(dict(key=key, first=rec.first)).groupby('key')['first'].min()
    newk = df[(df.index.isin(cur_keys) == False) & (df > t0) & (df <= t0 + window_steps)]
    keys = np.concatenate([cur_keys, newk.index.to_numpy()])
    s = np.concatenate([np.full(len(cur_keys), t0), newk.to_numpy()])
    typ = np.concatenate([np.zeros(len(cur_keys), dtype=np.int8), np.ones(len(newk), dtype=np.int8)])
    return dict(person=keys // npp, partner=keys % npp, s=s.astype(np.int64), B=typ)


def estimate_r0(rec, sim, n_rep=5, window_years=1.0, n_closed=200_000, t_end=None):
    '''
    Two-type next-generation matrix for every genotype in the sim, two ways:
      'window': for each candidate index case (see _events) draw an infectious period and credit
                1 - prod(1 - T) for each distinct other partner over the ACTUAL partnership
                timelines (so no assumptions about durations or formation); partners already
                current at infection are type-A offspring, later ones type B.
      'closed': the closed-form entries K[A<-X] = sum_tau S^X_tau E[T_cur],
                K[B<-X] = sum_tau C^X_tau E[T_new], with S^X, C^X and q measured from the same
                events over window_years and T averaged over sampled D and the layer's acts.
    Returns {genotype: {'window': {K, R0}, 'closed': {K, R0}}}, plus the measured network inputs.
    '''
    dt = sim['dt']
    W = int(round(window_years / dt))
    t_end = rec.t_end if t_end is None else t_end
    npp = len(rec.is_female) + 1
    nl = len(rec.lkeys)

    # Per-layer dissolution rate q (per year) from survival of t0's partnerships over the window
    cur0 = (rec.first <= rec.t0) & (rec.last >= rec.t0)
    q = {}
    for li, lkey in enumerate(rec.lkeys):
        sel = cur0 & (rec.layer == li)
        surv = ((rec.last >= rec.t0 + W) & sel).sum() / max(sel.sum(), 1)
        q[lkey] = -np.log(max(surv, 1e-12)) / window_years

    prepared = {}
    stats = {}
    for sex in ('m', 'f'):
        ev = _events(rec, sex, W)
        side = rec.f if sex == 'f' else rec.m
        other = rec.m if sex == 'f' else rec.f
        E = pd.DataFrame(dict(person=side, partner=other, first=rec.first, last=rec.last,
                              layer=rec.layer, e=np.arange(len(side))))
        evdf = pd.DataFrame(dict(ev=np.arange(len(ev['s'])), person=ev['person'], j=ev['partner'],
                                 s=ev['s'], B=ev['B']))
        X = evdf.merge(E, on='person')
        X = X[(X['partner'] != X['j']) & (X['last'] >= X['s'])]  # other partners, not over yet
        pk = X['ev'].to_numpy() * npp + X['partner'].to_numpy()
        upk, pair_idx = np.unique(pk, return_inverse=True)
        s_row = X['s'].to_numpy()
        first_row = X['first'].to_numpy()
        last_row = X['last'].to_numpy()
        cur_row = (first_row <= s_row) & (last_row >= s_row)
        pair_cur = np.bincount(pair_idx, weights=cur_row, minlength=len(upk)) > 0
        pair_ev = upk // npp
        # Closed-form inputs per event: other current partnerships by layer at s, and new
        # partners (pairs first seen in (s, s+W]) by layer
        n_ev = len(evdf)
        S_ev = np.zeros((n_ev, nl))
        C_ev = np.zeros((n_ev, nl))
        lay = X['layer'].to_numpy()
        ev_row = X['ev'].to_numpy()
        pair_first = np.full(len(upk), np.iinfo(np.int64).max)
        np.minimum.at(pair_first, pair_idx, first_row)
        is_first_edge = first_row == pair_first[pair_idx]
        for li in range(nl):
            S_ev[:, li] = np.bincount(ev_row[cur_row & (lay == li)], minlength=n_ev)
            newsel = (~pair_cur[pair_idx]) & is_first_edge & (first_row > s_row) & (first_row <= s_row + W) & (lay == li)
            C_ev[:, li] = np.bincount(ev_row[newsel], minlength=n_ev) / window_years
        # women's ages at infection, for dur_cin's age multiplier
        ages = rec.age0[evdf['person'].to_numpy()] + (evdf['s'].to_numpy() - rec.t0) * dt
        prepared[sex] = dict(evdf=evdf, X_e=X['e'].to_numpy(), s_row=s_row, first_row=first_row,
                             last_row=last_row, pair_idx=pair_idx, pair_cur=pair_cur, pair_ev=pair_ev,
                             row_ev=ev_row, n_ev=n_ev, ages=ages)
        Bmask = evdf['B'].to_numpy() == 1
        stats[sex] = {}
        for X_type, mask in (('A', ~Bmask), ('B', Bmask)):
            stats[sex][X_type] = dict(
                n_events=int(mask.sum()),
                S={lkey: float(S_ev[mask, li].mean()) for li, lkey in enumerate(rec.lkeys)},
                C={lkey: float(C_ev[mask, li].mean()) for li, lkey in enumerate(rec.lkeys)})
    stats['q'] = q

    out = {}
    cur_edge = cur0
    new_edge = rec.first > rec.t0
    for g in range(sim['n_genotypes']):
        Kw = np.zeros((4, 4))
        Kc = np.zeros((4, 4))
        for sex, direction in (('m', 'm2f'), ('f', 'f2m')):
            P = np.empty(len(rec.eid))
            for li, lkey in enumerate(rec.lkeys):
                sel = rec.layer == li
                P[sel] = step_prob(per_act_prob(sim, g, lkey, direction), rec.acts[sel], dt)
            logq_step = np.log1p(-np.minimum(P, 1 - 1e-15))
            pr = prepared[sex]
            evdf = pr['evdf']
            Bmask = evdf['B'].to_numpy() == 1
            cols = (0, 1) if sex == 'm' else (2, 3)   # parent types A_s, B_s
            rows = (2, 3) if sex == 'm' else (0, 1)   # offspring types A_s', B_s'

            # --- window estimator ---
            offA = np.zeros(pr['n_ev'])
            offB = np.zeros(pr['n_ev'])
            for _ in range(n_rep):
                m_ev = sample_transmitting_steps(sim, g, sex, pr['n_ev'],
                                                 ages=pr['ages'] if sex == 'f' else None)
                s_row = pr['s_row']
                end = np.minimum(s_row + m_ev[pr['row_ev']], t_end)  # last transmitting step
                lo = np.maximum(pr['first_row'], s_row + 1)
                hi = np.minimum(pr['last_row'], end)
                overlap = np.clip(hi - lo + 1, 0, None)
                log_nt = np.bincount(pr['pair_idx'], weights=overlap * logq_step[pr['X_e']],
                                     minlength=len(pr['pair_cur']))
                T_pair = -np.expm1(log_nt)
                offA += np.bincount(pr['pair_ev'][pr['pair_cur']], weights=T_pair[pr['pair_cur']], minlength=pr['n_ev'])
                offB += np.bincount(pr['pair_ev'][~pr['pair_cur']], weights=T_pair[~pr['pair_cur']], minlength=pr['n_ev'])
            offA /= n_rep
            offB /= n_rep
            for c, mask in zip(cols, (~Bmask, Bmask)):
                Kw[rows[0], c] = offA[mask].mean() if mask.any() else 0
                Kw[rows[1], c] = offB[mask].mean() if mask.any() else 0

            # --- closed form ---
            lam = hazard_from_step_prob(P, dt)
            D = sample_transmitting_steps(sim, g, sex, n_closed,
                                          ages=np.random.choice(pr['ages'][~np.isnan(pr['ages'])], n_closed)
                                          if sex == 'f' else None) * dt
            Tc, Tn = {}, {}
            for li, lkey in enumerate(rec.lkeys):
                lc = lam[cur_edge & (rec.layer == li)]
                ln = lam[new_edge & (rec.layer == li)]
                Tc[lkey] = T_current(np.random.choice(lc, n_closed), q[lkey], D).mean() if len(lc) else 0
                Tn[lkey] = T_new(np.random.choice(ln, n_closed), q[lkey], D).mean() if len(ln) else 0
            for c, X_type in zip(cols, ('A', 'B')):
                st = stats[sex][X_type]
                Kc[rows[0], c] = sum(st['S'][l] * Tc[l] for l in rec.lkeys)
                Kc[rows[1], c] = sum(st['C'][l] * Tn[l] for l in rec.lkeys)
        out[genotype_key(sim, g)] = dict(window=dict(K=Kw, R0=spectral_radius(Kw)),
                                         closed=dict(K=Kc, R0=spectral_radius(Kc)))
    return out, stats


def event_offspring(rec, sim, g, sex, person, partner, s, n_rep=20, t_end=None):
    '''
    Expected (type-A, type-B) offspring of index cases of sex `sex` infected at steps s through
    their partnership with `partner`, averaged over n_rep infectious-period draws. The window
    estimator's core, exposed for diagnostics (e.g. evaluating it on the ground-truth cases).
    '''
    dt = sim['dt']
    t_end = rec.t_end if t_end is None else t_end
    npp = len(rec.is_female) + 1
    side = rec.f if sex == 'f' else rec.m
    other = rec.m if sex == 'f' else rec.f
    direction = 'f2m' if sex == 'f' else 'm2f'
    P = np.empty(len(rec.eid))
    for li, lkey in enumerate(rec.lkeys):
        sel = rec.layer == li
        P[sel] = step_prob(per_act_prob(sim, g, lkey, direction), rec.acts[sel], dt)
    logq_step = np.log1p(-np.minimum(P, 1 - 1e-15))
    E = pd.DataFrame(dict(person=side, partner=other, first=rec.first, last=rec.last, e=np.arange(len(side))))
    evdf = pd.DataFrame(dict(ev=np.arange(len(s)), person=person, j=partner, s=s))
    X = evdf.merge(E, on='person')
    X = X[(X['partner'] != X['j']) & (X['last'] >= X['s'])]
    upk, pair_idx = np.unique(X['ev'].to_numpy() * npp + X['partner'].to_numpy(), return_inverse=True)
    s_row, first_row, last_row = X['s'].to_numpy(), X['first'].to_numpy(), X['last'].to_numpy()
    pair_cur = np.bincount(pair_idx, weights=(first_row <= s_row) & (last_row >= s_row), minlength=len(upk)) > 0
    pair_ev = upk // npp
    row_ev = X['ev'].to_numpy()
    ages = rec.age0[np.asarray(person)] + (np.asarray(s) - rec.t0) * dt
    n_ev = len(s)
    offA = np.zeros(n_ev)
    offB = np.zeros(n_ev)
    for _ in range(n_rep):
        m_ev = sample_transmitting_steps(sim, g, sex, n_ev, ages=ages if sex == 'f' else None)
        end = np.minimum(s_row + m_ev[row_ev], t_end)
        overlap = np.clip(np.minimum(last_row, end) - np.maximum(first_row, s_row + 1) + 1, 0, None)
        T_pair = -np.expm1(np.bincount(pair_idx, weights=overlap * logq_step[X['e'].to_numpy()], minlength=len(upk)))
        offA += np.bincount(pair_ev[pair_cur], weights=T_pair[pair_cur], minlength=n_ev)
        offB += np.bincount(pair_ev[~pair_cur], weights=T_pair[~pair_cur], minlength=n_ev)
    return offA / n_rep, offB / n_rep

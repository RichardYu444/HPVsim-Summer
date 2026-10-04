"""
reff_hpv.py
===========

Genotype-specific reproduction numbers in HPVsim, and how they move over a simulation with
vaccination. Builds on r0_hpv.py (read that first): same partnership recorder, same per-step
transmission probabilities, same two-type (existing vs new partner) next-generation matrix.
What is new here is immunity, current infection, and reinfection.


1. What is being computed
-------------------------
For genotype g and a case i (sex s, infected on step t_i), the number of infections i causes is

    N_i = sum over i's partners k of N_ik,

where N_ik is the number of times i infects k while i is infectious. Two ways of counting:

    all-events  every infection of k by i, including reinfecting the same partner after k clears
                (the "ping-pong" in long partnerships). This is the case reproduction number: in a
                stationary epidemic every infection has exactly one infector, so its mean over
                cases is exactly 1 at endemic equilibrium. It is the quantity whose crossing of 1
                decides whether a genotype grows or declines.
    distinct    at most one per partner, infector excluded -- r0_hpv.py's convention (R0 there is
                distinct people). Below 1 at equilibrium, because it leaves out reinfection.

The next-generation matrix K_g(t) has types (A_m, B_m, A_f, B_f): A = infected by a partner the
case already had, B = infected at the start of a new partnership (see r0_hpv.py section 2), and

    R_g(t) = spectral radius of K_g(t),     K[Y <- X] = E_{X-cases at t}[ #type-Y infections caused ].


2. The transmission model HPVsim actually runs (Sim.step, People.infect/check_clearance)
----------------------------------------------------------------------------------------
On each step u a partnership (i infectious, k not infected with g) transmits with probability

    p_ik(u) = P_e * (1 - sus_imm_g[k](u))

    P_e          = 1 - (1-p)^floor(a dt) (1 - frac(a dt) p),  p = beta rel_beta_g trans_dir (1 - condom eff)
                   (r0_hpv.step_prob); ~0.9-1 per step here, i.e. SATURATED.
    sus_imm_g[k] = min(1,  C[g,g] nab_g[k]                     own immunity (women only, see below)
                        + sum_{h != g} C[g,h] nab_h[k]          natural cross-immunity (women only)
                        + sum_v C[g,v] nab_v[k])                vaccine: C[g,v] = rel_imm (products_vx.csv)

    C = cross_immunity_sus: 1.0 own for hpv16/18, own_imm_hr=0.9 for hi5/ohr, cross_imm_sus_med=0.3,
    cross_imm_sus_high=0.5 (16<->18). nab = peak antibody level, never wanes (use_waning=False).

Durations. A man clears after ceil(D/dt) steps, D ~ lognormal(1, 1) -- no immunity term. A woman's
pre-CIN period is dur_precin * (1 - sev_imm_g[i]) and her CIN probability is a function of that
period, so her whole infection shortens with

    sev_imm_g[i] = min(1, C'[g,g] cell_g[i] + sum_{h != g} C'[g,h] cell_h[i])   (C' = cross_immunity_sev:
                                                                                 cross 0.5 / high 0.7)

Vaccines never set cell immunity, so they never shorten an infection.

IMPORTANT: People.check_clearance calls update_peak_immunity for CLEARED WOMEN ONLY. Men never gain
natural immunity of any kind; the only thing that reduces a man's susceptibility is vaccination.
So natural own- and cross-immunity act only through women: their susceptibility, and the length of
their infections.

Saturation matters for how immunity enters. In a partnership lasting many steps a partial immunity
of 0.2-0.3 only delays infection by a step or so (1-(1-P)^n with P~1), so immunity is NOT a
multiplicative factor on R -- it bites on short partnerships and at the end of an infectious period.
That is why the estimator below works partnership by partnership rather than as R0 x S.


3. The analytic estimator (expected_offspring)
----------------------------------------------
Given a set of index cases (person, sex, infection step t_i, infector j) and the population's state
at the start of the cohort year (a StateSnapshots snapshot), for each partner k of i over i's
infectious period, on the ACTUAL partnership timeline (R0Recorder):

  * i's infectious period: sampled from the genotype's duration distribution with i's sev_imm_g
    at infection (women; TxLog records it) -- i's realised infection length is not used;
  * k's availability: if k is infected at the snapshot, from k's clearance date on; if k is i's
    infector, from the infector's clearance on;
  * then renewal cycles: time to infection ~ Geometric(p_tot), k infectious for a sampled duration
    (women: with their sev_imm), k's immunity updated as HPVsim does on clearance (women:
    seroconvert w.p. sero_prob -> nab ~ imm_init, cell ~ cell_imm_init; men: nothing), repeat until
    the overlap ends. Vaccinations after the snapshot are applied at their real date (vaccination
    is policy, not epidemic).
  * COMPETITION. k's other partners l also infect k, and HPVsim credits one source per step (layers
    in order, so a spouse wins a tie against a casual partner). Per step
        p_tot = 1 - (1 - p_ik)(1 - q_before)(1 - q_same)(1 - q_after),
        credit to i = p_ik (1 - q_before)(1 - q_same / 2) / p_tot,
    where q_* combine k's other partners in earlier / the same / later layers, each infectious as
    known at the snapshot (until they clear), then with the fraction of the last 10 years they spent
    infected (infected_history -- this carries the core/periphery heterogeneity), and, right after
    k clears an infection, with certainty unless protected (k gave it to them and they have been
    ping-ponging it back). A partner whose partnership with i starts after the snapshot may already
    be infected then.

Without competition the estimator over-predicts badly (the verification's analytic_no_comp): the
bulk of all-events offspring are reinfections in long partnerships, and these are shared with the
partner's other partners.

R0_g is the same calculation with everyone immune-naive and uninfected (so no competition), over
the uniform index events of r0_hpv's window estimator (every current partnership / every new one in
a year).


4. Mean-field shortcut (mean_field_reff)
----------------------------------------
The textbook formula R_eff = R0 x (effective susceptible fraction), per sex for a heterosexual
two-type system:

    R_mf,g(t) = R0_g * sqrt( sigma_f(t) sigma_m(t) delta_f(t) )

    sigma_s  = partner-weighted mean over sex s of (1 - sus_imm_g)(1 - currently infected with g)
    delta_f  = mean transmitting duration of a woman newly infected now / of a naive woman

It is reported for comparison: because of saturation (above) it overstates what partial immunity
does, and because of reinfection it overstates what current infection does.


5. Verification
---------------
TxLog records every transmission with the infector's infection episode, so each cohort case's
realised offspring (all-events and distinct, typed A/B) can be counted exactly and turned into an
empirical K. run_reff_verification.py runs the comparison; plot_reff_verification.py draws it.
"""
import numpy as np
import pandas as pd
import sciris as sc

import hpvsim_working as hpv
from hpvsim_working import utils as hpu
from hpvsim_working import parameters as hppar
import r0_hpv

BIG = np.iinfo(np.int64).max // 4
TYPES = r0_hpv.TYPES  # ('A_m', 'B_m', 'A_f', 'B_f')


# =====================================================================
# Analyzers
# =====================================================================

class TxLog(hpv.Analyzer):
    '''
    Every transmission, with the infector's current infection episode. Needs track_transmission=True.

    Columns: t (step), g, src, tgt, src_start (step the infector's current episode began),
    src_clear / src_cancer (infector's clearance / cancer dates), tgt_clear / tgt_cancer (the new
    case's, set by People.infect on this step), layer (index into the contact layers), tgt_sev /
    tgt_sus (the new case's sev_imm / sus_imm for g on this step -- check_immunity runs before
    transmission, so this is the sev_imm set_prognoses used for the infection's length).
    '''

    def initialize(self, sim):
        super().initialize(sim)
        self.chunks = []
        self.lkeys = list(sim.people.contacts.keys())
        return

    def apply(self, sim):
        ppl = sim.people
        for sources, targets, _scale, g, lkey in ppl.new_transmissions:
            n = len(targets)
            if n == 0:
                continue
            self.chunks.append(np.column_stack([
                np.full(n, sim.t), np.full(n, g), sources, targets,
                ppl.date_infectious[g, sources], ppl.date_clearance[g, sources],
                ppl.date_cancerous[g, sources],
                ppl.date_clearance[g, targets], ppl.date_cancerous[g, targets],
                np.full(n, self.lkeys.index(lkey)),
                ppl.sev_imm[g, targets], ppl.sus_imm[g, targets],
            ]).astype(float))
        return

    def finalize(self, sim):
        super().finalize()
        cols = ['t', 'g', 'src', 'tgt', 'src_start', 'src_clear', 'src_cancer', 'tgt_clear',
                'tgt_cancer', 'layer', 'tgt_sev', 'tgt_sus']
        arr = np.concatenate(self.chunks) if self.chunks else np.empty((0, len(cols)))
        df = pd.DataFrame(arr, columns=cols)
        for c in ('t', 'g', 'src', 'tgt', 'layer'):
            df[c] = df[c].astype(np.int64)
        df['src_start'] = np.round(df['src_start']).astype(np.int64)
        self.df = df
        self.chunks = None
        return


class StateSnapshots(hpv.Analyzer):
    '''
    At the end of the step nearest each requested year, every person's immunity broken down by
    source, infection status and clearance dates; at the end of the sim, vaccination dates and the
    vaccine-derived immunity each person ends up with (so vaccinations after a snapshot can be
    applied at their real date).
    '''

    def __init__(self, years, **kwargs):
        super().__init__(**kwargs)
        self.years = list(years)
        return

    def initialize(self, sim):
        super().initialize(sim)
        self.steps = {int(sc.findnearest(sim.yearvec, y)): y for y in self.years}
        self.snaps = {}
        return

    @staticmethod
    def _decompose(sim):
        ppl = sim.people
        ng = sim['n_genotypes']
        C, Cs = sim['cross_immunity_sus'], sim['cross_immunity_sev']
        nab, cell = ppl.nab_imm, ppl.cell_imm
        nsrc = nab.shape[0]
        out = {k: np.zeros((ng, nab.shape[1]), dtype=np.float32)
               for k in ('own_sus', 'cross_sus', 'vx_sus', 'own_sev', 'cross_sev', 'vx_sev')}
        for g in range(ng):
            out['own_sus'][g] = C[g, g] * nab[g]
            out['own_sev'][g] = Cs[g, g] * cell[g]
            for h in range(ng):
                if h != g:
                    out['cross_sus'][g] += C[g, h] * nab[h]
                    out['cross_sev'][g] += Cs[g, h] * cell[h]
            for v in range(ng, nsrc):
                out['vx_sus'][g] += C[g, v] * nab[v]
                out['vx_sev'][g] += Cs[g, v] * cell[v]
        out['nab'] = nab[:ng].astype(np.float32).copy()
        out['cell'] = cell[:ng].astype(np.float32).copy()
        return out

    def apply(self, sim):
        if sim.t not in self.steps:
            return
        ppl = sim.people
        s = self._decompose(sim)
        s.update(step=sim.t, year=self.steps[sim.t], alive=ppl.alive.copy(),
                 female=ppl.is_female.copy(), age=ppl.age.astype(np.float32).copy(),
                 debut=ppl.debut.astype(np.float32).copy(),
                 infectious=ppl.infectious.copy(),
                 clear=ppl.date_clearance.astype(np.float64).copy(),
                 cancer=ppl.date_cancerous.astype(np.float64).copy(),
                 sus_imm=ppl.sus_imm[:sim['n_genotypes']].astype(np.float32).copy(),
                 sev_imm=ppl.sev_imm[:sim['n_genotypes']].astype(np.float32).copy())
        self.snaps[sim.t] = s
        return

    def finalize(self, sim):
        super().finalize()
        ppl = sim.people
        d = self._decompose(sim)
        self.vx_sus_end = d['vx_sus']
        self.date_vacc = ppl.date_vaccinated.astype(np.float64).copy()
        self.age_end = ppl.age.astype(np.float64).copy()
        self.alive_end = ppl.alive.copy()
        self.female_end = ppl.is_female.copy()
        self.t_end = sim.t
        return


# =====================================================================
# Durations
# =====================================================================

def _genotype_pars(pars, g):
    return pars['genotype_pars'][pars['genotype_map'][g]]


def sample_clear_steps(pars, g, female, sev=None, ages=None):
    '''
    Steps from infection to clearance (People.infect / set_prognoses / set_severity): a case
    infected on step t clears on t + d, is susceptible again from that step, and transmits on steps
    t+1 .. t+d-1. Women: dur_precin * (1 - sev_imm) [+ dur_cin * age_mod if CIN]. Progression to
    cancer is ignored (rare). Vectorised over people; `female` is a bool array.
    '''
    dt = pars['dt']
    female = np.asarray(female, bool)
    n = len(female)
    d = np.zeros(n, dtype=np.int64)
    nm = int((~female).sum())
    if nm:
        D = hpu.sample(**pars['dur_infection_male'], size=nm)
        d[~female] = np.ceil(D / dt).astype(np.int64)
    nf = int(female.sum())
    if nf:
        gp = _genotype_pars(pars, g)
        sv = np.zeros(nf) if sev is None else np.asarray(sev, float)[female]
        dur_precin = hpu.sample(**gp['dur_precin'], size=nf) * (1 - np.clip(sv, 0, 1))
        rel_sev = hpu.sample(**pars['sev_dist'], size=nf)
        cin = np.random.random(nf) < hppar.compute_severity(dur_precin, rel_sev=rel_sev, pars=gp['cin_fn'])
        age_mod = np.ones(nf)
        if ages is not None:
            a = np.asarray(ages, float)[female]
            age_mod[a >= pars['age_risk']['age']] = pars['age_risk']['risk']
        D = dur_precin + cin * hpu.sample(**gp['dur_cin'], size=nf) * age_mod
        x = D / dt
        d[female] = (np.floor(x) + (np.random.random(nf) < np.mod(x, 1))).astype(np.int64)
    return np.maximum(d, 1)


# =====================================================================
# Context: everything the estimator needs, detached from the Sim
# =====================================================================

def make_context(sim, rec, snaps):
    ''' Plain-data view of one finished sim: recorder arrays, snapshots, parameters. '''
    keys = ['dt', 'beta', 'transf2m', 'transm2f', 'condoms', 'eff_condoms', 'genotype_pars',
            'genotype_map', 'n_genotypes', 'dur_infection_male', 'sev_dist', 'age_risk', 'acts',
            'imm_init', 'cell_imm_init']
    pars = {k: sc.dcp(sim.pars[k]) for k in keys}
    C, Cs = sim['cross_immunity_sus'], sim['cross_immunity_sev']
    pars['own_sus'] = np.array([C[g, g] for g in range(sim['n_genotypes'])])
    pars['own_sev'] = np.array([Cs[g, g] for g in range(sim['n_genotypes'])])
    npp = len(rec.is_female) + 1
    return dict(pars=pars, rec=rec, snaps=snaps, npp=npp, t_end=rec.t_end,
                female=np.asarray(rec.is_female, bool))


def _pad(a, n, fill):
    a = np.asarray(a)
    if len(a) >= n:
        return a[:n]
    pad = np.full((n - len(a),) + a.shape[1:], fill, dtype=a.dtype if a.dtype != bool else bool)
    return np.concatenate([a, pad])


def person_state(ctx, snap, g, people, variant='actual'):
    '''
    Immunity/infection state of `people` for genotype g as of `snap` (None => immune-naive and
    uninfected, as for R0). Returns arrays: nab, cell (own), cross_sus, cross_sev, vx, t_vacc,
    avail (first step the person can be infected), age (at the snapshot, NaN if unknown).

    variant: 'actual'; 'no_cross' -- natural cross-immunity (sus and sev) removed;
             'no_vx' -- vaccine immunity removed;
             'vx_only' -- with snap=None: vaccine immunity only (no natural immunity, nobody
                          infected) -> the vaccine-only reproduction number R_vx.
    With snap=None and any other variant everyone is immune-naive (R0).
    '''
    ss = ctx['snaps']
    people = np.asarray(people, np.int64)
    n = len(people)
    out = dict(nab=np.zeros(n), cell=np.zeros(n), cross_sus=np.zeros(n), cross_sev=np.zeros(n),
               vx=np.zeros(n), t_vacc=np.full(n, np.inf), avail=np.full(n, -BIG, dtype=np.int64),
               age=np.full(n, np.nan))
    # Vaccination is exogenous: take each person's final vaccine immunity from its real date
    N_end = len(ss.date_vacc)
    ok = people < N_end
    tv = np.full(n, np.inf)
    tv[ok] = np.where(np.isnan(ss.date_vacc[people[ok]]), np.inf, ss.date_vacc[people[ok]])
    vx = np.zeros(n)
    vx[ok] = ss.vx_sus_end[g, people[ok]]
    if (snap is not None and variant != 'no_vx') or (snap is None and variant == 'vx_only'):
        out['vx'] = vx
        out['t_vacc'] = tv
    if snap is None:
        return out
    N = len(snap['alive'])
    inn = people < N
    p = people[inn]
    out['nab'][inn] = snap['nab'][g, p]
    out['cell'][inn] = snap['cell'][g, p]
    if variant != 'no_cross':
        out['cross_sus'][inn] = snap['cross_sus'][g, p]
        out['cross_sev'][inn] = snap['cross_sev'][g, p]
    out['age'][inn] = snap['age'][p]
    inf = np.zeros(n, bool)
    inf[inn] = snap['infectious'][g, p] & snap['alive'][p]
    clr = np.full(n, np.nan)
    clr[inn] = snap['clear'][g, p]
    av = np.where(np.isfinite(clr), clr, BIG).astype(np.int64)
    out['avail'][inf] = av[inf]
    return out


def age_at(ctx, snap, people, steps):
    ''' Ages at given steps: from the snapshot if the person was in it, else from the end state. '''
    ss = ctx['snaps']
    dt = ctx['pars']['dt']
    people = np.asarray(people, np.int64)
    steps = np.asarray(steps, float)
    age = np.full(len(people), 20.0)
    if snap is not None:
        N = len(snap['age'])
        inn = people < N
        age[inn] = snap['age'][people[inn]] + (steps[inn] - snap['step']) * dt
        rest = ~inn
    else:
        rest = np.ones(len(people), bool)
    alive_end = rest & (people < len(ss.alive_end))
    alive_end[alive_end] = ss.alive_end[people[alive_end]]
    age[alive_end] = ss.age_end[people[alive_end]] - (ss.t_end - steps[alive_end]) * dt
    return age


# =====================================================================
# The estimator
# =====================================================================

def _pair_rows(ctx, ev, female_index):
    '''
    Partnership rows of index cases of one sex: every edge of the index that is still running at
    its infection step. ev needs columns ev, person, s.
    '''
    rec = ctx['rec']
    side = rec.f if female_index else rec.m
    other = rec.m if female_index else rec.f
    E = pd.DataFrame(dict(person=side, partner=other, first=rec.first, last=rec.last,
                          e=np.arange(len(side))))
    X = ev[['ev', 'person', 's']].merge(E, on='person')
    X = X[X['last'] >= X['s'] + 1]
    npp = ctx['npp']
    key = X['ev'].to_numpy() * npp + X['partner'].to_numpy()
    upk, pair_idx = np.unique(key, return_inverse=True)
    s = X['s'].to_numpy()
    first, last = X['first'].to_numpy(), X['last'].to_numpy()
    cur = (first <= s) & (last >= s)
    return dict(ev=X['ev'].to_numpy(), e=X['e'].to_numpy(), first=first, last=last, s=s,
                pair_idx=pair_idx, pair_ev=upk // npp, pair_k=upk % npp,
                pair_cur=np.bincount(pair_idx, weights=cur, minlength=len(upk)) > 0,
                npair=len(upk))


def infected_history(ctx, tx, snap, g, years=10, min_active_years=1.0):
    '''
    Fraction of their sexually active time in the `years` before the snapshot that each person
    spent infected with g (from TxLog episodes) -- an individual-level predictor of how likely they
    are to be infectious later, which carries the core/periphery heterogeneity a single prevalence
    does not. NaN for people with less than min_active_years of activity in the window.
    '''
    dt = ctx['pars']['dt']
    step = snap['step']
    H = int(round(years / dt))
    e = tx[tx['g'] == g]
    end = np.where(np.isfinite(e['tgt_clear']), e['tgt_clear'],
                   np.where(np.isfinite(e['tgt_cancer']), e['tgt_cancer'], ctx['t_end']))
    ov = np.clip(np.minimum(end, step) - np.maximum(e['t'].to_numpy(), step - H), 0, None)
    npp = ctx['npp']
    inf_steps = np.bincount(e['tgt'].to_numpy(), weights=ov, minlength=npp)[:npp]
    N = len(snap['age'])
    active = np.zeros(npp)
    act_years = np.clip(snap['age'] - snap['debut'], 0, years)
    active[:N] = np.where(snap['alive'] & np.isfinite(act_years), act_years / dt, 0)
    out = np.full(npp, np.nan)
    ok = active >= min_active_years / dt
    out[ok] = np.clip(inf_steps[ok] / active[ok], 0, 1)
    return out


def _gather(ptr, pairs):
    ''' Rows of a pair-sorted table for the given pairs, and each row's position in `pairs`. '''
    counts = ptr[pairs + 1] - ptr[pairs]
    tot = int(counts.sum())
    loc = np.repeat(np.arange(len(pairs)), counts)
    rows = np.repeat(ptr[pairs], counts) + (np.arange(tot) - np.repeat(np.cumsum(counts) - counts, counts))
    return rows, loc


def partner_prevalence(ctx, snap, g, window_steps):
    ''' Partner-weighted prevalence of g by sex (True = women) at the snapshot. '''
    w_all = exposure_weights(ctx, snap['step'], window_steps)
    N = len(snap['alive'])
    w = w_all[:N] * snap['alive']
    inf = snap['infectious'][g]
    out = {}
    for fx, m in ((True, snap['female']), (False, ~snap['female'])):
        ww = w * m
        out[fx] = float((ww * inf).sum() / ww.sum()) if ww.sum() > 0 else 0.0
    return out


def expected_offspring(ctx, g, ev, snap, variant='actual', n_rep=3, fixed_end=None,
                       return_pairs=False, competition=None, window_steps=4, hist=None):
    '''
    Expected offspring of each index case in `ev` for genotype index g (docstring section 3).

    ev columns: ev (0..n-1), person, s (infection step), female (bool), infector,
                infector_avail (step the infector can be reinfected), optional sev (the index's
                sev_imm_g at infection; else taken from the snapshot)
    snap: StateSnapshots snapshot for partners' (and the index's) state, or None for immune-naive
          (R0: no immunity, nobody else infected, no competition).
    variant: 'actual' | 'no_cross' (natural cross-immunity removed) | 'no_vx'
    fixed_end: optional array of last transmitting steps (diagnostic: realised durations).
    competition: model third-party infection of partners (default: on whenever snap is given).
        Each partner k's other partners l are infectious as known at the snapshot (until they
        clear) and with the partner-weighted prevalence of their sex afterwards. Per step, k is
        infected with p_tot = 1 - (1-p_i)(1-q_before)(1-q_same)(1-q_after) and the index is
        credited with p_i (1-q_before)(1-q_same/2) / p_tot, mirroring Sim.step: layers are
        processed in order (a marital partner wins a tie against a casual one) and within a layer
        the first edge wins (taken as 50/50). A partner not infected at the snapshot whose
        partnership with the index starts later may already be infected then, with probability
        prev * (1 - exp(-lag / mean duration)), for a uniform fraction of a fresh infection.

    Returns DataFrame with ev, offA_all, offB_all, offA_dist, offB_dist (means over n_rep).
    '''
    pars = ctx['pars']
    rec = ctx['rec']
    dt = pars['dt']
    t_end = ctx['t_end']
    gp = _genotype_pars(pars, g)
    own_sus, own_sev = pars['own_sus'][g], pars['own_sev'][g]
    if competition is None:
        competition = snap is not None
    if competition:
        prev = partner_prevalence(ctx, snap, g, window_steps)
        tau = {fx: float(sample_clear_steps(pars, g, np.full(20_000, fx)).mean()) for fx in (False, True)}
    res = np.zeros((len(ev), 4))
    pair_out = []
    for female_index in (False, True):
        sub = ev[ev['female'].to_numpy() == female_index].reset_index(drop=True)
        if len(sub) == 0:
            continue
        orig = sub['ev'].to_numpy()             # position in `ev` / the result
        sub = sub.assign(ev=np.arange(len(sub)))  # local event ids for this sex
        rows = _pair_rows(ctx, sub, female_index)
        direction = 'f2m' if female_index else 'm2f'
        rev = 'm2f' if female_index else 'f2m'
        p_act = {li: r0_hpv.per_act_prob(pars, g, lkey, direction) for li, lkey in enumerate(rec.lkeys)}
        p_rev = {li: r0_hpv.per_act_prob(pars, g, lkey, rev) for li, lkey in enumerate(rec.lkeys)}

        def edge_P(e, table=p_act):
            out = np.empty(len(e))
            lay = rec.layer[e]
            for li, pa in table.items():
                sel = lay == li
                out[sel] = r0_hpv.step_prob(pa, rec.acts[e[sel]], dt)
            return out

        P = edge_P(rows['e'])
        log1mP = np.log1p(-np.minimum(P, 1 - 1e-12))
        npair = rows['npair']
        pi = rows['pair_idx']
        k = rows['pair_k']
        pev = rows['pair_ev']
        pair_layer = np.full(npair, 99, dtype=np.int64)
        np.minimum.at(pair_layer, pi, rec.layer[rows['e']].astype(np.int64))

        # partner state (per pair)
        st = person_state(ctx, snap, g, k, variant=variant)
        is_inf = k == sub['infector'].to_numpy()[pev]
        avail = st['avail'].copy()
        avail[is_inf] = sub['infector_avail'].to_numpy()[pev][is_inf]
        kfem = ctx['female'][k]
        s_ev = sub['s'].to_numpy()

        # the index's own sev_imm, which sets a woman's infection length
        isev = np.zeros(len(sub))
        if snap is not None:
            ist = person_state(ctx, snap, g, sub['person'].to_numpy(), variant=variant)
            snap_sev = np.minimum(1, own_sev * ist['cell'] + ist['cross_sev'])
            if 'sev' in sub and np.isfinite(sub['sev'].to_numpy(float)).any():
                at_inf = sub['sev'].to_numpy(float)
                if variant == 'no_cross':  # strip the (snapshot) cross part from the value at infection
                    full = person_state(ctx, snap, g, sub['person'].to_numpy())
                    at_inf = np.clip(at_inf - full['cross_sev'], 0, 1)
                isev = np.where(np.isfinite(at_inf), at_inf, snap_sev)
            else:
                isev = snap_sev
        iage = age_at(ctx, snap, sub['person'].to_numpy(), s_ev)
        kage0 = age_at(ctx, snap, k, s_ev[pev])  # partner age at the index's infection

        if competition:
            # k's OTHER partnerships (l != index): the competing sources
            kside = rec.m if female_index else rec.f
            lside = rec.f if female_index else rec.m
            Ek = pd.DataFrame(dict(k=kside, l=lside, ofirst=rec.first, olast=rec.last,
                                   oe=np.arange(len(kside))))
            PK = pd.DataFrame(dict(pair=np.arange(npair), k=k, i=sub['person'].to_numpy()[pev],
                                   s=s_ev[pev]))
            OP = PK.merge(Ek, on='k')
            OP = OP[(OP['l'] != OP['i']) & (OP['olast'] >= OP['s'] + 1)].sort_values('pair', kind='stable')
            op_pair = OP['pair'].to_numpy()
            op_ptr = np.searchsorted(op_pair, np.arange(npair + 1))
            op_first, op_last = OP['ofirst'].to_numpy(), OP['olast'].to_numpy()
            op_P = edge_P(OP['oe'].to_numpy())
            op_rel = np.sign(rec.layer[OP['oe'].to_numpy()].astype(np.int64) - pair_layer[op_pair])
            stl = person_state(ctx, snap, g, OP['l'].to_numpy())
            op_known = stl['avail']  # infectious (known) until this step; -BIG if not infected
            # induced competition: k passes her/his infection to l, who then competes for k
            op_Prev = edge_P(OP['oe'].to_numpy(), p_rev)  # k -> l
            vxl = np.where(stl['t_vacc'] <= s_ev[pev][op_pair], stl['vx'], 0.0)
            op_sus_l = np.minimum(1.0, own_sus * stl['nab'] + stl['cross_sus'] + vxl)
            # future infectiousness of competitors / partners: their own recent infection history
            # where there is one (infected_history), else the partner-weighted prevalence of their sex
            op_l = OP['l'].to_numpy()
            prev_l = np.full(len(op_l), prev[female_index])  # the competitors share the index's sex
            prev_k = np.where(kfem, prev[True], prev[False])
            if hist is not None:
                hl = hist[op_l]
                prev_l = np.where(np.isfinite(hl), hl, prev_l)
                hk = hist[k]
                prev_k = np.where(np.isfinite(hk), hk, prev_k)
            tau_k = np.where(kfem, tau[True], tau[False])
            vx0 = np.where(st['t_vacc'] <= s_ev[pev], st['vx'], 0.0)
            sus0 = np.minimum(1.0, own_sus * st['nab'] + st['cross_sus'] + vx0)

        acc = np.zeros((len(sub), 4))
        pair_cnt = np.zeros(npair)
        for _ in range(n_rep):
            if fixed_end is not None:
                end = np.minimum(np.asarray(fixed_end)[orig], t_end)
            else:
                d_i = sample_clear_steps(pars, g, np.full(len(sub), female_index), sev=isev, ages=iage)
                end = np.minimum(s_ev + d_i - 1, t_end)
            lo = np.maximum(rows['first'], rows['s'] + 1)
            hi = np.minimum(rows['last'], end[rows['ev']])
            L = hi - lo + 1
            ok = L > 0
            logq = np.bincount(pi[ok], weights=(L * log1mP)[ok], minlength=npair)
            plo = np.full(npair, BIG, dtype=np.int64)
            np.minimum.at(plo, pi[ok], lo[ok])
            phi = np.full(npair, -1, dtype=np.int64)
            np.maximum.at(phi, pi[ok], hi[ok])
            valid = phi >= plo
            span = np.where(valid, phi - plo + 1, 1)
            Peff = np.where(valid, -np.expm1(logq / span), 0.0)

            qb = np.zeros(npair)
            qs = np.zeros(npair)
            qa = np.zeros(npair)
            avail_r = avail.copy()
            if competition and len(op_pair):
                lo_o = np.maximum(op_first, plo[op_pair])
                hi_o = np.minimum(op_last, phi[op_pair])
                L_o = hi_o - lo_o + 1
                ok_o = (L_o > 0) & valid[op_pair]
                known = np.clip(np.minimum(hi_o, op_known - 1) - lo_o + 1, 0, None)
                known = np.minimum(known, np.maximum(L_o, 0))
                xbar = np.where(ok_o, (known + (L_o - known) * prev_l) / np.maximum(L_o, 1), 0.0)
                x = np.minimum(xbar * op_P * (1 - sus0[op_pair]), 1 - 1e-12)
                term = np.where(ok_o, (L_o / span[op_pair]) * np.log1p(-x), 0.0)
                for arr, rel in ((qb, -1), (qs, 0), (qa, 1)):
                    m = op_rel == rel
                    arr[:] = -np.expm1(np.bincount(op_pair[m], weights=term[m], minlength=npair))
            if competition:
                # already infected by someone else when the partnership with the index begins
                lag = plo - snap['step']
                cand = valid & ~is_inf & (st['avail'] == -BIG) & (lag > 0)
                ppre = prev_k * -np.expm1(-np.maximum(lag, 0) / tau_k)
                pre = cand & (np.random.random(npair) < ppre)
                if pre.any():
                    ip = np.flatnonzero(pre)
                    dfresh = sample_clear_steps(pars, g, kfem[ip], sev=None, ages=kage0[ip])
                    avail_r[ip] = plo[ip] + np.ceil(np.random.random(len(ip)) * dfresh).astype(np.int64)

            nab, cell = st['nab'].copy(), st['cell'].copy()
            cnt = np.zeros(npair)
            # k's availability follows an infection of k (so k's other partners caught it from k)
            was_inf = (avail_r > -BIG) & (avail_r >= plo)
            cur = np.maximum(plo, avail_r)
            active = valid & (cur <= phi)
            for _it in range(10_000):
                idx = np.flatnonzero(active)
                if len(idx) == 0:
                    break
                vx_now = np.where(st['t_vacc'][idx] <= cur[idx], st['vx'][idx], 0.0)
                sus = np.minimum(1.0, own_sus * nab[idx] + st['cross_sus'][idx] + vx_now)
                p = Peff[idx] * (1 - sus)
                cb, cs, ca = qb[idx], qs[idx], qa[idx]
                if competition and len(op_pair):
                    wi = np.flatnonzero(was_inf[idx])
                    if len(wi):
                        R, loc = _gather(op_ptr, idx[wi])
                        cur_r = cur[idx[wi]][loc]
                        act = (op_first[R] <= cur_r - 1) & (op_last[R] >= cur_r)
                        x = (1 - op_sus_l[R]) * op_Prev[R] * op_P[R] * (1 - sus[wi][loc])
                        term = np.where(act, np.log1p(-np.minimum(x, 1 - 1e-12)), 0.0)
                        cb, cs, ca = cb.copy(), cs.copy(), ca.copy()
                        for arr, rel in ((cb, -1), (cs, 0), (ca, 1)):
                            m = op_rel[R] == rel
                            qi = -np.expm1(np.bincount(loc[m], weights=term[m], minlength=len(wi)))
                            arr[wi] = 1 - (1 - arr[wi]) * (1 - qi)
                p_tot = 1 - (1 - p) * (1 - cb) * (1 - cs) * (1 - ca)
                credit = np.divide(p * (1 - cb) * (1 - cs / 2), p_tot,
                                   out=np.zeros_like(p), where=p_tot > 0)
                w = np.full(len(idx), BIG, dtype=np.int64)
                pos = p_tot > 0
                w[pos] = np.random.geometric(np.minimum(p_tot[pos], 1.0))
                u = cur[idx] + w - 1
                hit = u <= phi[idx]
                h = idx[hit]
                uh = u[hit]
                cnt[h] += np.random.random(len(h)) < credit[hit]
                # the partner's infection (whoever caused it): duration with their sev immunity...
                fem_h = kfem[h]
                sev_h = np.minimum(1, own_sev * cell[h] + st['cross_sev'][h])
                age_h = kage0[h] + (uh - s_ev[pev[h]]) * dt
                d = sample_clear_steps(pars, g, fem_h, sev=sev_h, ages=age_h)
                # ...then immunity on clearance, women only (People.check_clearance)
                fi = h[fem_h]
                if len(fi):
                    prior = nab[fi] > 0
                    new_nab = hpu.sample(**pars['imm_init'], size=len(fi))
                    new_cell = hpu.sample(**pars['cell_imm_init'], size=len(fi))
                    sero = np.random.random(len(fi)) < gp['sero_prob']
                    nab[fi] = np.where(prior, np.maximum(nab[fi], new_nab), sero * new_nab)
                    cell[fi] = np.where(prior, np.maximum(cell[fi], new_cell), new_cell)
                cur[h] = uh + d
                was_inf[h] = d >= 2  # k was infectious for at least one step
                active[:] = False
                active[h] = cur[h] <= phi[h]
            dist = (cnt > 0) & ~is_inf
            pair_cnt += cnt
            pc = rows['pair_cur']
            acc[:, 0] += np.bincount(pev[pc], weights=cnt[pc], minlength=len(sub))
            acc[:, 1] += np.bincount(pev[~pc], weights=cnt[~pc], minlength=len(sub))
            acc[:, 2] += np.bincount(pev[pc], weights=dist[pc], minlength=len(sub))
            acc[:, 3] += np.bincount(pev[~pc], weights=dist[~pc], minlength=len(sub))
        res[orig] = acc / n_rep
        if return_pairs:
            pair_out.append(pd.DataFrame(dict(ev=orig[pev], partner=k, cnt=pair_cnt / n_rep,
                                              is_infector=is_inf, cur=rows['pair_cur'])))
    if return_pairs:
        return pd.concat(pair_out, ignore_index=True)
    return pd.DataFrame(dict(ev=ev['ev'].to_numpy(), offA_all=res[:, 0], offB_all=res[:, 1],
                             offA_dist=res[:, 2], offB_dist=res[:, 3]))


# =====================================================================
# Index-case sets
# =====================================================================

def uniform_events(ctx, step, window_steps, snap_for_infector=None, g=None):
    '''
    r0_hpv's window events at `step`: every partnership current at `step` (type A, infected at
    `step` by that partner) and every partnership first formed in (step, step+window] (type B,
    infected at its start). Weighted equally -- the R0 construction. The infector's reinfection
    availability is a fresh clearance draw from its own infection (naive).
    '''
    rec = ctx['rec']
    npp = ctx['npp']
    out = []
    for female_index in (False, True):
        side = rec.f if female_index else rec.m
        other = rec.m if female_index else rec.f
        key = side * npp + other
        cur_keys = np.unique(key[(rec.first <= step) & (rec.last >= step)])
        fk = pd.DataFrame(dict(key=key, first=rec.first)).groupby('key')['first'].min()
        newk = fk[(~fk.index.isin(cur_keys)) & (fk > step) & (fk <= step + window_steps)]
        keys = np.concatenate([cur_keys, newk.index.to_numpy()])
        s = np.concatenate([np.full(len(cur_keys), step), newk.to_numpy()]).astype(np.int64)
        typ = np.concatenate([np.zeros(len(cur_keys), np.int8), np.ones(len(newk), np.int8)])
        out.append(pd.DataFrame(dict(person=keys // npp, infector=keys % npp, s=s, B=typ,
                                     female=female_index)))
    ev = pd.concat(out, ignore_index=True)
    ev['ev'] = np.arange(len(ev))
    return ev


def naive_infector_avail(ctx, g, ev, snap):
    ''' Infector reinfectable after a fresh infection of its own, as for an immune-naive case. '''
    fem = ctx['female'][ev['infector'].to_numpy()]
    ages = age_at(ctx, snap, ev['infector'].to_numpy(), ev['s'].to_numpy())
    return ev['s'].to_numpy() + sample_clear_steps(ctx['pars'], g, fem, sev=None, ages=ages)


def cohort_events(ctx, tx, g, step_lo, step_hi):
    '''
    Real cases of genotype g infected on steps (step_lo, step_hi], from TxLog, typed A/B the way
    r0_hpv's ground truth does: A if the infecting partnership already existed when the infector's
    own infection began.
    '''
    rec = ctx['rec']
    c = tx[(tx['g'] == g) & (tx['t'] > step_lo) & (tx['t'] <= step_hi)]
    ev = pd.DataFrame(dict(person=c['tgt'].to_numpy(), s=c['t'].to_numpy(),
                           infector=c['src'].to_numpy(), src_start=c['src_start'].to_numpy(),
                           src_clear=c['src_clear'].to_numpy(), src_cancer=c['src_cancer'].to_numpy(),
                           tgt_clear=c['tgt_clear'].to_numpy(), tgt_cancer=c['tgt_cancer'].to_numpy(),
                           sev=c['tgt_sev'].to_numpy() if 'tgt_sev' in c else np.nan))
    ev['female'] = ctx['female'][ev['person'].to_numpy()]
    typeA = r0_hpv._pair_present(rec, ev['infector'].to_numpy(), ev['person'].to_numpy(),
                                 np.maximum(ev['src_start'].to_numpy(), rec.t0))
    ev['B'] = (~typeA).astype(np.int8)
    ia = np.where(np.isfinite(ev['src_clear']), ev['src_clear'], BIG)
    ev['infector_avail'] = ia.astype(np.int64)
    # realised last transmitting step (diagnostic only)
    endc = np.where(np.isfinite(ev['tgt_clear']), ev['tgt_clear'] - 1,
                    np.where(np.isfinite(ev['tgt_cancer']), ev['tgt_cancer'] - 1, ctx['t_end']))
    ev['real_end'] = endc.astype(np.int64)
    ev['ev'] = np.arange(len(ev))
    return ev


def empirical_offspring(ctx, tx, g, ev):
    ''' Realised offspring of each cohort case (all-events and distinct, typed A/B). '''
    rec = ctx['rec']
    o = tx[tx['g'] == g][['src', 'src_start', 'tgt', 't']]
    m = ev[['ev', 'person', 's', 'infector']].merge(o, left_on=['person', 's'],
                                                    right_on=['src', 'src_start'])
    if len(m):
        A = r0_hpv._pair_present(rec, m['person'].to_numpy(), m['tgt'].to_numpy(), m['s'].to_numpy())
    else:
        A = np.zeros(0, bool)
    m = m.assign(A=A)
    n = len(ev)
    evi = m['ev'].to_numpy()
    offA_all = np.bincount(evi[A], minlength=n)
    offB_all = np.bincount(evi[~A], minlength=n)
    d = m[m['tgt'] != m['infector']].drop_duplicates(['ev', 'tgt'])
    dA = d['A'].to_numpy()
    offA_dist = np.bincount(d['ev'].to_numpy()[dA], minlength=n)
    offB_dist = np.bincount(d['ev'].to_numpy()[~dA], minlength=n)
    return pd.DataFrame(dict(ev=ev['ev'].to_numpy(), offA_all=offA_all, offB_all=offB_all,
                             offA_dist=offA_dist, offB_dist=offB_dist))


# =====================================================================
# From per-case offspring to K and R
# =====================================================================

def k_sums(ev, off):
    '''
    Sufficient statistics for pooling: for each count ('all', 'dist'), sums S[Y, X] of type-Y
    offspring over type-X cases and counts n[X]. K = S / n.
    '''
    cls = np.where(ev['female'].to_numpy(), 2, 0) + ev['B'].to_numpy().astype(int)
    out = {}
    for c in ('all', 'dist'):
        S = np.zeros((4, 4))
        for X in range(4):
            m = cls == X
            rows = (2, 3) if X < 2 else (0, 1)
            S[rows[0], X] = off[f'offA_{c}'].to_numpy()[m].sum()
            S[rows[1], X] = off[f'offB_{c}'].to_numpy()[m].sum()
        out[c] = S
    out['n'] = np.bincount(cls, minlength=4).astype(float)
    return out


def r_from_sums(S, n):
    ''' Spectral radius of K = S/n, plus per-sex mean offspring and their geometric mean. '''
    K = np.divide(S, n[None, :], out=np.zeros_like(S), where=n[None, :] > 0)
    Rm = S[:, :2].sum() / n[:2].sum() if n[:2].sum() else np.nan
    Rf = S[:, 2:].sum() / n[2:].sum() if n[2:].sum() else np.nan
    return dict(rho=r0_hpv.spectral_radius(K), R_m=Rm, R_f=Rf, gen=np.sqrt(Rm * Rf), K=K)


# =====================================================================
# Mean-field shortcut
# =====================================================================

def exposure_weights(ctx, step, window_steps):
    ''' Distinct partners of each person over (step, step+window] -- the chance of being someone's partner. '''
    rec = ctx['rec']
    sel = (rec.first <= step + window_steps) & (rec.last > step)
    pairs = np.unique(rec.f[sel] * ctx['npp'] + rec.m[sel])
    w = np.zeros(ctx['npp'])
    np.add.at(w, pairs // ctx['npp'], 1)
    np.add.at(w, pairs % ctx['npp'], 1)
    return w


def mean_field_reff(ctx, g, snap, R0, window_steps, n_dur=20_000):
    '''
    R0 * sqrt(sigma_f sigma_m delta_f) from population averages (section 4 of the docstring).
    Returns dict with sigma_f, sigma_m, delta_f, R_mf.
    '''
    pars = ctx['pars']
    w_all = exposure_weights(ctx, snap['step'], window_steps)
    N = len(snap['alive'])
    w = w_all[:N] * snap['alive']
    own_sus, own_sev = pars['own_sus'][g], pars['own_sev'][g]
    sus = np.minimum(1, own_sus * snap['nab'][g] + snap['cross_sus'][g] + snap['vx_sus'][g])
    free = 1 - snap['infectious'][g]
    fem = snap['female']
    out = {}
    for sx, m in (('f', fem), ('m', ~fem)):
        ww = w * m
        out[f'sigma_{sx}'] = float((ww * (1 - sus) * free).sum() / ww.sum())
    # delta_f: duration of women infected now (drawn by exposure x susceptibility) vs naive women
    ww = w * fem * (1 - sus) * free
    p = ww / ww.sum()
    pick = np.random.choice(N, n_dur, p=p)
    sev = np.minimum(1, own_sev * snap['cell'][g][pick] + snap['cross_sev'][g][pick])
    ages = snap['age'][pick]
    fem_arr = np.ones(n_dur, bool)
    d_now = sample_clear_steps(pars, g, fem_arr, sev=sev, ages=ages) - 1
    d_naive = sample_clear_steps(pars, g, fem_arr, sev=None, ages=ages) - 1
    out['delta_f'] = float(d_now.mean() / d_naive.mean())
    out['R_mf'] = float(R0 * np.sqrt(out['sigma_f'] * out['sigma_m'] * out['delta_f']))
    return out

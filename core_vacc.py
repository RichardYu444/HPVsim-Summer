"""
core_vacc.py
============

NHS_Vacc.py, but with the sexual-activity CORE GROUP vaccinated at 100%, and general coverage
reduced just enough to keep the total number of doses unchanged.

This is the intervention side of the core-group question that hpvsim_working/analysis.py's
``core_group_attribution`` opened up: at Pareto alpha=3 the top 10% most active agents transmit
24.8% of infections and cause 30.1% of cancers (50 runs,
csvs/powerlaw_alpha3_200k_50runs_summary.txt). If those people were vaccinated first, and the
doses came out of everybody else's allocation rather than out of a bigger budget, how much of
that burden goes away?

Everything not described below is IDENTICAL to NHS_Vacc.py -- same products, same age range, same
years, same sexes, same second/third-dose structure. Only the first-dose allocation changes.


1. Who counts as core
---------------------
The top ``CORE_FRAC`` of the latent partner-formation propensity ``theta`` -- the same quantity
core_group_attribution ranks by, and the same threshold. CORE_FRAC defaults to 10% and is set
from the environment (see below); for Pareto alpha=3 the cut is frac**(-1/3), i.e. theta = 2.1544
for the top 10% and 1.7100 for the top 20%.

theta is normally drawn at sexual debut (~age 16), which is AFTER the 12-13 vaccination window,
so on its own it cannot be used to target vaccination at all. theta_predraw.py fixes theta at
birth instead (deterministic in seed/sex/agent index, same distribution, and the network consumes
the same value at debut), which makes this a PERFECT-FORESIGHT counterfactual: the upper bound on
what targeting could achieve if the top decile were identifiable in advance. Importing this
module installs that patch, so the NHS_Vacc baseline you compare against must be run with
theta_predraw imported too -- see run_core_vacc.py's ARM switch.


2. Holding the dose count fixed
-------------------------------
Per birth cohort per year, with baseline first-dose coverage p1, core fraction c, and conditional
second/third-dose probabilities q2/q3:

    baseline expected doses per head:  p1 * k          where k = 1 + q2 + q2*q3
    core arm:                          [c + (1-c)*p1'] * k

because the core group flows through the SAME unchanged d2/d3 interventions and so earns the same
k. Setting them equal, k cancels:

    p1' = (p1 - c) / (1 - c)

so only the first-dose probability needs rescaling, and the multi-dose structure is irrelevant.
The 2022- first-dose probability of 0.80 becomes 0.7778 at c=0.10 and 0.7500 at c=0.20; running
this file as __main__ prints the full table for whatever CORE_FRAC is set.

The rescaling needs p1 >= c in every programme year, or vaccinating the whole core group would
already cost more doses than the baseline delivers. The binding year is 2020 at p1=0.565 (COVID),
so CORE_FRAC can go up to ~0.56 before _reduce() refuses.

Mechanically no exclusion logic is needed. The core intervention runs FIRST and sets doses=1; the
general first-dose interventions already gate on ``sim.people.doses == 0``, so they skip whoever
the core intervention just caught. Expected first doses per head are then
c*1 + (1-c)*p1' = c + (p1 - c) = p1.

Equality is exact in expectation over a full cohort-year but not per run, for three reasons:
  (a) the baseline spreads p1 over the 4 timesteps an agent spends in [12,13) (RoutineDelivery
      converts to 1-(1-p)**dt), whereas the core arm delivers on the first of them -- so agents
      entering the window part-way through a programme segment do slightly better in the core arm;
  (b) an agent dying between 12 and 13 gets partial exposure in the baseline, full in the core arm;
  (c) year boundaries where p1 changes mid-window.
Measured rather than assumed: at 20k agents to 2030, 3 runs per arm, final cum_doses came out
17,187,948 (core) against 17,182,323 (baseline) -- a 0.033% difference. cum_doses is written to
the output CSV so this stays checkable on the full runs rather than trusted.

Separately, and affecting BOTH arms identically, about 2.8% of any birth cohort is never present
in the population at ages 12-13 while the programme is running (they arrive later). Those agents
are never offered vaccination at all, in either arm, so measured coverage over a whole age band
sits a couple of points below the nominal rate. Among agents actually present in the window, the
delivered coverage is exactly as designed: core 1.0000, non-core 0.7716 against a target of
0.7778 (20k agents, 2022+ cohorts).


3. Girls-only years
-------------------
2008-2018 is a female-only programme and the core rule follows it (those core interventions keep
routine_vx's default sex=0). gamma_shape_U == gamma_shape_V, so both sexes share a theta
distribution and exactly 10% of females sit above the pooled cut -- the formula above holds
per-sex without modification.

Worth noting when reading results: 2020 has the lowest baseline coverage in the whole programme
(0.565, COVID), so it is where the reallocation is largest in relative terms -- at c=0.10 non-core
drops to 0.517 and at c=0.20 to 0.456, while core jumps to 1.0 either way. Expect the effect to
concentrate around that cohort.
"""


#--- Imports ---#
import os

import numpy as np

import hpvsim_working as hpv
import GlobalScreeningParameters as GlobalScreeningParameters
import theta_predraw  # MODULE-LEVEL: installs the pre-debut theta patch (see its docstring)


#--- Core group definition ---#

# Which slice of the activity distribution gets 100% coverage. Read from the environment so a
# sweep needs no edit here, and -- more importantly -- so the value survives process spawn: sciris
# parallelize re-imports this module in each Windows worker, and the environment is inherited
# where a module-level constant set from argv would not be. Every intervention below is built from
# it at import, so it must be set BEFORE core_vacc is imported:
#
#     CORE_FRAC=0.20 python run_core_vacc.py core
#
# Reference points from the 50-run attribution (csvs/powerlaw_alpha3_200k_50runs_summary.txt):
# the top 10% transmit 24.8% of infections and cause 30.1% of cancers; the top 20% transmit 37.6%
# and cause 43.5%. Pareto alpha=3 makes the theta cut exactly frac**(-1/3): 2.1544 at 10%,
# 1.7100 at 20%.
CORE_FRAC = float(os.environ.get('CORE_FRAC', 0.10))
if not 0.0 < CORE_FRAC < 1.0:
    raise ValueError(f'CORE_FRAC must be in (0, 1), got {CORE_FRAC}')

AGE_LO, AGE_HI = 12, 13  # the vaccination window, mirrored from the interventions' age_range


def _reduce(prob, c=CORE_FRAC):
    '''
    Non-core first-dose probability that leaves the expected total dose count unchanged:
    p1' = (p1 - c)/(1 - c). See section 2 of the module docstring for the derivation.
    '''
    p = np.asarray(prob, dtype=float)
    if np.any(p < c):
        errormsg = (f'Cannot hold doses fixed: baseline coverage {p.min():.3f} is below the core '
                    f'fraction {c:.3f}, so vaccinating the whole core group already spends more '
                    f'doses than the baseline programme delivers.')
        raise ValueError(errormsg)
    out = np.atleast_1d((p - c) / (1.0 - c))
    return tuple(out.tolist()) if out.size > 1 else float(out[0])


class CoreEligibility:
    '''
    Eligibility predicate for the core-group interventions: unvaccinated AND in the top
    ``frac`` of theta.

    One instance is shared by all four core interventions rather than one each, so the per-agent
    theta lookup happens once per timestep instead of four times. MultiSim deep-copies the sim
    per run, and deepcopy memoises, so the four references still point at a single object inside
    each run -- and each run gets its own cache, keyed to its own rand_seed.

    Only the age window is ever looked up (~1 birth cohort, a few thousand agents at 200k), and
    each agent only once, so the steady-state cost is the few hundred agents that enter the
    window each timestep.
    '''

    def __init__(self, frac=CORE_FRAC):
        self.frac = float(frac)
        self.cut = None      # resolved on first call, from the installed sampler
        self._theta = None   # per-agent cache, nan = not looked up yet
        self._seed = None    # rand_seed the cache was built under
        return

    def _ensure(self, sim):
        n = len(sim.people)
        seed = int(sim['rand_seed'])
        if self._seed != seed:  # different run: the old cache is meaningless
            self._theta, self._seed, self.cut = None, seed, None
        if self.cut is None:
            self.cut = theta_predraw.core_cut(sim, self.frac)
        if self._theta is None:
            self._theta = np.full(n, np.nan, dtype=np.float64)
        elif len(self._theta) < n:
            grown = np.full(max(n, int(len(self._theta) * 1.5) + 1), np.nan, dtype=np.float64)
            grown[:len(self._theta)] = self._theta
            self._theta = grown
        return n

    def __call__(self, sim):
        people = sim.people
        n = self._ensure(sim)
        theta = self._theta[:n]

        todo = np.where((people.age >= AGE_LO) & (people.age < AGE_HI)
                        & people.alive & np.isnan(theta))[0]
        if todo.size:
            theta[todo] = theta_predraw.theta_for_people(sim, todo)

        known = ~np.isnan(theta)
        out = np.zeros(n, dtype=bool)
        out[known] = theta[known] >= self.cut
        return out & (people.doses == 0)


# A single shared instance -- see the class docstring for why it is not one per intervention
core_eligible = CoreEligibility()

# Unchanged from NHS_Vacc.py
eligible_first_dose = lambda sim: sim.people.doses == 0
eligible_second_dose = lambda sim: sim.people.doses == 1
eligible_third_dose = lambda sim: sim.people.doses == 2


#--- Core-group first doses: 100% coverage, mirroring the programme's era/product/sex splits ---#

core_0811_d1 = hpv.routine_vx(prob=1,
                         start_year=2008,
                         end_year=2011,
                         age_range=[AGE_LO, AGE_HI],
                         product='bivalent',
                         eligibility=core_eligible,
                        )
core_1218_d1 = hpv.routine_vx(prob=1,
                         start_year=2012,
                         end_year=2018,
                         age_range=[AGE_LO, AGE_HI],
                         product='quadrivalent',
                         eligibility=core_eligible,
                        )
core_1921_d1 = hpv.routine_vx(prob=1,
                         start_year=2019,
                         end_year=2021,
                         sex=['f', 'm'],
                         age_range=[AGE_LO, AGE_HI],
                         product='quadrivalent',
                         eligibility=core_eligible,
                        )
core_22XX_d1 = hpv.routine_vx(prob=1,
                         start_year=2022,
                         sex=['f', 'm'],
                         age_range=[AGE_LO, AGE_HI],
                         product='nonavalent',
                         eligibility=core_eligible,
                        )


#--- General first doses: NHS_Vacc.py's, scaled down to pay for the core group ---#

#2008-2011  (0.86, 0.86, 0.86, 0.89) -> (0.8444, 0.8444, 0.8444, 0.8778)
vx_0811_d1 = hpv.routine_vx(prob=_reduce((0.86, 0.86, 0.86, 0.89)),
                         start_year=2008,
                         end_year=2011,
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='bivalent',
                         eligibility=eligible_first_dose,
                        )

#2012-2018  (0.91,0.90,0.90,0.88,0.86,0.86,0.85) -> (0.9000,0.8889,0.8889,0.8667,0.8444,0.8444,0.8333)
vx_1218_d1 = hpv.routine_vx(prob=_reduce((0.91, 0.90, 0.90, 0.88, 0.86, 0.86, 0.85)),
                         start_year=2012,
                         end_year=2018,
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='quadrivalent',
                         eligibility=eligible_first_dose,
                        )

#2019-2021  (0.85, 0.565, 0.74) -> (0.8333, 0.5167, 0.7111)
vx_1921_d1 = hpv.routine_vx(prob=_reduce((0.85, (0.60+0.53)/2, (0.77+0.71)/2)),
                         start_year=2019,
                         end_year=2021,
                         sex=['f','m'],
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='quadrivalent',
                         eligibility=eligible_first_dose,
                        )

#2022 - XX  0.80 -> 0.7778
vx_22XX_d1 = hpv.routine_vx(prob=_reduce(GlobalScreeningParameters.projected_teen_vaccination_uptake),
                         start_year=2022,

                         sex=['f','m'],
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='nonavalent',
                         eligibility=eligible_first_dose,
                        )


#--- Second and third doses: verbatim from NHS_Vacc.py. The dose-neutrality argument in section 2
#    of the module docstring depends on these being untouched, so do not rescale them. ---#

vx_0811_d2 = hpv.routine_vx(prob=(0.78/0.86 ,0.78/0.86 ,0.78/0.86 ,0.83/0.89),
                         start_year=2008,
                         end_year=2011,
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='bivalent2', #2nd dose; see interventions.default_vx() for assumed imm_boost of 1.2
                         eligibility=eligible_second_dose,
                        )
vx_0811_d3 = hpv.routine_vx(prob=1,
                         start_year=2008,
                         end_year=2011,
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='bivalent3',  #3rd dose; assumed imm_boost of 1.1
                         eligibility=eligible_third_dose,
                        )
vx_1218_d2 = hpv.routine_vx(prob=(0.87/0.91,0.84/0.9,0.85/0.9,0.84/0.88,0.83/0.86,0.82/0.86,0.82/0.85),
                         start_year=2012,
                         end_year=2018,
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='quadrivalent2', #2nd dose; assumed imm_boost of 1.2
                         eligibility=eligible_second_dose,
                        )
vx_1213_d3 = hpv.routine_vx(prob=1,
                         start_year=2012,
                         end_year=2013,
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='quadrivalent3',  #3rd dose; assumed imm_boost of 1.1
                         eligibility=eligible_third_dose,
                        )
vx_1921_d2 = hpv.routine_vx(prob=(0.82/0.85,0.64*2/(0.60+0.53),(0.59+0.48)/(0.77+0.71)), #TODO: inherited from NHS_Vacc -- something wrong with this second prob, perhaps combining boys and girls doesnt work here
                         start_year=2019,
                         end_year=2021,
                         sex=['f','m'],
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='quadrivalent2', #2nd dose; assumed imm_boost of 1.2
                         eligibility=eligible_second_dose,
                        )
vx_2222_d2 = hpv.routine_vx(prob=0.5/0.6, #0.7/0.8, #0.5/0.6
                         start_year=2022,
                         end_year=2022,
                         sex=['f','m'],
                         age_range=[12, 13], #(inclusive, exclusive)
                         product='nonavalent2', #2nd dose; assumed imm_boost of 1.2
                         eligibility=eligible_second_dose,
                        )


# ORDER MATTERS, FOR TWO SEPARATE REASONS.
#
# 1. Dose sequencing. The core interventions must come before the general first doses (so the
#    doses==0 gate excludes anyone they just vaccinated -- this is what prevents double counting)
#    and before the second doses (so a core recipient picks up dose 2 in the same timestep,
#    exactly as a general first-dose recipient does in NHS_Vacc.py). Within the tail,
#    NHS_Vacc.py's own relative ordering is preserved.
#
# 2. Immunity indexing, which is a booby trap in the vendored HPVsim. immunity.init_immunity()
#    assigns each vaccine product's imm_source from np.unique(names, return_index=True) -- i.e.
#    the POSITION OF ITS FIRST OCCURRENCE in this list -- but sizes the people.peak_imm array as
#    n_genotypes + n_unique_products. Those only agree if the first nv entries introduce nv
#    DISTINCT products; otherwise a later product gets an index off the end of the array and the
#    run dies mid-flight with an IndexError. (Products are keyed by base name, so 'bivalent2' and
#    'bivalent3' count as 'bivalent'.) NHS_Vacc.py's list happens to satisfy this, which is why
#    its ordering looks arbitrary. Here it means core_22XX_d1 (nonavalent) must come THIRD, ahead
#    of the second quadrivalent entry -- putting core_1921_d1 third instead pushes nonavalent to
#    index 3 and blows up. The four core interventions have disjoint year ranges and identical
#    prob/eligibility, so reordering them among themselves changes nothing else.
#    _check_imm_source_ordering() below enforces this at import rather than 30 minutes into a run.
vaccinations = [
    core_0811_d1, core_1218_d1, core_22XX_d1, core_1921_d1,  # 100% of the top CORE_FRAC

    vx_0811_d1, #2vHPV
    vx_1921_d1, #4vHPV
    vx_22XX_d1, #9vHPV

                vx_0811_d2, vx_0811_d3,
    vx_1218_d1, vx_1218_d2, vx_1213_d3,
                 vx_1921_d2,
                vx_2222_d2,
]


def _check_imm_source_ordering(interventions):
    '''
    Guard for reason 2 in the comment above vaccinations: the first occurrence of each distinct
    vaccine product must land at position 0, 1, ... nv-1, or immunity.init_immunity() hands out an
    imm_source past the end of people.peak_imm and the run dies partway through with an
    IndexError. Cheap, and it fires at import instead of mid-run.
    '''
    names = [iv.product.genotype_pars['name'].values[0] for iv in interventions]
    _, first = np.unique(names, return_index=True)
    if sorted(first.tolist()) != list(range(len(first))):
        errormsg = (
            f'Vaccination list ordering would break HPVsim immunity indexing. Distinct products '
            f'first appear at positions {sorted(first.tolist())}, but they must appear at '
            f'{list(range(len(first)))} -- i.e. the first {len(first)} interventions have to '
            f'introduce {len(first)} different products. Reorder so each new product appears as '
            f'early as possible. Current order: {names}')
        raise ValueError(errormsg)
    return


_check_imm_source_ordering(vaccinations)


if __name__ == '__main__':
    # The dose-neutrality arithmetic, with no simulation involved: for every programme segment,
    # expected first doses per head must be identical between the two arms.
    print(f'Dose-neutral rescaling at CORE_FRAC = {CORE_FRAC:.0%}\n')
    print(f"{'years':<12} {'sex':<5} {'baseline p1':>12} {'non-core p1':>12} "
          f"{'core arm E[d1]':>15} {'baseline E[d1]':>15}")
    segments = [
        ('2008-2011', 'f',   (0.86, 0.86, 0.86, 0.89)),
        ('2012-2018', 'f',   (0.91, 0.90, 0.90, 0.88, 0.86, 0.86, 0.85)),
        ('2019-2021', 'f+m', (0.85, (0.60+0.53)/2, (0.77+0.71)/2)),
        ('2022-',     'f+m', (GlobalScreeningParameters.projected_teen_vaccination_uptake,)),
    ]
    worst = 0.0
    for years, sex, probs in segments:
        reduced = np.atleast_1d(_reduce(probs))
        for p1, p1r in zip(probs, reduced):
            core_arm = CORE_FRAC * 1.0 + (1 - CORE_FRAC) * p1r
            worst = max(worst, abs(core_arm - p1))
            print(f'{years:<12} {sex:<5} {p1:>12.4f} {p1r:>12.4f} {core_arm:>15.6f} {p1:>15.6f}')
    print(f'\nLargest discrepancy across all programme years: {worst:.2e}  '
          f'({"OK" if worst < 1e-12 else "FAIL"})')

    print(f'\n{len(vaccinations)} interventions '
          f'({sum(1 for v in vaccinations if v.eligibility is core_eligible)} core-targeted)')

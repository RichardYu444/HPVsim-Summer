"""
Stores the base parameters for a HPVsim simulation using the current NHS strategy,
wired to the age+community bipartite network (pars['network'] == 'community', see
hpvsim_working/community_network.py) with a power-law (Pareto) partner-propensity
distribution instead of the package's own Gamma default -- see powerlaw.py for the
theta-sampler swap itself and calibrate_community_powerlaw.py for how community_pars
below was tuned. This is the power-law sibling of basePars_community.py (which keeps
the package's own Gamma propensity); everything else (mixing matrices, condoms,
genotype pars, NHS/vaccination wiring, beta/cross_layer) is identical to that file.
"""
import numpy as np
import NHS_2025_lambdamu, NHS_Vacc
from hpvsim_working.parameters import get_genotype_pars
import sciris as sc
import hpvsim_working as hpv

import powerlaw  # Installs the power-law theta sampler as a side effect of import
                 # (powerlaw._install_powerlaw_theta(), called at that module's own import
                 # time) in place of the package's Gamma default, for every
                 # CommunityNetworkBackend built afterwards in this process. Also the source
                 # of the shared age-mixing kernel used below.

married_matrix = [        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [10, 0, 0, 0.08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [15, 0, 0, 0.08, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [20, 0, 0, 0, 0, 0.6, 2, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],        [25, 0, 0, 0, 0, 0.6, 1, 2, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0],        [30, 0, 0, 0, 0, 0.5, 0.5, 2, 1, 0.5, 0.1, 0, 0, 0, 0, 0, 0],        [35, 0, 0, 0, 0, 1, 0.5, 1, 2, 1, 0.5, 0.2, 0, 0, 0, 0, 0],        [40, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.3, 0.1, 0, 0, 0, 0],        [45, 0, 0, 0, 0, 0.1, 1, 2, 2, 2, 1, 0.5, 0.2, 0.08, 0, 0, 0],        [50, 0, 0, 0, 0, 0, 0.1, 1, 2, 3, 2, 2, 0.5, 0.2, 0.05, 0, 0],        [55, 0, 0, 0, 0, 0, 0, 0.1, 1, 2, 3, 3, 2, 1, 0.3, 0.1, 0.1],        [60, 0, 0, 0, 0, 0, 0, 0.1, 0.5, 1, 2, 3, 3, 2, 0.5, 0.3, 0.1],        [65, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 2, 2, 3, 3, 2, 1, 0.2],        [70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 2, 3, 3, 2, 1],        [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3],    ]
married_matrix = np.array(married_matrix)
casual_matrix = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],        [5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],        [10,0,0,0,0.2,0.1,0.05,0,0,0,0,0,0,0,0,0,0],        [15,0,0,1,2,3,2,1,0.5,0,0,0,0,0,0,0,0],        [20,0,0,0.15,2,3,2,2,1,0.15,0,0,0,0,0,0,0],        [25,0,0,0.15,0.25,1,2,2,1,1,0,0,0,0,0,0,0],        [30,0,0,0,0,0.5,0.5,2,1,0.15,0,0,0,0,0,0,0],        [35,0,0,0,0,1,0.5,1,2,1,0.5,0,0,0,0,0,0],        [40,0,0,0,0,1,1,1,1,1,0.5,0.25,0,0,0,0,0],        [45,0,0,0,0,0.15,1,2,2,2,1,0.5,0.2,0.1,0,0,0],        [50,0,0,0,0,0,0.15,1,2,3,2,2,0.5,0.2,0.05,0,0],        [55,0,0,0,0,0,0,0.15,1,2,3,3,2,1,0.25,0.1,0.1],        [60,0,0,0,0,0,0,0.15,0.15,1,2,3,3,2,0.5,0.25,0.1],        [65,0,0,0,0,0,0,0,0,0,1,1,2,2,1,0.5,0],        [70,0,0,0,0,0,0,0,0,0,0,0,0,0.8,1,0.7,0.5],        [75,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.25]    ]
casual_matrix = np.array(casual_matrix)

start = 1980
end = 2055

# Calibrated community-network (power-law) partnership parameters -- from
# calibrate_community_powerlaw.py (200,000 agents, 6-iteration proportional calibration
# against the same Natsal-derived pooled targets used by calibrate_default_poisson.py --
# see that script's TARGETS dict and final printed knobs).
#
# THIRD calibration pass, run after the 2026-07-30 fixes to community_network.py (guaranteed
# pairing, targeted singleness gate, gate connectivity compensation, mortality-aware
# calibration) and powerlaw.py (shared blended age-mixing kernel). Realised vs target at the
# final iteration:
#
#     mean_degree_annual  1.447 vs 1.400   -- hit
#     p_long              0.605 vs 0.618   -- hit (was 0.22 before this pass)
#     p_single            0.271 vs 0.201   -- close (was 0.76 before this pass)
#     p_short             0.190 vs 0.278   -- low
#     mean_degree_5yr     1.943 vs 2.500   -- low
#     cv_degree_annual    0.988 vs 1.831   -- low
#
# SINCE that pass, D_mean_short/D_mean_long have been replaced with the Natsal-3 fitted
# durations already used by basePars_community.py, so both community-network variants now
# share the same empirical durations rather than letting the calibration invent them. The
# remaining knobs are still the calibrated ones, which were fitted against the (much longer)
# calibrated durations -- so the realised-vs-target table above is now only indicative, and
# a re-calibration with the durations held fixed at these Natsal values would be the clean
# way to restore it.
#
# Remaining caveats:
#   * p_single settles ABOVE p_single_annual (0.271 vs the 0.20 input) because the gate is
#     re-drawn only once a year: partnerships dissolve between annual boundaries, so extra
#     people drift into being unpartnered mid-year. Lower p_single_annual if you want the
#     realised instantaneous figure to sit on 0.20 exactly.
#   * cv_degree_annual is still unreachable with gamma_shape pinned at GAMMA_SHAPE_FLOOR
#     (2.05); the Pareto tail cannot get heavier without crossing the finite-variance bound.
#   * D_mean_short/D_mean_long are RAW dissolution hazards, applied BEFORE mortality: the
#     backend separately destroys any partnership whose partner dies (per-edge hazard
#     mu ~ 0.0016/month, see CommunityNetworkBackend._estimate_edge_mortality_hazard). So the
#     REALISED mean duration is 1/(1/D_mean + mu) -- about 135 months rather than 172.9 for
#     long ties (short ties are barely affected, 12.1 vs 12.3, since their own hazard
#     dominates). If the Natsal fit already included partnerships ended by bereavement this
#     is a mild double-count, and D_mean_long ~239 would be needed to make the realised mean
#     come out at 172.9.
community_pars = dict(
    mean_partners_per_year=0.6667,  # calibrated via calibrate_community_powerlaw.py
    gamma_shape=2.0500,              # calibrated -- Pareto alpha; pinned at GAMMA_SHAPE_FLOOR, cv_degree_annual target not reached
    D_mean_short=12.3,            # from Natsal 3 (months)
    D_mean_long=172.9,           #  from Natsal 3 (months) -- pre-mortality hazard, see caveat above
    frac_long=0.8662,                # calibrated
    n_communities=1,
    # Annual singleness control -- a fixed INPUT, deliberately not a calibration knob (the
    # user's call). Realised p_single settles somewhat below this, since gated people who
    # already hold a partnership keep it; see community_network.py's _force_pair_ungated().
    p_single_annual=0.20,
    # Shared blended age-mixing kernel, identical to the one the calibration harness uses --
    # see powerlaw.py section 2b for why the default sim['mixing']['s'] path is not used
    # (it left 15-25 year olds structurally single, and differed between calibration and
    # validation).
    age_mixing=powerlaw.AGE_MIXING,
    age_band_edges=powerlaw.AGE_BAND_EDGES,
)

base_pars = dict(n_agents= 200_000,#200_000,
                start=start, end=end, dt=0.25,
                location='united kingdom',
                verbose=-1,
                debut=dict(f=dict(dist='normal', par1=16.0, par2=3.1), m=dict(dist='normal', par1=16.0, par2=4.1)),
                mixing = {'s':married_matrix,
                          'l':casual_matrix},
                condoms = dict(s=0.17, l=0.50), #condom usage in (s)hort and (l)ong relationships
                network = 'community',
                community_pars = community_pars,
                genotypes     = ['hpv16', 'hpv18', 'hi5', 'ohr'],

                init_hpv_prev = {
                    'age_brackets'  : np.array([  16,   24,   34,   44,  54,   64, 150]),
                    'm'             : np.array([ 0.0, 0.25, 0.14,   0.08,  0.06,   0.06, 0.03]),
                    'f'             : np.array([ 0.0,0.24, 0.32,   0.35,  0.35,   0.35, 0.35])
                },

                init_hpv_dist = {
                    'hpv16': 2.3,
                    'hpv18': 0.9,
                    'hi5':  2.2, #HPV 33 is not listed as one of the top 10 most prevalent in general population in (), so we can assume its prevalence is at most 0.004 - so not adding this to the sum
                    'ohr': 2.1,
                }, #(note, this measure will be rescaled to a prob distribution by hpvsim.utils.choose_w)

                #interventions = #NHS_2025_lambdamu.get_interventions(l=1, m=1)
                #NHS_Vacc.vaccinations,

                burnin = 20,
                #added calibration results- these particular ones are Fabian ones
                beta = 0.3304907040374987,
                f_cross_layer = 0.04400514,
                m_cross_layer = 0.4996342079150136,
                analyzers=[hpv.network_history()]
                )
#initialise genotype_pars as a concept
base_pars['genotype_pars'] = sc.objdict()
#grab the genotypes dict from hpv source code
for g in base_pars['genotypes']:
    base_pars['genotype_pars'][g] = get_genotype_pars(genotype=g)

#now can put values directly in
base_pars['genotype_pars']['hi5']['cin_fn']['k'] = 0.000664989
base_pars['genotype_pars']['hi5']['dur_cin']['par1'] = 7.532364881439721
base_pars['genotype_pars']['hi5']['rel_beta'] = 0.064268569
base_pars['genotype_pars']['hpv16']['cin_fn']['k'] = 3.5355730436187815e-05
base_pars['genotype_pars']['hpv16']['dur_cin']['par1'] = 7.280958479418904
base_pars['genotype_pars']['hpv18']['cin_fn']['k'] = 0.0282588677537481
base_pars['genotype_pars']['hpv18']['dur_cin']['par1'] = 2.4115489232500136
base_pars['genotype_pars']['ohr']['cin_fn']['k'] = 0.074766782
base_pars['genotype_pars']['ohr']['dur_cin']['par1'] = 10.941333033716116
base_pars['genotype_pars']['ohr']['rel_beta'] = 0.9408537321469647

#because of the order, you have to import a new variable instead of base_pars (?)
base_pars_geno = base_pars

if __name__ == "__main__":
    import hpvsim as hpv

    base_pars['verbose'] = 0

    sim = hpv.Sim(base_pars_geno)
    print(sim.pars['genotype_pars']['hi5'])
    sim.run()
    sim.plot()

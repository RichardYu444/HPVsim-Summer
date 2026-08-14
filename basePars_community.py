"""
Stores the base parameters for a HPVsim simulation using the current NHS strategy,
but wired to the age+community bipartite network (pars['network'] == 'community',
see hpvsim_working/community_network.py) instead of HPVsim's built-in default network
used in basePars.py.

"""
import numpy as np
import NHS_2025_lambdamu, NHS_Vacc
from hpvsim_working.parameters import get_genotype_pars
import sciris as sc
import hpvsim_working as hpv

married_matrix = [        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [10, 0, 0, 0.08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [15, 0, 0, 0.08, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [20, 0, 0, 0, 0, 0.6, 2, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],        [25, 0, 0, 0, 0, 0.6, 1, 2, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0],        [30, 0, 0, 0, 0, 0.5, 0.5, 2, 1, 0.5, 0.1, 0, 0, 0, 0, 0, 0],        [35, 0, 0, 0, 0, 1, 0.5, 1, 2, 1, 0.5, 0.2, 0, 0, 0, 0, 0],        [40, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.3, 0.1, 0, 0, 0, 0],        [45, 0, 0, 0, 0, 0.1, 1, 2, 2, 2, 1, 0.5, 0.2, 0.08, 0, 0, 0],        [50, 0, 0, 0, 0, 0, 0.1, 1, 2, 3, 2, 2, 0.5, 0.2, 0.05, 0, 0],        [55, 0, 0, 0, 0, 0, 0, 0.1, 1, 2, 3, 3, 2, 1, 0.3, 0.1, 0.1],        [60, 0, 0, 0, 0, 0, 0, 0.1, 0.5, 1, 2, 3, 3, 2, 0.5, 0.3, 0.1],        [65, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 2, 2, 3, 3, 2, 1, 0.2],        [70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 2, 3, 3, 2, 1],        [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3],    ]
married_matrix = np.array(married_matrix)
casual_matrix = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],        [5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],        [10,0,0,0,0.2,0.1,0.05,0,0,0,0,0,0,0,0,0,0],        [15,0,0,1,2,3,2,1,0.5,0,0,0,0,0,0,0,0],        [20,0,0,0.15,2,3,2,2,1,0.15,0,0,0,0,0,0,0],        [25,0,0,0.15,0.25,1,2,2,1,1,0,0,0,0,0,0,0],        [30,0,0,0,0,0.5,0.5,2,1,0.15,0,0,0,0,0,0,0],        [35,0,0,0,0,1,0.5,1,2,1,0.5,0,0,0,0,0,0],        [40,0,0,0,0,1,1,1,1,1,0.5,0.25,0,0,0,0,0],        [45,0,0,0,0,0.15,1,2,2,2,1,0.5,0.2,0.1,0,0,0],        [50,0,0,0,0,0,0.15,1,2,3,2,2,0.5,0.2,0.05,0,0],        [55,0,0,0,0,0,0,0.15,1,2,3,3,2,1,0.25,0.1,0.1],        [60,0,0,0,0,0,0,0.15,0.15,1,2,3,3,2,0.5,0.25,0.1],        [65,0,0,0,0,0,0,0,0,0,1,1,2,2,1,0.5,0],        [70,0,0,0,0,0,0,0,0,0,0,0,0,0.8,1,0.7,0.5],        [75,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.25]    ]
casual_matrix = np.array(casual_matrix)

start = 1980
end = 2055

# ---------------------------------------------------------------------------
# Ethnicity as the network's "communities"
# ---------------------------------------------------------------------------
# Communities are used here to stand in for ethnic groups, in this fixed order:
ETHNICITIES = ['White', 'Asian', 'Black', 'Chinese']   # 'Asian' = Asian excluding Chinese

# Population shares (England & Wales). These do NOT sum to 1 -- the ~5% Mixed/Other
# population has no category here, so the four shares are renormalised over themselves.
# That implicitly redistributes Mixed/Other proportionally, which is a real assumption:
# Mixed-ethnicity individuals partner far less assortatively than any of these four.
eth_pop_shares = np.array([0.817, 0.086, 0.040, 0.007])
community_probs = eth_pop_shares / eth_pop_shares.sum()   # [0.8600, 0.0905, 0.0421, 0.0074]

# Observed symmetric joint distribution of partnership ends by ethnicity pair (Natsal,
# all relationship types). Rows/cols in ETHNICITIES order.
eth_joint = np.array([
    [0.8909, 0.0100, 0.0098, 0.0034],
    [0.0100, 0.0326, 0.0007, 0.0009],
    [0.0098, 0.0007, 0.0240, 0.0002],
    [0.0034, 0.0009, 0.0002, 0.0028],
])
eth_joint = eth_joint / eth_joint.sum()


def _rescale_to_margins(mat, target, iters=500):
    '''
    Biproportional (IPF) rescaling of a symmetric joint so both margins equal `target`,
    leaving every odds ratio -- i.e. the whole assortativity structure -- unchanged.

    Needed because the survey's marginal ethnic composition of partnership *ends*
    (White 91.4%, Asian 4.4%, Black 3.5%, Chinese 0.7%) is not the population
    composition (86.0 / 9.1 / 4.2 / 0.7). Taken at face value that mismatch implies
    Asian mean degree is ~0.49x White, but it is confounded by age structure (the
    minority populations are much younger than the population as a whole) and by the
    dropped Mixed/Other group, so we do not import it -- see the note below.
    '''
    x = np.array(mat, dtype=float)
    for _ in range(iters):
        x *= (target / x.sum(axis=1))[:, None]
        x *= (target / x.sum(axis=0))[None, :]
    return x / x.sum()


# CommunityNetworkBackend's community_mixing is a matrix of *relative affinities*, not
# probabilities: block-pair edge counts go as C[g,h] * S_g * S_h, where S_g is the
# propensity mass (~ the size) of community g, so the group sizes are already in the
# formula (see age_community_bipartite_network_model._sample_edges). The conditional
# P(f|m) / P(m|f) tables must NOT be used here -- passing one in applies the minority
# marginal twice and collapses within-group partnering by ~20x. The correct input is
# joint / outer(shares, shares).
#
# Because we IPF the joint onto the population shares first, every propensity-weighted
# row sum of C is exactly 1, so ethnicity changes *who* partners with whom without
# changing mean degree in any group. To instead reproduce the raw survey joint --
# assortativity AND its implied degree gradient (White 1.06x, Asian 0.49x, Black 0.82x,
# Chinese 0.99x) -- drop the _rescale_to_margins() call and use eth_joint directly.
community_mixing = (_rescale_to_margins(eth_joint, community_probs)
                    / np.outer(community_probs, community_probs))
# C is symmetric, which also makes it immune to the row=female / col=male indexing of
# W (community_network.py binds U=female, V=male, contrary to the network model's own
# "U=man" docstrings).

# Parameters for the age+community bipartite network (see hpvsim_working/parameters.py's
# pars['community_pars'] defaults and community_testing.py's tuned COMMUNITY_PARS, which these
# mirror). age_mixing/age_band_edges are deliberately left unset -- CommunityNetworkBackend
# derives them from base_pars['mixing']['s'] (married_matrix) below instead.
community_pars = dict(
    mean_partners_per_year=1.5,
    gamma_shape=3,
    # Partnership durations are exponentially distributed (constant monthly
    # hazard q = 1/D_mean), fit to Natsal: short relations lambda=0.0811/month
    # (mean 12.3 months), long relations lambda=0.0058/month (mean 172.9 months).
    D_mean_short=12.3,   # months
    D_mean_long=172.9,   # months
    frac_long=0.618,      # target standing fraction of long partnerships
    # Communities = ethnic groups (ETHNICITIES above). community_off_diag is deliberately
    # not set: it only supplies a default when community_mixing is absent.
    n_communities=len(ETHNICITIES),
    community_probs=community_probs,
    community_mixing=community_mixing,
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

                interventions = NHS_2025_lambdamu.get_interventions(l=1, m=1) + 
                NHS_Vacc.vaccinations,

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

"""
Stores the base parameters for a HPVsim simulation using the current NHS strategy.
Can override aspects of this dictionary, particularly interventions, if needed.
"""
import numpy as np
import NHS_2025_lambdamu, NHS_Vacc
from hpvsim.parameters import get_genotype_pars
import sciris as sc

married_matrix = [        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [10, 0, 0, 0.08, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [15, 0, 0, 0.08, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],        [20, 0, 0, 0, 0, 0.6, 2, 0.2, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],        [25, 0, 0, 0, 0, 0.6, 1, 2, 0.4, 0.1, 0, 0, 0, 0, 0, 0, 0],        [30, 0, 0, 0, 0, 0.5, 0.5, 2, 1, 0.5, 0.1, 0, 0, 0, 0, 0, 0],        [35, 0, 0, 0, 0, 1, 0.5, 1, 2, 1, 0.5, 0.2, 0, 0, 0, 0, 0],        [40, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0.5, 0.3, 0.1, 0, 0, 0, 0],        [45, 0, 0, 0, 0, 0.1, 1, 2, 2, 2, 1, 0.5, 0.2, 0.08, 0, 0, 0],        [50, 0, 0, 0, 0, 0, 0.1, 1, 2, 3, 2, 2, 0.5, 0.2, 0.05, 0, 0],        [55, 0, 0, 0, 0, 0, 0, 0.1, 1, 2, 3, 3, 2, 1, 0.3, 0.1, 0.1],        [60, 0, 0, 0, 0, 0, 0, 0.1, 0.5, 1, 2, 3, 3, 2, 0.5, 0.3, 0.1],        [65, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 2, 2, 3, 3, 2, 1, 0.2],        [70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5, 1, 2, 3, 3, 2, 1],        [75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3],    ]
married_matrix = np.array(married_matrix)
casual_matrix = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],        [5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],        [10,0,0,0,0.2,0.1,0.05,0,0,0,0,0,0,0,0,0,0],        [15,0,0,1,2,3,2,1,0.5,0,0,0,0,0,0,0,0],        [20,0,0,0.15,2,3,2,2,1,0.15,0,0,0,0,0,0,0],        [25,0,0,0.15,0.25,1,2,2,1,1,0,0,0,0,0,0,0],        [30,0,0,0,0,0.5,0.5,2,1,0.15,0,0,0,0,0,0,0],        [35,0,0,0,0,1,0.5,1,2,1,0.5,0,0,0,0,0,0],        [40,0,0,0,0,1,1,1,1,1,0.5,0.25,0,0,0,0,0],        [45,0,0,0,0,0.15,1,2,2,2,1,0.5,0.2,0.1,0,0,0],        [50,0,0,0,0,0,0.15,1,2,3,2,2,0.5,0.2,0.05,0,0],        [55,0,0,0,0,0,0,0.15,1,2,3,3,2,1,0.25,0.1,0.1],        [60,0,0,0,0,0,0,0.15,0.15,1,2,3,3,2,0.5,0.25,0.1],        [65,0,0,0,0,0,0,0,0,0,1,1,2,2,1,0.5,0],        [70,0,0,0,0,0,0,0,0,0,0,0,0,0.8,1,0.7,0.5],        [75,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.25]    ]
casual_matrix = np.array(casual_matrix)

start = 1980
end = 2055

base_pars = dict(n_agents= 20_000,#200_000, 
                start=start, end=end, dt=0.25, 
                location='united kingdom', 
                verbose=-1,
                debut=dict(f=dict(dist='normal', par1=16.0, par2=3.1), m=dict(dist='normal', par1=16.0, par2=4.1)),
                mixing = {'m':married_matrix,
                          'c':casual_matrix},
                condoms = dict(m=0.17, c=0.50), #condom usage in (m)arried and (c)asual relationships
                #network = 'random',
                genotypes     = ['hpv16'],#, 'hpv18', 'hi5', 'ohr'],

                init_hpv_prev = {
                    'age_brackets'  : np.array([  16,   24,   34,   44,  54,   64, 150]),
                    'm'             : np.array([0, 0, 0, 0, 0, 0, 0]),#np.array([ 0.0, 0.25, 0.14,   0.08,  0.06,   0.06, 0.03]),
                    'f'             : np.array([0, 0, 0, 0, 0, 0, 0]),#np.array([ 0.0,0.24, 0.32,   0.35,  0.35,   0.35, 0.35])
                },

                init_hpv_dist = {
                    'hpv16': 2.3#,
                #    'hpv18': 0.9,
                 #   'hi5':  2.2, #HPV 33 is not listed as one of the top 10 most prevalent in general population in (), so we can assume its prevalence is at most 0.004 - so not adding this to the sum
                  #  'ohr': 2.1,
                }, #(note, this measure will be rescaled to a prob distribution by hpvsim.utils.choose_w)

                #interventions = #NHS_2025_lambdamu.get_interventions(l=1, m=1)
                #NHS_Vacc.vaccinations,

                burnin = 20,
                #added calibration results- these particular ones are Fabian ones
                beta =0,# 0.3304907040374987,
                f_cross_layer = 0.04400514,
                m_cross_layer = 0.4996342079150136

                )
#initialise genotype_pars as a concept
#base_pars['genotype_pars'] = sc.objdict()
#grab the genotypes dict from hpv source code
#for g in base_pars['genotypes']:
 #   base_pars['genotype_pars'][g] = get_genotype_pars(genotype=g)

#now can put values directly in
#base_pars['genotype_pars']['hi5']['cin_fn']['k'] = 0.000664989
#base_pars['genotype_pars']['hi5']['dur_cin']['par1'] = 7.532364881439721
#base_pars['genotype_pars']['hi5']['rel_beta'] = 0.064268569
#base_pars['genotype_pars']['hpv16']['cin_fn']['k'] = 3.5355730436187815e-05
#base_pars['genotype_pars']['hpv16']['dur_cin']['par1'] = 7.280958479418904
#base_pars['genotype_pars']['hpv18']['cin_fn']['k'] = 0.0282588677537481
#base_pars['genotype_pars']['hpv18']['dur_cin']['par1'] = 2.4115489232500136
#base_pars['genotype_pars']['ohr']['cin_fn']['k'] = 0.074766782
#base_pars['genotype_pars']['ohr']['dur_cin']['par1'] = 10.941333033716116
#base_pars['genotype_pars']['ohr']['rel_beta'] = 0.9408537321469647

#because of the order, you have to import a new variable instead of base_pars (?)
base_pars_geno = base_pars

if __name__ == "__main__":
    import hpvsim as hpv

    base_pars['verbose'] = 0

    sim = hpv.Sim(base_pars_geno)
    print(sim.pars['genotype_pars']['hi5'])
    sim.run()
    sim.plot()
"""
This file contains an model of the NHS England vaccination strategy(ies) since 2008.

File usage:
    -> Import this file to easily integrate vaccination model into a HPVsim simulation
    -> Run this file (as __main__) to run a HPVsim simulation with this screening and debug changes to the algorithm

Notable modelling assumptions:
    -> HIV is not being modelled in this study, justified by low HIV prevalence in the UK, particularly among those with cervixes. Therefore, we do not include branches of the pathway which relate to HIV, assuming all individuals are HIV -ve.

Below are sources to justify modelling decisions.


Compliance:
    -> Covergae of (complete; one dose) vaccination of females by year in UK (2010-2021): (78,83,87,84,85,84,83,82,82, 82,64,59); (86,89,91,90,90,88,86,86,85, 85,60,77)
        ^https://immunizationdata.who.int/global/wiise-detail-page/human-papillomavirus-(hpv)-vaccination-coverage, accessed 24 Oct 2022
    -> Coverage of (complete; one dose) vaccination of males by year in UK (2020-2021): (-, 48); (53,71) 
            ^https://immunizationdata.who.int/global/wiise-detail-page/human-papillomavirus-(hpv)-vaccination-coverage, accessed 24 Oct 2022

            

"""



#--- Imports ---#
import hpvsim_working as hpv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from copy import copy


#NOTE: the imports below are a little unstable, as they change depending on whether we are running things from within the directory InterventionAlgorithms, or outside of it.

import GlobalScreeningParameters as GlobalScreeningParameters #USE THIS iff running code which imports this from outside of Intervention Algorithms

"""
Strategy to be implemented: - NHS policies over time as enumerated by https://doi.org/10.1016/j.lanepe.2024.101157
2008-2011: starting vaccination
    Bivalent Cervarix vaccine to 12-13 year old girls, 3 doses

    + 2008-2010: catch-up campaign
            Targeting girls aged 14-18 years with Bivalent Cervarix vaccine, 3 doses

2012-2013: switch to 4vHPV
    Quadrivalent Gardasil vaccine to 12-13 year old girls, 3 doses

2014-2018: down to 2 doses
    Quadrivalent Gardasil vaccine to 12-13 year old girls, 2 doses

2019-2021: adding boys
    Quadrivalent Gardasil vaccine to 12-13 year old girls+boys, 2 doses

2022-2022: switch to 9vHPV
    Nonavalent Gardasil-9 vaccine to 12-13 year old girls+boys, 2 doses

2023-: down to 1 dose
    Nonavalent Gardasil-9 vaccine to 12-13 year old girls+boys, 1 dose

Modelling several doses:
    -> We have data on (A)#1-dose vaccinatons, (B)#completed vaccine courses
    -> Therefore, model all people who are 'vaccinated' (that is, have at least one dose) as elibile for further courses,
    -> ... where prob of second dose is p(full course|1 dose) and we model p(3rd dose|2nd dose)=1 (iff 3rd dose being offered) (rather than, say, a linear dropoff between probs, all people who go for a second dose also get a 3rd, if applicable)

Modelling girls and boys:
    -> right now, I am averaging girl and boy participation data and doing the vaccination regieme to both at the same time for years where there is sex-neutral vaccination. However, as boys have consistently lower vaccine uptake than girls, I suppose I could improve this!    

Modelling beyond 2021:
    -> Assuming 90% first dose coverage, and 80% second dose coverage in 2022. TODO: get vaccination uptake for 22,23,24 and then get a better assumption for coverage! Of course, will do a notable sensitivity analysis here too!    
"""


eligible_first_dose = lambda sim: sim.people.doses == 0 
eligible_second_dose = lambda sim: sim.people.doses == 1
eligible_third_dose = lambda sim: sim.people.doses == 2


#2008-2011
vx_0811_d1 = hpv.routine_vx(prob=(0.86,0.86,0.86,0.89),
                         start_year=2008,
                         end_year=2011, #TODO: I think the end year is inclusive, but can validate results to ensure it is (also ig must be if it doesnt hate the prob lists)
                         age_range=[12,13], #(inclusive, exclusive)
                         product='bivalent',
                         eligibility=eligible_first_dose,
                        )
vx_0811_d2 = hpv.routine_vx(prob=(0.78/0.86 ,0.78/0.86 ,0.78/0.86 ,0.83/0.89),
                         start_year=2008,
                         end_year=2011,
                         age_range=[12,13], #(inclusive, exclusive)
                         product='bivalent2', #2nd dose; see interventions.default_vx() for assumed imm_boost of 1.2, and interventions CNTRL+F for 'doses' and immunity.py CNTRL+F for 'imm_boost' for how this works
                         eligibility=eligible_second_dose, 
                        )
vx_0811_d3 = hpv.routine_vx(prob=1,
                         start_year=2008,
                         end_year=2011,
                         age_range=[12,13], #(inclusive, exclusive)
                         product='bivalent3',  #3rd dose; see interventions.default_vx() for assumed imm_boost of 1.1, and interventions CNTRL+F for 'doses' and immunity.py CNTRL+F for 'imm_boost' for how this works
                         eligibility=eligible_third_dose,
                        )

#2012-2018 (the 3rd dose drops off at 2013, rest continues on till 2018)
vx_1218_d1 = hpv.routine_vx(prob=(0.91,0.90,0.90,0.88,0.86,0.86,0.85),
                         start_year=2012,
                         end_year=2018, 
                         age_range=[12,13], #(inclusive, exclusive)
                         product='quadrivalent', 
                         eligibility=eligible_first_dose,
                        )
vx_1218_d2 = hpv.routine_vx(prob=(0.87/0.91,0.84/0.9,0.85/0.9,0.84/0.88,0.83/0.86,0.82/0.86,0.82/0.85),
                         start_year=2012,
                         end_year=2018,
                         age_range=[12,13], #(inclusive, exclusive)
                         product='quadrivalent2', #2nd dose; see interventions.default_vx() for assumed imm_boost of 1.2, and interventions CNTRL+F for 'doses' and immunity.py CNTRL+F for 'imm_boost' for how this works
                         eligibility=eligible_second_dose, 
                        )
vx_1213_d3 = hpv.routine_vx(prob=1,
                         start_year=2012,
                         end_year=2013,
                         age_range=[12,13], #(inclusive, exclusive)
                         product='quadrivalent3',  #3rd dose; see interventions.default_vx() for assumed imm_boost of 1.1, and interventions CNTRL+F for 'doses' and immunity.py CNTRL+F for 'imm_boost' for how this works
                         eligibility=eligible_third_dose, 
                        )

#2019-2021
vx_1921_d1 = hpv.routine_vx(prob=(0.85,(0.60+0.53)/2,(0.77+0.71)/2),
                         start_year=2019,
                         end_year=2021, 
                         sex=['f','m'],
                         age_range=[12,13], #(inclusive, exclusive)
                         product='quadrivalent', 
                         eligibility=eligible_first_dose,
                        )
vx_1921_d2 = hpv.routine_vx(prob=(0.82/0.85,0.64*2/(0.60+0.53),(0.59+0.48)/(0.77+0.71)), #TODO: something wrong this with second prob, perhaps combinging boys and gitrls doesnt work here
                         start_year=2019,
                         end_year=2021,
                         sex=['f','m'],
                         age_range=[12,13], #(inclusive, exclusive)
                         product='quadrivalent2', #2nd dose; see interventions.default_vx() for assumed imm_boost of 1.2, and interventions CNTRL+F for 'doses' and immunity.py CNTRL+F for 'imm_boost' for how this works
                         eligibility=eligible_second_dose,
                        )

#2022 - XX (the 2nd dose drops off at end of 2022, rest continues on till end of simulation)
vx_22XX_d1 = hpv.routine_vx(prob=GlobalScreeningParameters.projected_teen_vaccination_uptake,
                         start_year=2022,
                         
                         sex=['f','m'],
                         age_range=[12,13], #(inclusive, exclusive)
                         product='nonavalent',
                         eligibility=eligible_first_dose,
                        )
vx_2222_d2 = hpv.routine_vx(prob=0.5/0.6, #0.7/0.8, #0.5/0.6
                         start_year=2022,
                         end_year=2022,
                         sex=['f','m'],
                         age_range=[12,13], #(inclusive, exclusive)
                         product='nonavalent2', #2nd dose; see interventions.default_vx() for assumed imm_boost of 1.2, and interventions CNTRL+F for 'doses' and immunity.py CNTRL+F for 'imm_boost' for how this works
                         eligibility=eligible_second_dose,
                        )


vaccinations = [
    vx_0811_d1, #2vHPV

    vx_1921_d1, #4vHPV
    vx_22XX_d1, #9vHPV

    #other routine vaccination interventions to be added, which repeat the use of already-mentioned vaccines
                vx_0811_d2, vx_0811_d3,
    vx_1218_d1, vx_1218_d2, vx_1213_d3,
                 vx_1921_d2,
                vx_2222_d2,
]

if __name__=="__main__":
    #HPVsim simulation parameters
    pars = dict(
            n_agents      = 20e3,       
            start=1970, end=2100,
            #verbose       = 0,   
            #rand_seed     = 1, #comment out random seed to get different results when runnning the same sim many times in a multisim and averaging over results
            genotypes     = [16,18, 'hi5'],
            burnin=30,
            location='united kingdom',
            )


    #Define simulations
    sim_basic = hpv.Sim(pars, label='Sim, no interventions')

    sim_fullvacc =  hpv.Sim(pars, interventions = vaccinations, label='Full vaccination programme')
    
    sim_firstdosesonly =  hpv.Sim(pars, interventions = [vx_0811_d1, vx_1921_d1, vx_22XX_d1, vx_1218_d1], label='First doses only 2008-2022')

    #Run and plot sims
    reduced_sims = []
    for sim in [sim_basic, sim_fullvacc, sim_firstdosesonly]:
        msim = hpv.MultiSim(sim, n_runs=10)  
        msim.run()         
        reduced_sims.append(msim.reduce(use_mean=True, output=True))  


    msim = hpv.MultiSim(reduced_sims)
    msim.plot()

    #NOTE: we can only really see a difference between running with the first dose only vs all doses when we decrease 
    #       the efficacy of the vaccines (e.g. it is pretty clear with 10 runs for each sim type and putting all vaccines' 16/18 efficacy at 70%, and nonavalent effiaccy against hi5 at 70% too.)
    #      At HPVsim's default (high) vaccine efficacy, though, the difference between one or many doses isnt that big - as is correct.
    # ^this does satisfy me that this code is correct though!:)

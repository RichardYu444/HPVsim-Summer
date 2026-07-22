"""
This file contains an model of the current NHS England Cervical Screening Algorithm, with adjustments to allow for changing regular screening callback intervals to differ between vaccinated and unvaccinated individuals.
    ^https://www.gov.uk/government/publications/cervical-screening-care-pathway/cervical-screening-care-pathway (version as of 1st July 2025)

File usage:
    -> Import this file to easily integrate screening algorithm model into a HPVsim simulation
    -> Run this file (as __main__) to run a HPVsim simulation with this screening and debug changes to the algorithm


Compliance:
    -> Attendance at early recall and colposcopy referral were 80% and 95%, respectively : "Rebolj2019" (its name in the bibtex file)
            ^note that this comes from a large pilot study (just over half a million women around 2013-14), the figures may be a bit different in reality/since then
            TODO: need some senitivity analysis for lower values

            #NOTE: it isnt feasible to redo a calibration for every sensitivty analysis, so instead I will do a calibration with the best guesses for all the parameter values, and then do sensitivyt analysis on these parameters following a lil jittering just to get a bit of robustness on those parameters at least

"""


###--- IMPORTS ---###

import hpvsim_working as hpv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


from copy import copy



#NOTE: the imports below are a little unstable, as they change depending on whether we are running things from within the directory InterventionAlgorithms, or outside of it.

from generalCancerTreatment import  GeneralCancerTreatment, treat_num_cancer, treat_num_canceragnostic  #USE THIS iff running code which imports this from outside of Intervention Algorithms
import GlobalScreeningParameters as GlobalScreeningParameters  #USE THIS iff running code which imports this from outside of Intervention Algorithms

#from generalCancerTreatment import  GeneralCancerTreatment, treat_num_cancer, treat_num_canceragnostic #USE THIS iff running this code itself, or other code which is inside and imports this code
#import GlobalScreeningParameters as GlobalScreeningParameters #USE THIS iff running this code itself, or other code which is inside and imports this code








def init_intervention_trackers(sim):
    """
    For t=0, sets up tracking apparatus within the sim for other interventions in this file to run.
    WARNING: Must be added to a sim before all other interventions defined in this file (as sim runs interventions in the order of its intervention list, and therefore we need this to run first, to define the interventions themselves)
    """
    if sim.t == 0:
        #Tracks properties of agents as dictionaries {agent_id: tracked property} to handle dynamic populations; agent_id is permanent for any given agent
        sim.people.last_hpv_result = {} #track whether an agent's last hpv test result was positive or not
            #TODO: could I refactor this s.t. I just keep track of a list of agent_id's whose last hpv test results were positive, as we dont need to clog things up with keeping track of who has had negative results
        sim.people.needs_consec_screening_2 = {} #keeps track of who needs a second round of consecutive screening, if contains (pid: t) then agent pid needs screening at earliest time t
        sim.people.needs_consec_screening_3 = {} #as above, but for a third round of consecutive screening
init_intervention_trackers.label="init_intervention_trackers"


#Custom interventions to update attributes that keep track of the other interventions' results, persistently across years
def update_last_hpv_result(sim):
    #For everyone who has undergone a screening of some sort, update their 'last_hpv_result' property to reflect the outcome of the screening
    hpv_test_names = ["routine_screening_under50","routine_screening_50andover", "second_consecutive_screening", "third_consecutive_screening"]
        #the above list is in order of administration of the tests within a given timepoint; noone should undergo several of the same tests at any point, but just in case we want to use the most recent result!
    for name in hpv_test_names:
        hpv_test = sim.get_intervention(name)
        for positive in hpv_test.outcomes.get('positive'):
            sim.people.last_hpv_result[positive] = +1
        for negative in hpv_test.outcomes.get('negative'):
            sim.people.last_hpv_result[negative] = -1
update_last_hpv_result.label="update_last_hpv_result" #by adding an intervention as a function, when searching 'get_intervention', interventions without a label attribute cause an error

def update_needs_consec_screening_2(sim, name="first_cytology"):
    #If a woman has a -ve result from her first cytology, she is eligible for a second consecutive screening in 12 months
    for i in sim.get_intervention(name).outcomes['normal']: #iterate over all negative outcomes from first cytology IN THiS TIMESTEP (as each interventions 'outcomes' property is wiped at the start of each timestep)
        sim.people.needs_consec_screening_2[i] = sim.t+1/sim["dt"]
        ##UNCOMMENT THE BELOW TO FIND THE PID OF AN AGENT WHO REQUIRES A SECOND SCREENING, FOR DEBUGGING
        #print(f"current time = {sim.t}")
        #print(f"{i} needs a second screening, which will occur at time {sim.people.needs_consec_screening_2[i]}")  
        #quit(0)  
    
    #All women who have undergone their second conseucutive screening have this updated in their record (no more need for 2nd consecutive screening)
    for i in sim.get_intervention("second_consecutive_screening").outcomes['positive']:
        sim.people.needs_consec_screening_2.pop(i)
    for i in sim.get_intervention("second_consecutive_screening").outcomes['negative']:
        sim.people.needs_consec_screening_2.pop(i)
    for i in sim.get_intervention("second_consecutive_screening").outcomes['inadequate']:
        sim.people.needs_consec_screening_2.pop(i)
update_needs_consec_screening_2.label="update_needs_consec_screening_2"

def update_needs_consec_screening_3(sim, name="second_cytology"):
    #If a woman has a -ve result from her second cytology, she is eligible for a third consecutive screening in 12 months
    for i in sim.get_intervention(name).outcomes['normal']: #iterate over all negative outcomes from second cytology IN THiS TIMESTEP (as each interventions 'outcomes' property is wiped at the start of each timestep)
        sim.people.needs_consec_screening_3[i] = sim.t+1/sim["dt"]
        
    #All women who have undergone their third conseucutive screening have this updated in their record (no more need for 3rd consecutive screening)
    for i in sim.get_intervention("third_consecutive_screening").outcomes['positive']:
        sim.people.needs_consec_screening_3.pop(i)
    for i in sim.get_intervention("third_consecutive_screening").outcomes['negative']:
        sim.people.needs_consec_screening_3.pop(i)
    for i in sim.get_intervention("third_consecutive_screening").outcomes['inadequate']:
        sim.people.needs_consec_screening_3.pop(i)
update_needs_consec_screening_3.label="update_needs_consec_screening_3"



#NOTE: routine screening is split into two eligibility functions and two interventions, to target under 50s and 50-and-overs seperately, due to their differently-modelled probabilities of screening uptake (this models the split easily as two instances of the same screening intervention, which then feed into the same followup process) 

# Setting up routine screening, according to last HPV screening outcome (if any), time since last screening, and age
def routine_screen_eligible_under50(sim, l, m, switch_year = GlobalScreeningParameters.switch_year, end_screening_at_switch_year = False):
    #A woman under 50 is eligible for routine screening {3 IF last hpv test +ve else 5} years after their most recent screening
    for_screening = np.isnan(sim.people.date_screened) & (sim.people.age>=24) & (sim.people.age<50) #eligible for screening if they have not yet been screened

    #If we are simply ending screening at the year at which we switch, then mark everyone as screening-ineligible
    if end_screening_at_switch_year and sim.t>(switch_year-sim['start'])/sim['dt']:
        for i in range(sim.n):
            for_screening[i] = False
        return for_screening


    #add further eligibility for those who have been screened before, but are due a followup
    for i in range(sim.n): #sim.n==len(sim.people)
        if i in sim.people.last_hpv_result.keys(): # check if the agent has been screened before
            last_hpv_result = sim.people.last_hpv_result[i]
            age = sim.people.age[i]
            date_screened = sim.people.date_screened[i]
            vaccinated = sim.people.vaccinated[i]


            if sim.t>(switch_year-sim['start'])/sim['dt']:
                screen_interval_scale_factor = l if (not vaccinated) else m
            else:
                screen_interval_scale_factor = 1

            if age<65: #no regular screening for over-65s, incluidng followups (a moot condition)
                screen_interval = 3 if last_hpv_result==1 else 5*screen_interval_scale_factor

                if sim.t>=date_screened+screen_interval/sim['dt']:
                    for_screening[i] = True
            
    return for_screening

def routine_screen_eligible_50andover(sim, l, m, switch_year = GlobalScreeningParameters.switch_year, end_screening_at_switch_year = False):
    #A woman 50 or over is eligible for a routine screening 5 years after their most recent screening
    for_screening = np.isnan(sim.people.date_screened) & (sim.people.age>=50) & (sim.people.age<=65) #eligible for screening if they have not yet been screened

    #If we are simply ending screening at the year at which we switch, then mark everyone as screening-ineligible
    if end_screening_at_switch_year and sim.t>(switch_year-sim['start'])/sim['dt']:
        for i in range(sim.n):
            for_screening[i] = False
        return for_screening


    #add further eligibility for those who have been screened before, but are due a followup
    for i in range(sim.n): #sim.n==len(sim.people)
        if i in sim.people.last_hpv_result.keys(): # check if the agent has been screened before
            age = sim.people.age[i]
            date_screened = sim.people.date_screened[i]
            vaccinated = sim.people.vaccinated[i]


            if sim.t>(switch_year-sim['start'])/sim['dt']:
                screen_interval_scale_factor = l if (not vaccinated) else m
            else:
                screen_interval_scale_factor = 1

            if age<65: #no regular screening for over-65s, including followups
                screen_interval = 5*screen_interval_scale_factor

                if sim.t>=date_screened+screen_interval/sim['dt']:
                    for_screening[i] = True
            
    return for_screening




def get_routine_screening_interventions(l, m, end_screening_at_switch_year = False):
    routine_screening_under50  = hpv.routine_screening(eligibility=lambda sim: routine_screen_eligible_under50(sim, l, m, end_screening_at_switch_year = end_screening_at_switch_year), 
                                    start_year=GlobalScreeningParameters.screening_start_year, 
                                    age_range=[5,150],#overriding the default age range of [30,50] with a very large one - so as not to interfere with the age range I define in my eligibility checking
                                    prob=GlobalScreeningParameters.primary_screen_prob_under50, 
                                    product='hpv',        #Screening: DNA HPV Testing
                                    label='routine_screening_under50') 
    
    routine_screening_50andover  = hpv.routine_screening(eligibility=lambda sim: routine_screen_eligible_50andover(sim, l, m, end_screening_at_switch_year = end_screening_at_switch_year), 
                                    start_year=GlobalScreeningParameters.screening_start_year, 
                                    age_range=[5,150],#overriding the default age range of [30,50] with a very large one - so as not to interfere with the age range I define in my eligibility checking
                                    prob=GlobalScreeningParameters.primary_screen_prob_50andover, 
                                    product='hpv',        #Screening: DNA HPV Testing
                                    label='routine_screening_50andover') 
    return [routine_screening_under50, routine_screening_50andover]



# First (Consecutive) Cyotology for women with positive routine screening
    #[We will refer to actions as 'consecutive' if they can all be traced back to the same initial routine screening]
to_first_cytology = lambda sim: list(set(
    sim.get_intervention('routine_screening_under50').outcomes['positive'].tolist() +
    sim.get_intervention('routine_screening_50andover').outcomes['positive'].tolist()
))
first_cytology = hpv.routine_triage(eligibility=to_first_cytology,
                                    prob = GlobalScreeningParameters.triage_screen_prob,
                                    age_range=[5,150], #I think routine traige doesnt introduce a constricted age range in by default, but in case it does, overriding it here
                                    product='lbc',
                                    annual_prob=False,
                                    label="first_cytology")

#If first cytology negative, send to a screening in a year (i.e. a second consecutive screening)
def second_screening_eligible(sim):
    for_screening = np.array([False,]*sim.n)
    for i in sim.people.needs_consec_screening_2.keys():
        if sim.people.needs_consec_screening_2[i] is not None and sim.people.needs_consec_screening_2[i] <= sim.t <= sim.people.needs_consec_screening_2[i] + GlobalScreeningParameters.abandon_followup_invites_threshold:
            for_screening[i]=True 
    return for_screening

second_consecutive_screening  = hpv.routine_screening(eligibility=second_screening_eligible, 
                                    start_year=GlobalScreeningParameters.screening_start_year,                                     
                                    age_range=[5,150],#overriding the default age range of [30,50] with a very large one - so as not to interfere with the age range I define in my eligibility checking
                                    prob=GlobalScreeningParameters.secondary_screen_prob, 
                                    product='hpv',        #Screening: DNA HPV Testing
                                    label='second_consecutive_screening') 

#-ve from second screening are returned to standard screening - their date_screened is updated from the second screening, so nothing needed for them
#+ve from second screening do second cytology
to_second_cytology = lambda sim: sim.get_intervention('second_consecutive_screening').outcomes['positive']
second_cytology = hpv.routine_triage(eligibility=to_second_cytology,
                                    prob = GlobalScreeningParameters.triage_screen_prob,
                                    age_range=[5,150], #I think routine traige doesnt introduce a constricted age range in by default, but in case it does, overriding it here
                                    product='lbc',
                                    annual_prob=False,
                                    label="second_cytology")

#-ve from second cytology do a third consecutive HPV screening in 12 months
def third_screening_eligible(sim):
    for_screening = np.array([False,]*sim.n)
    for i in sim.people.needs_consec_screening_3.keys():
        if sim.people.needs_consec_screening_3[i] is not None and  sim.people.needs_consec_screening_3[i] <= sim.t <= sim.people.needs_consec_screening_3[i] + GlobalScreeningParameters.abandon_followup_invites_threshold:
            for_screening[i]=True
    return for_screening

third_consecutive_screening  = hpv.routine_screening(eligibility=third_screening_eligible, 
                                    start_year=GlobalScreeningParameters.screening_start_year, 
                                    age_range=[5,150],#overriding the default age range of [30,50] with a very large one - so as not to interfere with the age range I define in my eligibility checking
                                    prob=GlobalScreeningParameters.third_screen_prob, 
                                    product='hpv',        #Screening: DNA HPV Testing
                                    label='third_consecutive_screening') 

#-ve from third screening are returned to standard screening. As their latest screening time is updated in {date_screened}, no need to do anything here
#+ve from third screening perform a third cyotology followed by a mandatory colposcopy
to_third_cytology = lambda sim: sim.get_intervention('third_consecutive_screening').outcomes['positive']
third_cytology = hpv.routine_triage(eligibility=to_third_cytology,
                                    prob = GlobalScreeningParameters.triage_screen_prob,
                                    age_range=[5,150], #I think routine traige doesnt introduce a constricted age range in by default, but in case it does, overriding it here
                                    product='lbc',
                                    annual_prob=False,
                                    label="third_cytology")

#Eligible for a colposcopy if any of the following hold: A)tested not normal for any cytology; B) went all the way to a third screening/cytology
colposcopy_eligible = lambda sim:list(set(sim.get_intervention('first_cytology').outcomes['abnormal'].tolist() + sim.get_intervention('first_cytology').outcomes['ascus'].tolist() +
                                        sim.get_intervention('second_cytology').outcomes['abnormal'].tolist() + sim.get_intervention('second_cytology').outcomes['ascus'].tolist() +
                                        sim.get_intervention('third_cytology').outcomes['normal'].tolist() + sim.get_intervention('third_cytology').outcomes['abnormal'].tolist() + sim.get_intervention('third_cytology').outcomes['ascus'].tolist()))


colposcopy =  hpv.routine_triage(eligibility=colposcopy_eligible, 
                                prob=GlobalScreeningParameters.colpo_prob, 
                                age_range=[5,150], #I think routine traige doesnt introduce a constricted age range in by default, but in case it does, overriding it here
                                product='colposcopy', 
                                annual_prob=False, 
                                label='colposcopy')

hsil_by_colpo = lambda sim: sim.get_intervention('colposcopy').outcomes['hsil'] #i.e. covers all women diagnosed with HSIL by a colposcopy
#ablation = hpv.treat_num(eligibility=hsil_by_colpo, 
ablation = treat_num_canceragnostic(eligibility=hsil_by_colpo, 
                        prob=GlobalScreeningParameters.ablate_prob, 
                        #No need to add an age range here, as by default subclasses of BaseTreatment have an age range of [0,99] - that is, they don't restrict the age range beyond any calculations in the provided eligibility function 
                        product='ablation', 
                        label='ablation') #TODO: can model a max capacity with treat_num too, but as it stands we pretend infinite capacity

cancer_by_colpo = lambda sim: sim.get_intervention('colposcopy').outcomes['cancer'] 
general_cancer_treatment = GeneralCancerTreatment(p_clear=GlobalScreeningParameters.cancer_treatment_effectiveness, p_extend=0.00, extend_dur=dict(dist='normal', par1=18.0, par2=2.))
gct = treat_num_cancer(eligibility=cancer_by_colpo, 
                        prob=GlobalScreeningParameters.generalcancertreatment_prob, 
                        #No need to add an age range here, as by default subclasses of BaseTreatment have an age range of [0,99] - that is, they don't restrict the age range beyond any calculations in the provided eligibility function
                        product=general_cancer_treatment, 
                        label='general_cancer_treatment') 






#TODO: neaten the below! and in checking all of this ofc I will need to clear it up adn probs make it all neater with a big restructure anyways

def get_interventions(l, m, end_screening_at_switch_year=False):
    interventions = [init_intervention_trackers] + get_routine_screening_interventions(l,m,end_screening_at_switch_year=end_screening_at_switch_year) + [
                            first_cytology, second_consecutive_screening, 
                                            second_cytology, third_consecutive_screening, third_cytology, 
                                            colposcopy, 
                                            ablation, gct, 
                                        update_last_hpv_result, update_needs_consec_screening_2,  update_needs_consec_screening_3,
                                       ]
    return interventions



if __name__=="__main__":
    #HPVsim simulation parameters
    pars = dict(
            n_agents      = 5e3,#20e3,       
            start=1970, end=2200,#2080,
            #verbose       = 0,   
            rand_seed     = 1, 
            genotypes     = [16,18, 'hi5'],
            burnin=30,
            location='united kingdom',
            )


    #Define simulation without interventions
    sim_basic = hpv.Sim(pars,
                        label='Sim, no interventions')

    #Define simulation with interventions
    sim_interventions =  hpv.Sim(pars, interventions = get_interventions(1,1)
                                 , #interventions can just be functions of a sim
                                            label='NHS Cervical Screening Pathway 2025')


    
    #Run and plot sims
    reduced_sims = []
    for sim in [sim_basic, sim_interventions]:
        msim = hpv.MultiSim(sim, n_runs=10)  
        msim.run()         
        reduced_sims.append(msim.reduce(use_mean=True, output=True))  


    msim = hpv.MultiSim(reduced_sims)
    msim.plot()
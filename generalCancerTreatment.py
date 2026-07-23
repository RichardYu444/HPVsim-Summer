"""
This is where I am trying to build an importable intervention for treating cancers, and treating cancers only (not CINs or anything else; the other treatments deal fine with this)


Original contributions relevant to this file:
    - Creating a HPVsim intervention which actively treats cancers (as opposed to treating stages before an agent is cancerous)
"""

import hpvsim_working as hpv

import numpy as np
import sciris as sc

from tqdm import tqdm

print(hpv.__path__)


class GeneralCancerTreatment(hpv.interventions.Product):
    """
    Models a general cancer treatment product. When applied to an agent with cervical cancer, it may:
        - clear the cancer completely
        - increase agent's years until death-due-to-cancer according to some distribution
        - do nothing 
    """
    def __init__(self,p_clear, p_extend=None, extend_dur=None):
        """
        Clears cancer with probability {p_clear}, extends with probability {p_extend}. If {p_clear+p_extend<1}, remaining probability is that of the treatment having no effect.
        {extend_dur} gives the distribtion from which we sample to get number of years by which to extend cancerous agent's lifespan if extending.
        If {p_extend} not specified, all agents are either cleared or extended.
        If {extend_dur} not specified, extention distribution consistnet with that in hpv.interventions.radiation
        """
        assert p_clear >=0
        if p_extend:
            assert p_extend>=0
            assert p_clear + p_extend <= 1 
        else:
            assert p_clear <= 1

        self.p_clear = p_clear
        if self.p_clear<1:
            self.p_extend_ifNotClear = p_extend/(1-p_clear) if p_extend is not None else 1 
        else:
            self.p_extend_ifNotClear = 0
        self.extend_dur = extend_dur or dict(dist='normal', par1=18.0, par2=2.)

        #print(f"self.p_extend_ifNotClear={self.p_extend_ifNotClear}")

    def administer(self, sim, inds):
        if len(inds)==0:
            return {'cleared'   : [],  'extended'  : [], 'failed'    : []}

        people = sim.people

        #Only allow this treatment to be applied to cancerous patients
        cancerous_inds = people.true("cancerous")
        inds = np.array(list(set(inds) & set(cancerous_inds))).astype(int) #intersection of two lists of indices

        #Determine which indices undergo which treatments
        clear_probs = np.full(len(inds), self.p_clear, dtype=hpv.defaults.default_float)
        to_clear = hpv.utils.binomial_arr(clear_probs) #note that clear_probs, to_clear are lists of length len(inds); that is, their numbers refer to the list of pids that have been sent to this function for cancer treatment
        to_clear_inds = inds[np.where(to_clear==True)[0]] #{to_clear_inds} is a list of person id's

        extend_probs = np.full(len(inds), self.p_extend_ifNotClear, dtype=hpv.defaults.default_float)
        for i in np.where(to_clear==True)[0]: extend_probs[i] = 0 #agent may only be extended if not cleared
        to_extend = hpv.utils.binomial_arr(extend_probs)
        to_extend_inds = inds[np.where(to_extend==True)[0]]

        #Clear cancers for those to be cleared
        for ind in to_clear_inds: #TODO: parelellise this like the HPVsim source code, I think when I refer to [ind] in the index I can also do a list of inds or smth like that, and get several in at once!
            for genotype in range(sim.people.cancerous.shape[0]):
                # Clear cancer
                people.cancerous[genotype, ind] = False
                # Record the clearance
                people['date_cancerous'][genotype, ind] = np.nan
                people['date_clearance'][genotype, ind] = people.t + 1
                #print(f"{ind} cleared!")
                
        #Extend lifetime for relevant inds
        new_durs = hpv.utils.sample(**self.extend_dur, size=len(to_extend_inds)) #samples a vector (1 element for each person to be extended) of how much additional time they have until dying of cancer (added to the time before) due to this radiotherapy treatment
        people.date_dead_cancer[to_extend_inds] += np.ceil(new_durs / people.pars['dt']) #NOTE: while all agents have a date_dead_cancer property, it only is used if they have cancer

        # Bookkeeping of cancer treatments - individual level (note that we use {inds}, not {to_clear_inds+to_extend_inds}, because any remaining agents still are modelled to have had a treatment, just a failed one)
        people.cancer_treated[inds] = True #mark this person as having undergone *some* form of cancer treatment
        people.cancer_treatments[inds] += 1 #this tracks the number of (any form of) cancer treatment a person has undergone
        people.date_cancer_treated[inds] = sim.t #update the time of this person's most recent cancer treatment with the current timestep

        # Store population-level results - as above, we use {inds}, not {to_clear_inds+to_extend_inds}
        idx = int(sim.t / sim.resfreq)
        new_cctreat_inds = hpv.utils.ifalsei(sim.people.cancer_treated, inds)  # Figure out people who are getting this treament as their first cancer treatment of any kind
        n_new_radiaitons = sim.people.scale_flows(inds)  # Scale
        n_new_people = sim.people.scale_flows(new_cctreat_inds)  # Scale
        sim.results['new_cancer_treated'][idx] += n_new_people
        sim.results['new_cancer_treatments'][idx] += n_new_radiaitons

        #if len(to_clear_inds)>0:
         #   print(to_clear_inds)

        #Compute outcomes for treatment montioring 
        outcomes = {'cleared'   : np.array(list(set(to_clear_inds))), 
                    'extended'  : np.array(list(set(to_extend_inds))),
                    'failed'    : np.setdiff1d(inds, list(set(to_clear_inds))+list(set(to_extend_inds)))
                    }
        return outcomes


class treat_num_cancer(hpv.interventions.treat_num):
    '''
    Treat a fixed number of people each timestep *for cancer*. 
    Unlike treat_num (and all children of hpv.interventions.BaseTreatment without function overriding), which make cancerous agents ineligible for treatment, here cancerous patients ONLY are eligible for treatment.

    Args:
         max_capacity (int): maximum number who can be treated each timestep
    '''
    #Override this function to make cancerous agents ONLY eligible for treatment
    def check_eligibility(self, sim):
        '''
        Check people's eligibility for treatment
        '''
        females         = sim.people.is_female
        in_age_range    = (sim.people.age >= self.age_range[0]) * (sim.people.age <= self.age_range[1])
        alive           = sim.people.alive
        cancer          = sim.people.cancerous.any(axis=0)
        conditions      = (females * in_age_range * alive * cancer)
        return conditions
    """
    def initialize(self, sim):
        super().initialize(sim)
        self.outcomes = {k: np.array([], dtype=hpv.defaults.default_int) for k in ['cleared', 'extended', 'failed']} 
"""

class treat_num_canceragnostic(hpv.interventions.treat_num):
    '''
    Treat a fixed number of people each timestep and agnostic to whether the agent does or doesn't have cancer
    
    Args:
         max_capacity (int): maximum number who can be treated each timestep
    '''
    #Override this function to make cancerous agents ONLY eligible for treatment
    def check_eligibility(self, sim):
        '''
        Check people's eligibility for treatment
        '''
        females         = sim.people.is_female
        in_age_range    = (sim.people.age >= self.age_range[0]) * (sim.people.age <= self.age_range[1])
        alive           = sim.people.alive
        conditions      = (females * in_age_range * alive)
        return conditions
    """
    def initialize(self, sim):
        super().initialize(sim)
        self.outcomes = {k: np.array([], dtype=hpv.defaults.default_int) for k in ['cleared', 'extended', 'failed']} 
"""


if __name__ =="__main__":
    #An eligibility function `for delivering a treatment to absolutely everyone, every 5 timesteps.
    def eligible(sim):
        for_screening = np.isnan(sim.people.date_screened) & (sim.people.age>=24) & (sim.people.age<=60) #eligible for screening if they have not yet been screened
        
        for i in range(sim.n): #sim.n==len(sim.people)
            for_screening[i] = True
        

        regularity = 10 #deliver the treatment every {regularity} timesteps

        if sim.t % regularity == 0:
            return np.where(for_screening==True)
        else:
            return np.where(for_screening==False)


    general_cancer_treatment = GeneralCancerTreatment(p_clear=0.90, p_extend=0.00, extend_dur=dict(dist='normal', par1=18.0, par2=2.))


    gct = treat_num_cancer(eligibility=eligible, 
                                            prob=0.9, 
                                            product=general_cancer_treatment, #my_radiation radiation 'ablation' 'cancerPerfectTreatment'
                                            label='gct') 


    # Create and run the sim
    sim = hpv.Sim(rand_seed=1,
                start=1970, end=2010,
                n_agents=5e3,#1e3,#1e4,
                interventions=[gct],
                )
    sim.run()

    df = sim.to_df()
    df.to_csv("perfectCancerTreatmentTesting.csv")
    #print(df.columns)

    #print(df['cancer_deaths'])

    #print(df['cum_cancer_treated'])

    #sim.plot()

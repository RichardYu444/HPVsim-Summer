"""
Docstring for InterventionAlgorithms.complianceAndCancerTreatmentParameters

Set parameters for compliance and general cancer treatment efficacy here, for easy sensitivity analysis
"""
import numpy as np
from math import floor

#Set treatment efficacies
cancer_treatment_effectiveness = 0.8

#Set compliance levels
projected_teen_vaccination_uptake = 0.8 #from 2022 onwards, the probability that an agent in the UK at age 12-13 will take up the HPV vaccine dose offered to them at some point in that time 

primary_screen_prob_under50 = 0.68 #Probability of a person under 50 taking up their primary screening invitation
primary_screen_prob_50andover = 0.76 #Probability of a person 50 or over taking up their primary screening invitation
secondary_screen_prob = 0.8 #probability that an agent invited to secondary screening at timepoint T attends within {abandon_followup_invites_threshold} years
third_screen_prob =  0.8  #probability that an agent invited to third screening at timepoint T attends within {abandon_followup_invites_threshold} years
triage_screen_prob = 1.00
colpo_prob = 0.95
ablate_prob = 0.90
generalcancertreatment_prob = 0.90

#Set screening policy
screening_start_year = 1980 #year at which we begin modelling our screening in each sim
switch_year = 2026 #this is the year at which we switch from the NHS 2025 algorithm to alternative algorithms
abandon_followup_invites_threshold = 1  #the number of years until which we stop sending invites to a followup SCREENING (i.e. NOT RELEVANT to primary screenings), after the invites have not been taken up. Modelling: if a patient is due a followup screning at time t_1, then they remain elibile for all times t satisfying t_1 <= t<=t_1 + abandon_followup_invites_threshold
#NOTE: all other followups are happen immediately (within the same timestep) and are modelled as 1-and-done - that is, an agent which doesn't take up the treatment in that timestep will never take up that treatment
 #NOTE: and the initial screening (i.e. the only thing not a followup) is modelled as a repeated probability again and again - so if it is not an annual probability it works out not quite right.

#TODO: all the changed I have made for NHS_2025_lambdamu.py need to also go into the other screening policy file (maybe recopy and remake the latter from the updated former?)


#Logic to adjust secondary_screen_prob and third_screen_prob according to the number of years we allow for followup
secondary_screen_prob = 1 - np.pow(1-secondary_screen_prob, 1/(floor(abandon_followup_invites_threshold)+1)) #within each round of attempting to get people to do their secondary screening, of those eligible we assume it is i.i.d. between them
third_screen_prob = 1 - np.pow(1-third_screen_prob, 1/(floor(abandon_followup_invites_threshold)+1))
"""
Derivation of the above: 
event has prob p. We want, in a sequence of n attempts of teh event, to have the event happen at least once with prob gamma.

P(at least 1 occurance in n trials) = 1-P(no occurance in n trials)
                                    = 1 - (1-p)^n
                                    = gamma

Therefore,
    (1-p)^n = 1-gamma
    1-p = (1-gamma)^(1/n)
    p=1-(1-gamma)^(1/n)
"""
    #NOTE on the interpretation of these probabilities: in their raw (hardcoded) form above, they represent the probability of a person offered a particular intervention taking it up at all. When processed (as any which need processing are done by here in the file), they represent year-wise probabiolities of an agnet taking up a particular reccomended treatment - the probabolity mass is not equally distributed across the year, however, as HPVsim can't know in advance how many times an agent will be offered a particular treatment in a given year, so instead each agent has ONE chance per year to get the treatment (even if reccomended it on several timesteps that year), which is the first time taht year, and then thier next chance at getting the treatment isd the first offering of the treatment next year.
        #^ this is how I understand the probabilities to work, given when I change abandon_followup_invites_threshold without changing its floor, nothing changes - while if I take away the 'floor' functions from my formalae and change, say, 1 to 0.99, it all goes out of whack
         #TODO: confirm the above with Robyn,but I am pretty confident this is correct and therefore i can move on in the meantime!
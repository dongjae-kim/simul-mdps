import numpy as np
import scipy as sci
import scipy.special as sc
import time
import collapsed_gibbs_sampler as cgs
np.random.seed(np.int64(time.time()))


def initialize_pe(behavioral_data,param_set,mode):
    # This part cal PEs due to simulation models.
    # TODO: fill the simulation model's PE generation part..

    n_state = 9
    n_action = 2
    n_session = 4
    n_session_trial = [166, 152, 154, 164]
    SPE = np.zeros((1,636))
    SPE_T = []
    for i in range(n_session):
        SPE_T.append(np.zeros((n_state,n_action,n_state,n_session_trial[i])))
    RPE = np.zeros((1,636))
    RPE_SARSA = []
    for i in range(n_session):
        RPE_SARSA.append(np.zeros((n_session_trial[i], 5))) # 5 for sarsa
    return [SPE, SPE_T, RPE, RPE_SARSA]



def init_dpgmm(SPE,RPE,alpha=1, num_sweeps = 1000):
    [class_id, K_record, lp_record, alpha_record] = cgs.sampler(SPE,alpha=1, num_sweeps = num_sweeps)
    MAX=np.max(class_id[:,num_sweeps-1])
    class_id_t = class_id[:,num_sweeps-1]
    Y=[]
    for i in range(MAX):
        Y.append(SPE[0,np.where(class_id_t==(i+1))])
    SPE_ = dict(SAMPLED_CLASS = class_id_t, CLUSTER_DATA = Y, NUMBER_OF_CLUSTER = K_record, MODEL_SCORE = lp_record, ALPHA_RECORD = alpha_record)


    [class_id, K_record, lp_record, alpha_record] = cgs.sampler(RPE,alpha=1, num_sweeps = num_sweeps)
    MAX=np.max(class_id[:,num_sweeps-1])
    class_id_t = class_id[:,num_sweeps-1]
    Y=[]
    for i in range(MAX):
        Y.append(RPE[0,np.where(class_id_t==(i+1))])
    RPE_ = dict(SAMPLED_CLASS = class_id_t, CLUSTER_DATA = Y, NUMBER_OF_CLUSTER = K_record, MODEL_SCORE = lp_record, ALPHA_RECORD = alpha_record)

    return [SPE_, RPE_]

def pe2(behavioral_data,param_set,mode):
    [SPE, SPE_T, RPE, RPE_SARSA] = initialize_pe(behavioral_data,param_set,mode) # TODO: fill the simulation model's PE generation part..
    [SPE_, RPE_]                 = init_dpgmm(SPE, RPE, alpha=1, num_sweeps=1000)
    return [SPE_, RPE_, SPE_T, RPE_SARSA]
""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import torch
import numpy as np
import pandas as pd
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random

from tqdm import tqdm
from mdp_task3 import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator, BayesRelEstimator_zpe, Arbitrator_zpe
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS, Q_COLUMNS
from common import makedir

import zero_pe

# preset constants
MDP_STAGES            = 2
TOTAL_EPISODES        = 100
TRIALS_PER_EPISODE    = 80
SPE_LOW_THRESHOLD     = 0.25#0.3
SPE_HIGH_THRESHOLD    = 0.45#0.5
RPE_LOW_THRESHOLD     = 4
RPE_HIGH_THRESHOLD    = 9 #10
MF_REL_HIGH_THRESHOLD = 0.8
MF_REL_LOW_THRESHOLD  = 0.5
MB_REL_HIGH_THRESHOLD = 0.7
MB_REL_LOW_THRESHOLD  = 0.3
CONTROL_REWARD        = 1
CONTROL_REWARD_BIAS   = 0
INIT_CTRL_INPUT       = [10, 0.5]
DEFAULT_CONTROL_MODE  = 'max-spe'
CONTROL_MODE          = DEFAULT_CONTROL_MODE
CTRL_AGENTS_ENABLED   = False
RPE_DISCOUNT_FACTOR   = 0.003
ACTION_PERIOD         = 3
STATIC_CONTROL_AGENT  = False
ENABLE_PLOT           = True
DISABLE_C_EXTENSION   = False
LEGACY_MODE           = False
MORE_CONTROL_INPUT    = True
SAVE_CTRL_RL          = False
MODEL_LIST = ['MB','MF','LEE2014','ZPE']
MODEL_LIST_IDF = -1

RESET = False
SAVE_LOG_Q_VALUE = True
MIXED_RANDOM_MODE = False
RANDOM_MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe']

# ADDED BY DONGAJE KIM
ARBI_SIMULGEN_MODE = True
TRIALS_PER_EPISODE_sbjtv = []
CTXT_sbjtv = []
MODEL_LIST_IDF = 0
TASK_IDF = 0
# toggle of task idf
# 0. ladder and tree
ladder_tree_ = 1
# 1. GRW payoff or not
GRW_ = 0 #1: GRW payoff 0 for legacy mode 2: for every state transition
# 2. num_action applicable
num_action_ = 2


error_reward_map = {
    # x should be a 4-tuple: rpe, spe, mf_rel, mb_rel
    'min-rpe' : (lambda x: x[0] < RPE_LOW_THRESHOLD),
    'max-rpe' : (lambda x: x[0] > RPE_HIGH_THRESHOLD),
    'min-spe' : (lambda x: x[1] < SPE_LOW_THRESHOLD),
    'max-spe' : (lambda x: x[1] > SPE_HIGH_THRESHOLD),
    'min-mf-rel' : (lambda x: x[2] < MF_REL_LOW_THRESHOLD),
    'max-mf-rel' : (lambda x: x[2] > MF_REL_HIGH_THRESHOLD),
    'min-mb-rel' : (lambda x: x[3] < MB_REL_LOW_THRESHOLD),
    'max-mb-rel' : (lambda x: x[3] > MB_REL_HIGH_THRESHOLD),
    'min-rpe-min-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['min-spe'](x),
    'max-rpe-max-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['max-spe'](x),
    'min-rpe-max-spe' : lambda x: error_reward_map['min-rpe'](x) and error_reward_map['max-spe'](x),
    'max-rpe-min-spe' : lambda x: error_reward_map['max-rpe'](x) and error_reward_map['min-spe'](x)
}

def create_lst(x):
    return [x] * TRIALS_PER_EPISODE

static_action_map = {
    'min-rpe' : create_lst(0),
    'max-rpe' : create_lst(3),
    'min-spe' : create_lst(0),
    'max-spe' : create_lst(1),
    'min-rpe-min-spe' : create_lst(0),
    'max-rpe-max-spe' : create_lst(3),
    'min-rpe-max-spe' : create_lst(1),
    'max-rpe-min-spe' : create_lst(2)
}
static_action_map2 = {
    'min-rpe' : create_lst(0),
    'max-rpe' : create_lst(3),
    'min-spe' : create_lst(0),
    'max-spe' : create_lst(1),
    'min-rpe-min-spe' : create_lst(0),
    'max-rpe-max-spe' : create_lst(3),
    'min-rpe-max-spe' : create_lst(1),
    'max-rpe-min-spe' : create_lst(2)
}

def error_to_reward(error, mode=DEFAULT_CONTROL_MODE, bias=CONTROL_REWARD_BIAS):
    try:
        cmp_func = error_reward_map[mode]
    except KeyError:
        print("Warning: control mode {0} not found, use default mode {1}".format(mode, DEFAULT_CONTROL_MODE))
        cmp_func = error_reward_map[DEFAULT_CONTROL_MODE]

    if cmp_func(error):
        return CONTROL_REWARD + bias
    else:
        return bias

def compute_human_action(arbitrator, human_obs, model_free, model_based):
    """Compute human action by compute model-free and model-based separately
    then integrate the result by the arbitrator

    Args:
        arbitrator (any callable): arbitrator object
        human_obs (any): valid in human observation space
        model_free (any callable): model-free agent object
        model_based (any callable): model-based agent object
    
    Return:
        action (int): action to take by human agent
    """
    return arbitrator.action(model_free.get_Q_values(human_obs),
                             model_based.get_Q_values(human_obs))
def compute_likelihood(arbitrator, human_obs, human_action, model_free, model_based):
    p_s = arbitrator.get_likelihood(model_free.get_Q_values(human_obs),
                             model_based.get_Q_values(human_obs))
    return p_s[human_action]

def simulation(threshold=BayesRelEstimator.THRESHOLD, estimator_learning_rate=AssocRelEstimator.LEARNING_RATE,
               amp_mb_to_mf=Arbitrator.AMPLITUDE_MB_TO_MF, amp_mf_to_mb=Arbitrator.AMPLITUDE_MF_TO_MB,
               temperature=Arbitrator.SOFTMAX_TEMPERATURE, rl_learning_rate=SARSA.LEARNING_RATE, performance=300, PARAMETER_SET='DEFAULT',
               return_res=False, start_con=False, con=None, continue_gData=False, transferred_gData=None, continue_steps=False, con2=None, behave_=[]):
    CHANGE_MODE_TERM = int(TOTAL_EPISODES) # so it means we are not going to change mode since we have only one model in simulation generation process.
    behave_ = np.array(behave_).astype(np.int32)
    if return_res:
        res_data_df = pd.DataFrame(columns=COLUMNS)
        res_detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)
        res_Q_df = pd.DataFrame(columns=Q_COLUMNS)

    if GRW_ == 0:
        # trans_prob = [0.7, 0.3]
        env = MDP(MDP_STAGES, num_actions=num_action_, more_control_input=MORE_CONTROL_INPUT, \
              legacy_mode=LEGACY_MODE, ladder_tree=ladder_tree_, GRW=GRW_)
    elif GRW_ == 1:
        trans_prob = [0.7, 0.3]
        env = MDP(MDP_STAGES, trans_prob=trans_prob, num_actions=num_action_, more_control_input=MORE_CONTROL_INPUT, \
                  legacy_mode=LEGACY_MODE, ladder_tree=ladder_tree_, GRW=GRW_)
    elif GRW_ == 2:
        trans_prob = [0.7, 0.3]
        env = MDP(MDP_STAGES, trans_prob=trans_prob, num_actions=num_action_, more_control_input=MORE_CONTROL_INPUT, \
                  legacy_mode=LEGACY_MODE, ladder_tree=ladder_tree_, GRW=GRW_)
    elif GRW_ == 3:
        trans_prob = [0.7, 0.3]
        env = MDP(MDP_STAGES, trans_prob=trans_prob, num_actions=num_action_, more_control_input=MORE_CONTROL_INPUT, \
                  legacy_mode=LEGACY_MODE, ladder_tree=ladder_tree_, GRW=GRW_)
    elif GRW_ == 4:
        trans_prob = [0.7, 0.3]
        env = MDP(MDP_STAGES, trans_prob=trans_prob, num_actions=num_action_, more_control_input=MORE_CONTROL_INPUT, \
                  legacy_mode=LEGACY_MODE, ladder_tree=ladder_tree_, GRW=GRW_)
    else:
        env = MDP(MDP_STAGES, more_control_input=MORE_CONTROL_INPUT, legacy_mode=LEGACY_MODE)


    # ladder_tree_ = 0
    # fixed_trans_ = 1  # 1:  0.7 and 0.3, 0 for legacy mode
    # GRW_ = 0  # 1: GRW payoff 0 for legacy mode
    # num_action_ = 2


    # if MIXED_RANDOM_MODE:
    #     random_mode_list = np.random.choice(RANDOM_MODE_LIST, 4 , replace =False) # order among 4 mode is random
    #     print ('Random Mode Sequence : %s' %random_mode_list)

    # if it is mixed random mode, 'ddpn_loaded' from torch.save(model, filepath) is used instead of 'ddpn' 
    # ddqn    = DoubleDQN(env.observation_space[MDP.CONTROL_AGENT_INDEX],
    #                     env.action_space[MDP.CONTROL_AGENT_INDEX],
    #                     torch.cuda.is_available()) # use DDQN for control agent

    # gData.new_simulation()
    # gData.add_human_data([amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature, performance])
    # control_obs_extra = INIT_CTRL_INPUT

    control_obs_extra = INIT_CTRL_INPUT

    # if continue_gData:
    #     sarsa.set_Q_values     ## defaultdict
    #     forward.set_Q_values ## defaultdict
    #     forward.set_Transition ## {}
    #     ## continuing development


    human_action_list_t = []
    Task_structure = []
    Q_value_forward_t = []
    Q_value_sarsa_t = []
    Q_value_arb_t = []
    Transition_t = []

    # ADDED
    BHV_sim = []
    BEHAVIOR_SIMUL_COLUMN_ORI = gData.BEHAVIOR_SIMUL_COLUMN

    gData.new_simulation()

    if not RESET:
        # initialize human agent one time
        if MODEL_LIST_IDF == 3:
            sarsa = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX],
                          learning_rate=rl_learning_rate)  # SARSA model-free learner
            forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                              env.action_space[MDP.HUMAN_AGENT_INDEX],
                              env.state_reward_func, env.output_states_offset, env.reward_map_func, env.output_states,
                              learning_rate=rl_learning_rate,
                              disable_cforward=DISABLE_C_EXTENSION)  # forward model-based learner
            arb     = Arbitrator_zpe(BayesRelEstimator_zpe(learning_rate=rl_learning_rate),
                                BayesRelEstimator_zpe(learning_rate=rl_learning_rate),
                                amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature)

            '''
            spe = forward.optimize(human_obs, human_action, next_human_obs)
            next_human_action = compute_human_action(arb, next_human_obs, sarsa,
                                                     forward)  # required by models like SARSA
            if env.is_flexible == 1:  # flexible goal condition
                rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
            else:  # specific goal condition human_reward should be normalized to sarsa
                if human_reward > 0:  # if reward is 10, 20, 40
                    rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
                else:
                    rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
            '''
        else:
            sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], learning_rate=rl_learning_rate) # SARSA model-free learner
            forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                            env.action_space[MDP.HUMAN_AGENT_INDEX],
                            env.state_reward_func, env.output_states_offset, env.reward_map_func, env.output_states,
                                learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION) # forward model-based learner
            arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                                BayesRelEstimator(thereshold=threshold),
                                amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature)
            # register in the communication controller
        env.agent_comm_controller.register('model-based', forward)

        if MODEL_LIST_IDF == 3:
            human_obs, control_obs_frag = env.reset()
            bwd_idf = -1
            bwd_idf_prev = -1
            try:
                BHV_trial = np.array(behave_[0, :, :]).astype(dtype=np.int16)
            except:
                BHV_trial = behave_[0, :]
            spe_hist = []
            rpe_hist = []
            for tr_i in range(behave_.shape[1]):
                BHV_trial_temp = np.array(BHV_trial[tr_i][:19]).astype(np.int32).tolist()
                bwd_idf = BHV_trial_temp[17]
                human_obs, control_obs_frag = env.reset()
                game_terminate = False

                if MODEL_LIST_IDF == 0:
                    arb.p_mb = 0.999
                    arb.p_mf = 0.001
                elif MODEL_LIST_IDF == 1:
                    arb.p_mb = 0.001
                    arb.p_mf = 0.999
                else:
                    ''
                if env.GRW == 0:
                    if bwd_idf == -1:
                        env.is_flexible = 1
                    else:
                        env.is_flexible = 0
                elif env.GRW == 1:
                    env.is_flexible = 2  # GRW
                elif env.GRW == 2:
                    env.is_flexible = 2  # GRW

                env.ctxt = CTXT_sbjtv[tr_i % len(CTXT_sbjtv)][0]

                if env.is_flexible == 1:
                    arb.p_mb = 0.8
                    arb.p_mf = 0.2
                else:
                    arb.p_mb = 0.2
                    arb.p_mf = 0.8
                # udpate for bwd

                if env.GRW == 0:
                    if bwd_idf != bwd_idf_prev:
                        forward.bwd_update(bwd_idf, env)

                trial_index = 0
                while not game_terminate:
                    human_obs = BHV_trial_temp[3 + trial_index]

                    """human choose action"""
                    human_action = BHV_trial_temp[6 + trial_index]

                    """human act on environment"""
                    next_human_obs = BHV_trial_temp[4 + trial_index]
                    if next_human_obs < env.output_states_offset:
                        human_reward = 0
                    else:
                        human_reward = env.output_states[next_human_obs - 6]

                    if trial_index == 1:
                        game_terminate = True

                    """update human agent"""
                    spe = forward.optimize(human_obs, human_action, next_human_obs)
                    next_human_action = int(BHV_trial_temp[7 + trial_index])
                    if trial_index == 1:
                        next_human_action = 0
                    if env.is_flexible == 1:  # flexible goal condition
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs,
                                             next_human_obs)
                    else:  # specific goal condition human_reward should be normalized to sarsa
                        if human_reward == forward.output_states[next_human_obs - 5]:  # if reward is 10, 20, 40
                            rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs,
                                                 next_human_obs)
                        else:
                            rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
                    if trial_index == 1:
                        if env.is_flexible == 1:
                            BHV_trial_temp[15] = human_reward
                        else:
                            BHV_trial_temp[15] = forward.output_states[next_human_obs - 5]

                    """iterators update"""
                    human_obs = next_human_obs


                    trial_index += 1
                    spe_hist.append(spe)
                    rpe_hist.append(rpe)

                bwd_idf_prev = bwd_idf
            spe__, rpe__ = zero_pe.init_dpgmm((np.array(spe_hist)).reshape(1, -1), (np.array(rpe_hist)).reshape(1, -1),
                                              1, 500)
            zpe__ = dict(SPE=np.array(spe_hist), RPE=np.array(rpe_hist), Z_SPE=spe__, Z_RPE=rpe__)

    for episode in tqdm(range(TOTAL_EPISODES)):

        gData.new_simulation()
        gData.add_human_data(
            [amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature,
             performance])
        control_obs_extra = INIT_CTRL_INPUT


        if RESET:
            # reinitialize human agent every episode
            if MODEL_LIST_IDF == 3:
                sarsa = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX],
                              learning_rate=rl_learning_rate)  # SARSA model-free learner
                forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                                  env.action_space[MDP.HUMAN_AGENT_INDEX],
                                  env.state_reward_func, env.output_states_offset, env.reward_map_func, env.output_states,
                                  learning_rate=rl_learning_rate,
                                  disable_cforward=DISABLE_C_EXTENSION)  # forward model-based learner
                arb     = Arbitrator_zpe(BayesRelEstimator_zpe(learning_rate=rl_learning_rate),
                                    BayesRelEstimator_zpe(learning_rate=rl_learning_rate),
                                    amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb, temperature=temperature)
                arb.mf_rel_estimator.init_zpe(zpe__["RPE"],zpe__["Z_RPE"])
                arb.mf_rel_estimator.init_zpe(zpe__["SPE"],zpe__["Z_SPE"])
            else:
                sarsa   = SARSA(env.action_space[MDP.HUMAN_AGENT_INDEX], learning_rate=rl_learning_rate) # SARSA model-free learner
                forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                            env.action_space[MDP.HUMAN_AGENT_INDEX],
                            env.state_reward_func, env.output_states_offset, env.reward_map_func, env.output_states,
                                learning_rate=rl_learning_rate, disable_cforward=DISABLE_C_EXTENSION) # forward model-based learner
                arb     = Arbitrator(AssocRelEstimator(estimator_learning_rate, env.max_rpe),
                                BayesRelEstimator(thereshold=threshold),
                                amp_mb_to_mf=amp_mb_to_mf, amp_mf_to_mb=amp_mf_to_mb)
                # register in the communication controller
            env.agent_comm_controller.register('model-based', forward)
        human_obs, control_obs_frag = env.reset()

        # if MIXED_RANDOM_MODE and episode % CHANGE_MODE_TERM ==0: # Load ddqn model from exist torch.save every CHANGE_MODE_TERM
        #     random_mode_index = int(episode/CHANGE_MODE_TERM)
        #     CONTROL_MODE_temp = random_mode_list[random_mode_index]
        #     print ('Load DDQN model. Current model : %s Current episode : %s' %(CONTROL_MODE_temp, episode))
        #     ddqn_loaded = torch.load('ControlRL/' + CONTROL_MODE_temp + '/MLP_OBJ_Subject' + PARAMETER_SET)
        #     ddqn_loaded.eval() # evaluation mode
        #     ddqn.eval_Q = ddqn_loaded

        cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = 0
        cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
        BHV_trial = []
        human_action_list_episode = []
        bwd_idf = -1
        bwd_idf_prev = -1
        if MODEL_LIST_IDF == 3:
            BHV_trial_ = behave_[episode,:]
        for trial in range(TRIALS_PER_EPISODE_sbjtv):
            BHV_trial_temp = [episode+1, trial+1, CTXT_sbjtv[trial][0], 0, -999 ,-999, -999, -999, -999, -999, -999, -999, -999, -999, \
                              -999, -999, -999, CTXT_sbjtv[trial][1]]
                                # -999 indicates there is no valid data currently.
                                # some of them sould be filled in the followed section (while game_terminate)
            GRW_hist = []

            if MODEL_LIST_IDF == 3:
                BHV_trial_temp_ = BHV_trial_[trial,:]
            bwd_idf = BHV_trial_temp[-1]
            t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = 0
            human_obs, control_obs_frag = env.reset()
            game_terminate              = False
            control_obs                 = np.append(control_obs_frag, control_obs_extra)

            """control agent choose action"""
            # if STATIC_CONTROL_AGENT:
            #     if episode < TOTAL_EPISODES-1:
            #         control_action = random.randint(0,3)
            #     else:
            #         control_action = static_action_map[CONTROL_MODE][trial]
            # else:
            #     control_action = ddqn.action(control_obs)
            # cum_ctrl_act[control_action] += 1
            #
            # if control_action == 3:

            # update env
            if env.GRW == 0:
                if bwd_idf == -1:
                    env.is_flexible=1
                else:
                    env.is_flexible=0
            elif env.GRW == 3:
                if bwd_idf == -1:
                    env.is_flexible=1
                else:
                    env.is_flexible=0
            elif env.GRW == 4:
                if bwd_idf == -1:
                    env.is_flexible=1
                else:
                    env.is_flexible=0

            elif env.GRW == 1: # only for the pay-off
                env.is_flexible=2 # GRW
            elif env.GRW == 2: # for both
                env.is_flexible=2 # GRW

            env.ctxt = CTXT_sbjtv[trial][0]
            env._update_trans_prob()

            if MODEL_LIST_IDF == 0:
                    arb.p_mb = 0.999
                    arb.p_mf = 0.001
            elif MODEL_LIST_IDF == 1:
                    arb.p_mb = 0.001
                    arb.p_mf = 0.999
            else:
                ''
            if env.is_flexible == 1:
                arb.p_mb = 0.8
                arb.p_mf = 0.2
            else:
                arb.p_mb = 0.2
                arb.p_mf = 0.8
            # udpate for bwd
            if env.GRW == 0:
                if bwd_idf != bwd_idf_prev:
                    forward.bwd_update(bwd_idf, env)



            if SAVE_LOG_Q_VALUE:
                Task_structure.append( np.concatenate((env.reward_map, env.trans_prob, env.output_states), axis=None) )


            # """control act on environment"""
            # if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
            #     _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])

            trial_index = 0
            likelihood_=[]
            for trial_index in range(2):
            # while not game_terminate:
                if trial_index ==0:
                    BHV_trial_temp[3] = human_obs
                """human choose action"""
                if MODEL_LIST_IDF==3:
                    human_action_ = behave_[episode, trial, 6 + trial_index]
                    BHV_trial_temp_[6+trial_index] = human_action_
                    human_action = compute_human_action(arb, human_obs, sarsa, forward)
                else:
                    human_action = compute_human_action(arb, human_obs, sarsa, forward)
                likelihood_.append(compute_likelihood(arb, human_obs, human_action, sarsa, forward))
                BHV_trial_temp[6+trial_index] = human_action
#                if episode < TOTAL_EPISODES:
#                    human_action = random.randint(0,1)
#                 if start_con:
#                     if env.is_flexible != con[0]:
#                         env._specific_flexible_switch()
#                     if env.trans_prob[0]!= con[1]:
#                         env.trans_prob[0] = con[1]
#                         env.trans_prob[1] = 1-con[1]

                # if BHV_trial_temp[6+trial_index] == -999:
                #     print('bwd update')
                #print("human action : ", human_action)
                if SAVE_LOG_Q_VALUE:
                    human_action_list_episode.append(human_action)

                """human act on environment"""

                if MODEL_LIST_IDF == 3:
                    next_human_obs_ = behave_[episode, trial, 4 + trial_index]
                    if next_human_obs_<env.output_states_offset:
                        human_reward_ = 0
                    else:
                        human_reward_ = env.output_states[next_human_obs_-6]
                    BHV_trial_temp_[4+trial_index] = next_human_obs_
                    if env.is_flexible == 1:
                        BHV_trial_temp_[15] = human_reward_
                    else:
                        BHV_trial_temp_[15] = forward.output_states[next_human_obs_-5]

                    next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                        = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
                else:
                    next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                        = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
                    BHV_trial_temp[15] = human_reward
                if env.GRW != 0 and trial_index == 0:
                    GRW_hist.append(env.GRW_list[trial_index][0])
                elif env.GRW != 0:
                    GRW_hist = [*GRW_hist, *np.array(env.GRW_list)[:,0].tolist()]
                BHV_trial_temp[4+trial_index] = next_human_obs

                if MODEL_LIST_IDF == 3:
                    """update human agent"""
                    # spe_ = forward.optimize(human_obs, human_action_, next_human_obs_)
                    spe_ = zpe__["SPE"][trial]
                    spe  = forward.optimize(human_obs, human_action, next_human_obs)
                    next_human_action  = compute_human_action(arb, next_human_obs, sarsa, forward) # required by models like SARSA
                    if env.is_flexible == 1: #flexible goal condition
                        rpe_ = zpe__["RPE"][trial]
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                    else: # specific goal condition human_reward should be normalized to sarsa
                        if human_reward == forward.output_states[next_human_obs-5]: # if reward is 10, 20, 40
                            rpe_ = zpe__["RPE"][trial]
                            rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                        else:
                            rpe_ = zpe__["RPE"][trial]
                            rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
                else:
                    """update human agent"""
                    spe = forward.optimize(human_obs, human_action, next_human_obs)
                    next_human_action = compute_human_action(arb, next_human_obs, sarsa, forward) # required by models like SARSA
                    if env.is_flexible == 1: #flexible goal condition
                        rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                    else: # specific goal condition human_reward should be normalized to sarsa
                        if human_reward == forward.output_states[next_human_obs-5]: # if reward is 10, 20, 40
                            rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
                        else:
                            rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
                if trial_index == 1:
                    if env.is_flexible == 1:
                        BHV_trial_temp[15] = human_reward
                    else:
                        BHV_trial_temp[15] = forward.output_states[next_human_obs-5]

                if max(env.output_states) < BHV_trial_temp[15] :
                    print('')


                if MODEL_LIST_IDF == 3:
                    mf_rel, mb_rel, p_mb = arb.add_pe(rpe_, spe_, zpe__["Z_RPE"]["SAMPLED_CLASS"][trial], \
                                                      zpe__["Z_SPE"]["SAMPLED_CLASS"][trial])
                else:
                    mf_rel, mb_rel, p_mb = arb.add_pe(rpe, spe)
                t_p_mb   += p_mb
                t_mf_rel += mf_rel
                t_mb_rel += mb_rel
                t_rpe    += abs(rpe)
                t_spe    += spe
                t_score  += human_reward # if not the terminal state, human_reward is 0, so simply add here is fine

                """iterators update"""
                # if MODEL_LIST_IDF == 3:
                #     human_obs = next_human_obs
                #     human_obs_ = next_human_obs_
                # else:
                human_obs = next_human_obs

                # trial_index += 1


            # calculation after one trial
            p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
            t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])) # map to average value

            cum_p_mb   += p_mb
            cum_mf_rel += mf_rel

            cum_mb_rel += mb_rel
            cum_rpe    += rpe
            cum_spe    += spe
            cum_score  += t_score

            # update trial data
            if env.GRW != 0:
                BHV_trial.append([*BHV_trial_temp, *likelihood_, *GRW_hist])
            else:
                BHV_trial.append([*BHV_trial_temp, *likelihood_])

            # """update control agent"""
            # if MIXED_RANDOM_MODE:
            #     t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), CONTROL_MODE_temp)
            # else:
            #     t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), CONTROL_MODE)
            #
            cum_reward += t_reward
            next_control_obs = np.append(next_control_obs_frag, [rpe, spe])
            # if not MIXED_RANDOM_MODE: # if it is mixed random mode, don't train ddpn anymore. Using ddqn that is already trained before
            #     ddqn.optimize(control_obs, control_action, next_control_obs, t_reward)
            control_obs_extra = [rpe, spe]
            # detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, t_reward, t_score] + [control_action]

            # Cuz we dont have any control_action, just made this part comment
            detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, t_reward, t_score] + [0]
            if not return_res:
                gData.add_detail_res(trial + TRIALS_PER_EPISODE_sbjtv * episode, detail_col)

            if env.GRW != 0:
                gData.add_behavior_simul(trial, [*BHV_trial_temp, *likelihood_, *GRW_hist])
            else:
                gData.add_behavior_simul(trial, [*BHV_trial_temp, *likelihood_])

            if BHV_trial_temp[7] == -999:
                print('bwd update')
            # else:
            #     res_detail_df.loc[trial + TRIALS_PER_EPISODE * episode] = detail_col

            if episode==TOTAL_EPISODES-1:#SAVE_LOG_Q_VALUE:
                Q_value_forward = []
                Q_value_sarsa = []
                Q_value_arb = []
                Transition = []
                #print("#############Task_structure##############")
                #print(np.concatenate((env.reward_map, env.trans_prob,env.output_states), axis=None))
                #print("#############forward_Q##############")
                for state in range(5):
                    #print(state ,forward.get_Q_values(state))
                    Q_value_forward += list(forward.get_Q_values(state))
                #print("#############sarsa_Q##############")
                for state in range(5):
                    #print(state ,sarsa.get_Q_values(state))  
                    Q_value_sarsa += list(sarsa.get_Q_values(state))
                #print("#############arb_Q##############")
                for state in range(5):
                    #print(state ,arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state))) 
                    Q_value_arb += list(arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state))) 
                #print("#############Transition##############")
                for state in range(5):
                    for action in range(2):
                        #print(state , action, forward.get_Transition(state, action), sum(forward.get_Transition(state, action)))
                        Transition += list(forward.get_Transition(state, action))
                
                Q_value_forward_t.append(Q_value_forward)
                Q_value_sarsa_t.append(Q_value_sarsa)
                Q_value_arb_t.append(Q_value_arb)
                Transition_t.append(Transition)

                #print(human_action_list_t, human_action_list_t.shape)
                #print(Task_structure, Task_structure.shape)
                '''print(Transition, len(Transition))
                print(Q_value_forward, len(Q_value_forward))
                print(Q_value_sarsa, len(Q_value_sarsa))
                print(Q_value_arb, len(Q_value_arb))'''
            bwd_idf_prev = bwd_idf

                
        data_col = list(map(lambda x: x / TRIALS_PER_EPISODE, 
                            [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_reward, cum_score] + list(cum_ctrl_act)))
        if not return_res:
            gData.add_res(episode, data_col)
        else:
            res_data_df.loc[episode] = data_col
            if episode==TOTAL_EPISODES-1:
                res_Q_df.loc[0] = [Q_value_forward, Q_value_sarsa, Q_value_arb, Transition]
        if SAVE_LOG_Q_VALUE:
            human_action_list_t.append(human_action_list_episode)

        gData.complete_simulation()
    # if continue_steps:
    #     for episode in tqdm(range(TOTAL_EPISODES)):
    #         cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = 0
    #         cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)
    #         human_action_list_episode = []
    #         for trial in range(TRIALS_PER_EPISODE):
    #             t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = 0
    #             game_terminate              = False
    #             human_obs, control_obs_frag = env.reset()
    #             control_obs                 = np.append(control_obs_frag, control_obs_extra)
    #
    #             """control agent choose action"""
    #             if STATIC_CONTROL_AGENT:
    #                 control_action = static_action_map2[CONTROL_MODE][trial]
    #             else:
    #                 control_action = ddqn.action(control_obs)
    #             cum_ctrl_act[control_action] += 1
    #
    #             if control_action == 3:
    #                 if env.is_flexible == 1:
    #                     arb.p_mb = 0.8
    #                     arb.p_mf = 0.2
    #                 else:
    #                     arb.p_mb = 0.2
    #                     arb.p_mf = 0.8
    #
    #             """control act on environment"""
    #             if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
    #                 _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])
    #             if SAVE_LOG_Q_VALUE:
    #                 Task_structure.append( np.concatenate((env.reward_map, env.trans_prob, env.output_states), axis=None) )
    #
    #
    #             while not game_terminate:
    #                 """human choose action"""
    #                 human_action = compute_human_action(arb, human_obs, sarsa, forward)
    # #                if episode < TOTAL_EPISODES:
    # #                    human_action = random.randint(0,1)
    #                 if start_con:
    #                     if env.is_flexible != con2[0]:
    #                         env._specific_flexible_switch()
    #                     if env.trans_prob[0]!= con2[1]:
    #                         env.trans_prob[0] = con2[1]
    #                         env.trans_prob[1] = 1-con2[1]
    #
    #                 #print("human action : ", human_action)
    #                 if SAVE_LOG_Q_VALUE:
    #                     human_action_list_episode.append(human_action)
    #
    #                 """human act on environment"""
    #                 next_human_obs, human_reward, game_terminate, next_control_obs_frag \
    #                     = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
    #
    #                 """update human agent"""
    #                 spe = forward.optimize(human_obs, human_action, next_human_obs)
    #                 next_human_action = compute_human_action(arb, next_human_obs, sarsa, forward) # required by models like SARSA
    #                 if env.is_flexible == 1: #flexible goal condition
    #                     rpe = sarsa.optimize(human_reward, human_action, next_human_action, human_obs, next_human_obs)
    #                 else: # specific goal condition human_reward should be normalized to sarsa
    #                     if human_reward > 0: # if reward is 10, 20, 40
    #                         rpe = sarsa.optimize(40, human_action, next_human_action, human_obs, next_human_obs)
    #                     else:
    #                         rpe = sarsa.optimize(0, human_action, next_human_action, human_obs, next_human_obs)
    #
    #                 mf_rel, mb_rel, p_mb = arb.add_pe(rpe, spe)
    #                 t_p_mb   += p_mb
    #                 t_mf_rel += mf_rel
    #                 t_mb_rel += mb_rel
    #                 t_rpe    += abs(rpe)
    #                 t_spe    += spe
    #                 t_score  += human_reward # if not the terminal state, human_reward is 0, so simply add here is fine
    #
    #                 """iterators update"""
    #                 human_obs = next_human_obs
    #
    #             # calculation after one trial
    #             p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
    #             t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])) # map to average value
    #
    #             cum_p_mb   += p_mb
    #             cum_mf_rel += mf_rel
    #
    #             cum_mb_rel += mb_rel
    #             cum_rpe    += rpe
    #             cum_spe    += spe
    #             cum_score  += t_score
    #
    #             """update control agent"""
    #             if MIXED_RANDOM_MODE:
    #                 t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), CONTROL_MODE_temp)
    #             else:
    #                 t_reward = error_to_reward((rpe, spe, mf_rel, mb_rel), CONTROL_MODE)
    #             cum_reward += t_reward
    #             next_control_obs = np.append(next_control_obs_frag, [rpe, spe])
    #             if not MIXED_RANDOM_MODE: # if it is mixed random mode, don't train ddpn anymore. Using ddqn that is already trained before
    #                 ddqn.optimize(control_obs, control_action, next_control_obs, t_reward)
    #             control_obs_extra = [rpe, spe]
    #             detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, t_reward, t_score] + [control_action]
    #             if not return_res:
    #                 gData.add_detail_res(trial + TRIALS_PER_EPISODE * episode, detail_col)
    #             else:
    #                 res_detail_df.loc[trial + TRIALS_PER_EPISODE * episode] = detail_col
    #             if episode==TOTAL_EPISODES-1:#SAVE_LOG_Q_VALUE:
    #                 Q_value_forward = []
    #                 Q_value_sarsa = []
    #                 Q_value_arb = []
    #                 Transition = []
    #                 #print("#############Task_structure##############")
    #                 #print(np.concatenate((env.reward_map, env.trans_prob,env.output_states), axis=None))
    #                 #print("#############forward_Q##############")
    #                 for state in range(5):
    #                     #print(state ,forward.get_Q_values(state))
    #                     Q_value_forward += list(forward.get_Q_values(state))
    #                 #print("#############sarsa_Q##############")
    #                 for state in range(5):
    #                     #print(state ,sarsa.get_Q_values(state))
    #                     Q_value_sarsa += list(sarsa.get_Q_values(state))
    #                 #print("#############arb_Q##############")
    #                 for state in range(5):
    #                     #print(state ,arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state)))
    #                     Q_value_arb += list(arb.get_Q_values(sarsa.get_Q_values(state),forward.get_Q_values(state)))
    #                 #print("#############Transition##############")
    #                 for state in range(5):
    #                     for action in range(2):
    #                         #print(state , action, forward.get_Transition(state, action), sum(forward.get_Transition(state, action)))
    #                         Transition += list(forward.get_Transition(state, action))
    #
    #                 Q_value_forward_t.append(Q_value_forward)
    #                 Q_value_sarsa_t.append(Q_value_sarsa)
    #                 Q_value_arb_t.append(Q_value_arb)
    #                 Transition_t.append(Transition)
    #
    #                 #print(human_action_list_t, human_action_list_t.shape)
    #                 #print(Task_structure, Task_structure.shape)
    #                 '''print(Transition, len(Transition))
    #                 print(Q_value_forward, len(Q_value_forward))
    #                 print(Q_value_sarsa, len(Q_value_sarsa))
    #                 print(Q_value_arb, len(Q_value_arb))'''
    #
    #         data_col = list(map(lambda x: x / TRIALS_PER_EPISODE,
    #                             [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_reward, cum_score] + list(cum_ctrl_act)))
    #         if not return_res:
    #             gData.add_res(episode, data_col)
    #         else:
    #             res_data_df.loc[episode] = data_col
    #             if episode==TOTAL_EPISODES-1:
    #                 res_Q_df.loc[0] = [Q_value_forward, Q_value_sarsa, Q_value_arb, Transition]
    #         if SAVE_LOG_Q_VALUE:
    #             human_action_list_t.append(human_action_list_episode)
    if SAVE_LOG_Q_VALUE: 
        #human_action_list_t = np.array(human_action_list_t, dtype=np.int32)
        Task_structure = np.array(Task_structure)
        Transition_t = np.array(Transition_t)
        #Q_value_forward_t = np.array(Q_value_forward_t) 
        #Q_value_sarsa_t = np.array(Q_value_sarsa_t)
        #Q_value_arb_t = np.array(Q_value_arb_t)
        '''print(human_action_list_t, human_action_list_t.shape)
        print(Task_structure, Task_structure.shape)
        print(Transition_t, Transition_t.shape)
        print(Q_value_forward_t, Q_value_forward_t.shape )
        print(Q_value_sarsa_t, Q_value_sarsa_t.shape)
        print(Q_value_arb_t, Q_value_arb_t.shape)'''

        gData.add_log_Q_value(human_action_list_t, Task_structure, Transition_t, Q_value_forward_t, Q_value_sarsa_t, Q_value_arb_t )

    # if MIXED_RANDOM_MODE:
    #     #print (list(random_mode_list))
    #     gData.add_random_mode_sequence(random_mode_list.tolist())

    # if SAVE_CTRL_RL and not MIXED_RANDOM_MODE:
    #     makedir(RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE)
    #     #torch.save(ddqn.eval_Q.state_dict(), RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save model as dictionary
    #     torch.save(ddqn.eval_Q, RESULTS_FOLDER + 'ControlRL/' + CONTROL_MODE + '/MLP_OBJ_Subject' + PARAMETER_SET) # save entire model
    # if ENABLE_PLOT:
    #     gData.plot_all_human_param(CONTROL_MODE + ' Human Agent State - parameter set: ' + PARAMETER_SET)
    #     gData.plot_pe(CONTROL_MODE, CONTROL_MODE + ' - parameter set: ' + PARAMETER_SET)
    #     gData.plot_action_effect(CONTROL_MODE, CONTROL_MODE + ' Action Summary - parameter set: ' + PARAMETER_SET)
    if return_res:
        return (res_data_df, res_detail_df, res_Q_df)


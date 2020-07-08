""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import torch
import numpy as np
import pandas as pd
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import random
import torch.nn as nn
import torch.nn.functional as F
import gym

from collections import namedtuple
from torch.autograd import Variable
from gym import spaces
from common import MLP, Memory
from functools import reduce

from gym import spaces
from tqdm import tqdm
from mdp_task3 import MDP
from ddqn import DoubleDQN
from sarsa import SARSA
from forward import FORWARD
from arbitrator import BayesRelEstimator, AssocRelEstimator, Arbitrator, BayesRelEstimator_zpe, Arbitrator_zpe
from analysis import gData, RESULTS_FOLDER, COLUMNS, DETAIL_COLUMNS, Q_COLUMNS
from common import makedir
import zero_pe
import time


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

RESET = False
SAVE_LOG_Q_VALUE = False
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
ori_human_action = []

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
               return_res=False, start_con=False, con=None, continue_gData=False, transferred_gData=None, continue_steps=False, con2=None, behave_=[], bhv_pseudo=True,
               TRIALS_PER_EPISODE_sbjtv = 0, NLL_sbj = -1, no_fitting = False, player_MDP = None, bhv_copy = False ):
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

    control_obs_extra = INIT_CTRL_INPUT

    if env.GRW != 0:
        GRW_length = 1 + len([*np.array(env.GRW_list)[:, 0].tolist()])

    if no_fitting == False:
        if bhv_pseudo == False:
            if env.GRW != 0:
                gData.behavior_simul = np.zeros((TOTAL_EPISODES, TRIALS_PER_EPISODE_sbjtv, 18 + 4 + GRW_length + 1 + 1))
            else:
                gData.behavior_simul = np.zeros((TOTAL_EPISODES, TRIALS_PER_EPISODE_sbjtv, 18 + 4 + 1 + 1))
        else:
            if env.GRW != 0:
                gData.behavior_simul = np.zeros((TOTAL_EPISODES, TRIALS_PER_EPISODE_sbjtv, 18 + 4 + GRW_length + 1))
            else:
                gData.behavior_simul = np.zeros((TOTAL_EPISODES, TRIALS_PER_EPISODE_sbjtv, 18 + 4 + 1))
    else:
        if env.GRW != 0:
            gData.behavior_simul = np.zeros((TOTAL_EPISODES, TRIALS_PER_EPISODE_sbjtv, 18 + 4 + GRW_length))
        else:
            gData.behavior_simul = np.zeros((TOTAL_EPISODES, TRIALS_PER_EPISODE_sbjtv, 18 + 4))

    gData.sim_simul = np.zeros((TOTAL_EPISODES, TRIALS_PER_EPISODE_sbjtv))


    human_action_list_t = []
    Task_structure = []
    Q_value_forward_t = []
    Q_value_sarsa_t = []
    Q_value_arb_t = []
    Transition_t = []

    # ADDED
    BHV_sim = []
    sim_sim = []
    BEHAVIOR_SIMUL_COLUMN_ORI = gData.BEHAVIOR_SIMUL_COLUMN

    gData.new_simulation()
    if not RESET:
        print(env.num_states+4)
        player_agent = DoubleDQN(spaces.Discrete(env.num_states+4), env.action_space[MDP.HUMAN_AGENT_INDEX], torch.cuda.is_available())  # use DDQN for player agent
        if no_fitting :
            player_agent.eval_Q = player_MDP[0]
            player_agent.target_Q = player_MDP[0]
        # register in the communication controller
        forward = FORWARD(env.observation_space[MDP.HUMAN_AGENT_INDEX],
                          env.action_space[MDP.HUMAN_AGENT_INDEX],
                          env.state_reward_func, env.output_states_offset, env.reward_map_func, env.output_states,
                          learning_rate=rl_learning_rate,
                          disable_cforward=DISABLE_C_EXTENSION)  # forward model-based learner

    for episode in tqdm(range(TOTAL_EPISODES)):
        p_mb = mf_rel = mb_rel = 0
        gData.new_simulation()
        gData.add_human_data([amp_mf_to_mb / amp_mb_to_mf, rl_learning_rate, estimator_learning_rate, threshold, temperature,performance])
        control_obs_extra = INIT_CTRL_INPUT


        human_obs, control_obs_frag = env.reset()
        obs = np.zeros(len(env.output_states))
        obs[human_obs] = 1

        cum_p_mb = cum_mf_rel = cum_mb_rel = cum_rpe = cum_spe = cum_reward = cum_score = 0
        cum_ctrl_act = np.zeros(MDP.NUM_CONTROL_ACTION)


        human_action_list_episode = []
        bwd_idf = -1
        bwd_idf_prev = -1
        if bhv_pseudo==True and no_fitting==False :
            TRIALS_PER_EPISODE_sbjtv = 100
            pseudo_goals = np.zeros(100)
            pseudo_prob = np.zeros(100)
            for ii in range(25):
                pseudo_goals[4*ii-4]=int(-1)
                pseudo_goals[4*ii-3]=int(6)
                pseudo_goals[4*ii-2]=int(7)
                pseudo_goals[4*ii-1]=int(8)
                pseudo_prob[4*ii-4]=int(0)
                pseudo_prob[4*ii-3]=int(0)
                pseudo_prob[4*ii-2]=int(1)
                pseudo_prob[4*ii-1]=int(1)
            pseudo_goals = np.random.permutation(pseudo_goals)
            pseudo_prob = np.random.permutation(pseudo_prob)

        if env.GRW != 0:
            BHV_trial = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18 + GRW_length + 4 +1))
        else:
            BHV_trial = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18 + 4 +1))
        sim_trial = np.zeros((TRIALS_PER_EPISODE_sbjtv))
        if no_fitting == False:
            if bhv_pseudo == False:
                if env.GRW != 0:
                    gData.behavior_simul_df = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18+4+GRW_length+1+1))
                else:
                    gData.behavior_simul_df = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18 + 4 + 1 + 1))
            else:
                if env.GRW != 0:
                    gData.behavior_simul_df = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18 + 4 + GRW_length + 1))
                else:
                    gData.behavior_simul_df = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18 + 4 + 1))
        else:
            if env.GRW != 0:
                gData.behavior_simul_df = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18 + 4 + GRW_length))
            else:
                gData.behavior_simul_df = np.zeros((TRIALS_PER_EPISODE_sbjtv, 18 + 4))

        gData.sim_simul_df = np.zeros((TRIALS_PER_EPISODE_sbjtv))

        cum_likelihood = 0
        cum_ddqn_reward = 0
        for trial in range(TRIALS_PER_EPISODE_sbjtv):
#           BHV_trial_temp = [0:episode,1:trial,2:block condition(SF HL),3:obs1,4:obs2,5:obs3,6:action1,7:action2,8,,,,,,14 : RTs and onsets,15:reward,16:??,17:bwd_idf / goal_type]
            BHV_trial_temp = [episode + 1, trial + 1, CTXT_sbjtv[trial][0], 0, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, CTXT_sbjtv[trial][1]]
            if bhv_pseudo == True and no_fitting == False:
                BHV_trial_temp[-1]=pseudo_goals[trial]

#            BHV_trial_temp = [episode+1, trial+1, CTXT_sbjtv[trial][0], 0, -999 ,-999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, CTXT_sbjtv[trial][1]]
#            BHV_trial_temp = [episode + 1, trial + 1, CTXT_sbjtv[trial][0], 0, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -999, -1]
            sim_trial_temp = [-999]
                                # -999 indicates there is no valid data currently.
                                # some of them sould be filled in the followed section (while game_terminate)

            if env.GRW != 0:
                GRW_hist = np.zeros(GRW_length)
            else:
                GRW_hist = []


            bwd_idf = BHV_trial_temp[-1]
            t_p_mb = t_mf_rel = t_mb_rel = t_rpe = t_spe = t_reward = t_score = 0
            human_obs, control_obs_frag = env.reset()
            game_terminate              = False

            """control agent choose action"""
            if env.GRW == 0:
                if bwd_idf == -1:
                    env.is_flexible=1
                else:
                    env.is_flexible=0

            elif env.GRW == 1: # only for the pay-off
                env.is_flexible=2 # GRW
            elif env.GRW == 2: # for both
                env.is_flexible=2 # GRW
            elif env.GRW == 3: # for both
                env.is_flexible=2 # GRW
            elif env.GRW == 4:  # for both
                env.is_flexible = 2  # GRW

            if bhv_pseudo == True and no_fitting == False :
                if pseudo_prob[trial] == 0 :
                    if pseudo_goals[trial] == -1 :
                        # 1: specific low 4: flexible low 2: specific high 3: flexible high
                        env.ctxt = 4
                    else:
                        env.ctxt = 1
                else :
                    if pseudo_goals[trial] == -1 :
                        env.ctxt = 3
                    else:
                        env.ctxt = 2
            else :
                env.ctxt = CTXT_sbjtv[trial][0]
            if env.GRW == 0:
                if bwd_idf != bwd_idf_prev:
                        forward.bwd_update(bwd_idf,env)
            env._update_trans_prob()

            # """control act on environment"""
            # if CTRL_AGENTS_ENABLED:  # and (trial % ACTION_PERIOD is 0 or not STATIC_CONTROL_AGENT): ## why trial % ACTION_PERIOD ??
            #     _, _, _, _ = env.step([MDP.CONTROL_AGENT_INDEX, control_action])

            trial_index = 0
            likelihood_=np.zeros(4)
            obs_len=player_agent._linear_size(env.observation_space[MDP.HUMAN_AGENT_INDEX])
            if no_fitting:
                obs_len = player_agent.eval_Q.fc1.in_features-4
            prev_ddqn_reward = 1
            for trial_index in range(2):
                ddqn_reward = 0
                obs = np.zeros(obs_len+4)
                obs[human_obs] = 1
                if BHV_trial_temp[-1] == -1:
                    obs[-4] = 1
                    env.reward_map = [0,10,20,40]
                elif BHV_trial_temp[-1] == 6:
                    obs[-3] = 1
                    env.reward_map = [0,10,0,0]
                elif BHV_trial_temp[-1] == 7:
                    obs[-2] = 1
                    env.reward_map = [0,0,20,0]
                elif BHV_trial_temp[-1] == 8:
                    obs[-1] = 1
                    env.reward_map = [0,0,0,40]
                else :
                    print('bang')
            # while not game_terminate:
                if trial_index ==0:
                    BHV_trial_temp[3] = human_obs
                """human choose action"""
                human_action = player_agent.action(obs)
                if bhv_copy == True:
                    human_action = ori_human_action[trial][trial_index]-1
                likelihood_[2*trial_index:2*trial_index+2] = player_agent.get_ddqn_likelihood(obs)
                if bhv_pseudo == False or no_fitting == True:
                    cum_likelihood += -1*np.log(player_agent.get_ddqn_likelihood(obs)[ori_human_action[trial][trial_index]-1])
                ## Unselected likelihood
                BHV_trial_temp[6+trial_index] = human_action

                if BHV_trial_temp[6+trial_index] == -999:
                    print('bwd update')
                #print("human action : ", human_action)


                """human act on environment"""

                next_human_obs, human_reward, game_terminate, next_control_obs_frag \
                    = env.step((MDP.HUMAN_AGENT_INDEX, human_action))
                next_obs = np.zeros(obs_len+4)
                next_obs[next_human_obs] = 1
                if BHV_trial_temp[-1] == -1:
                    next_obs[-4] = 1
                elif BHV_trial_temp[-1] == 6:
                    next_obs[-3] = 1
                elif BHV_trial_temp[-1] == 7:
                    next_obs[-2] = 1
                elif BHV_trial_temp[-1] == 8:
                    next_obs[-1] = 1
                BHV_trial_temp[15] = human_reward
                if env.GRW != 0 and trial_index == 0:
                    GRW_hist[0] = env.GRW_list[trial_index][0]
                    #GRW_hist.append(env.GRW_list[trial_index][0])
                elif env.GRW != 0:
                    GRW_hist[1:] = np.array(env.GRW_list)[:,0].tolist()
                    #GRW_hist = [*GRW_hist, *np.array(env.GRW_list)[:, 0].tolist()]
                BHV_trial_temp[4+trial_index] = next_human_obs

                if bhv_pseudo :
                    ddqn_reward = human_reward
                else :
                    if prev_ddqn_reward :
                        ddqn_reward = int(BHV_trial_temp[6+trial_index] == ori_human_action[trial][trial_index]-1)
                        cum_ddqn_reward += ddqn_reward
                        prev_ddqn_reward = ddqn_reward
                """RPE acqusition """
                #obs = Variable(player_agent.Tensor(obs))
                #next_obs = Variable(player_agent.Tensor(next_obs))
                #if trial_index == 0:
                #    rpe = max(player_agent.eval_Q(next_obs))-player_agent.eval_Q(obs)[human_action]
                #else :
                #    rpe = human_reward-player_agent.eval_Q(obs)[human_action]
                rpe = spe = -1
                """update human agent"""

#                spe = forward.optimize(human_obs, human_action, next_human_obs)
                if no_fitting == False :
#                    rpe = player_agent.get_loss(obs, human_action, next_obs, ddqn_reward)
                    player_agent.optimize(obs, human_action, next_obs, ddqn_reward)
                if rpe == None :
                    rpe = -1
                """Human next agent action"""
                next_human_action = player_agent.action(next_obs)
                if trial_index == 1:
                    BHV_trial_temp[15] = human_reward
                    #if env.is_flexible == 1:
                        #BHV_trial_temp[15] = human_reward
                    #else:
                        #BHV_trial_temp[15] = forward.output_states[next_human_obs-5]
                        #BHV_trial_temp[15] = env.output_states[next_human_obs-5]


                t_rpe    += abs(rpe)
                t_spe    += spe
                t_score  += human_reward # if not the terminal state, human_reward is 0, so simply add here is fine
                human_obs = next_human_obs # check this

            # calculation after one trial
            p_mb, mf_rel, mb_rel, rpe, spe = list(map(lambda x: x / MDP_STAGES, [
            t_p_mb, t_mf_rel, t_mb_rel, t_rpe, t_spe])) # map to average value

            cum_p_mb   += p_mb
            cum_mf_rel += mf_rel

            cum_mb_rel += mb_rel
            cum_rpe    += rpe
            cum_spe    += spe
            cum_score  += t_score

            sim_trial_temp[-1] = rpe

            # update trial data
            if env.GRW != 0:
                BHV_trial[trial]=[*BHV_trial_temp, *likelihood_, *GRW_hist,cum_likelihood]
            else:
                BHV_trial[trial]=[*BHV_trial_temp, *likelihood_,cum_likelihood]


            sim_trial[trial]= sim_trial_temp[-1]

            cum_reward += t_reward


            control_obs_extra = [rpe, spe]

            # Cuz we dont have any control_action, just made this part comment


            detail_col = [rpe, spe, mf_rel, mb_rel, p_mb, t_reward, t_score] + [0]
            if not return_res:
                gData.add_detail_res(trial + TRIALS_PER_EPISODE_sbjtv * episode, detail_col)

            if no_fitting == False :
                if bhv_pseudo == False :
                    if env.GRW != 0:
                        gData.behavior_simul_df[trial] = [*BHV_trial_temp, *likelihood_, *GRW_hist, cum_ddqn_reward, cum_likelihood]
                    else:
                        gData.behavior_simul_df[trial] = [*BHV_trial_temp, *likelihood_, cum_ddqn_reward, cum_likelihood]
                else:
                    if env.GRW != 0:
                        gData.behavior_simul_df[trial] = [*BHV_trial_temp, *likelihood_, *GRW_hist, ddqn_reward]
                    else:
                        gData.behavior_simul_df[trial] = [*BHV_trial_temp, *likelihood_, ddqn_reward]
            else :
                if env.GRW != 0:
                    gData.behavior_simul_df[trial] = [*BHV_trial_temp, *likelihood_,*GRW_hist]
                else:
                    gData.behavior_simul_df[trial] = [*BHV_trial_temp, *likelihood_]

            gData.sim_simul_df[trial]=sim_trial_temp[-1]

            # else:
            #     res_detail_df.loc[trial + TRIALS_PER_EPISODE * episode] = detail_col

        if no_fitting == False :
            if bhv_pseudo == False :
                if cum_likelihood < NLL_sbj and episode > 100 :
                    print(episode)
                    '''add NN'''
                    gData.add_NN(player_agent.eval_Q)
                    gData.NN.append(gData.NN_df)
                    gData.behavior_simul[episode] = gData.behavior_simul_df
                    gData.sim_simul[episode] = gData.sim_simul_df
                    gData.behavior_simul = gData.behavior_simul[1:episode+1]
                    gData.sim_simul= gData.sim_simul_df[1:episode+1]
                    break
            if episode % int(TOTAL_EPISODES/10) == 0:
                print(episode)
                '''add NN'''
                gData.add_NN(player_agent.eval_Q)
                gData.NN.append(gData.NN_df)


        data_col = list(map(lambda x: x / TRIALS_PER_EPISODE, 
                            [cum_rpe, cum_spe, cum_mf_rel, cum_mb_rel, cum_p_mb, cum_reward, cum_score] + list(cum_ctrl_act)))
        if not return_res:
            gData.add_res(episode,data_col)
        else:
            res_data_df.loc[episode] = data_col
            if episode==TOTAL_EPISODES-1:
                res_Q_df.loc[0] = [Q_value_forward, Q_value_sarsa, Q_value_arb, Transition]


        #gData.complete_simulation()
        gData.behavior_simul[episode] = gData.behavior_simul_df
        gData.sim_simul[episode] = gData.sim_simul_df

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

#        gData.add_log_Q_value(human_action_list_t, Task_structure, Transition_t, Q_value_forward_t, Q_value_sarsa_t, Q_value_arb_t )

    if return_res:
        return (res_data_df, res_detail_df, res_Q_df)


import getopt
import sys
import csv
import os
# import simulation as sim
import simulation_gen_task5 as sim
import analysis
import dill as pickle # see https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
import torch
import numpy as np

from analysis import gData, MODE_MAP
from tqdm import tqdm
from numpy.random import choice
from training_layer2 import Net
from torch.autograd import Variable

usage_str =  """
Model-free, model-based learning simulation

Usage:

Simulation control parameters:

    -d load parameters from csv file, default is regdata.csv

    -n [number of parameters entries to simulate]

    --episodes [num episodes]

    --trials [num trials per episodes]

    --set-param-file [parameter file]             Specify the parameter csv file to load

    --mdp-stages [MDP environment stages]         Specify how many stages in MDP environment

    --ctrl-mode <min-spe/max-spe/min-rpe/max-rpe/min-mf-rel/max-mf-rel/min-mb-rel/max-mb-rel> 
                                                  Choose control agent mode

    --legacy-mode                                 Use legacy MDP environment, which treats one type of terminal reward as one
                                                  state. Since C++ ext is not implemented for this type of environment, legacy
                                                  pure Python implemenation for FORWARD will be use.

    --disable-control                             Disable control agents

    --all-mode                                    Execute all control mode

    --enable-static-control                       Use static control instead of DDQN control

    --disable-detail-plot                         Disable plot for each simulation

    --disable-c-ext                               Disable using C extension

    --less-control-input                          Less environment input for control agent

    --save-ctrl-rl                                Save control RL agent object for further use

    --no-reset                                    Not reset environment and agent when start every episode. Default is reset

    --save-log-Q-value                            Save action list, task structure, state transition(Forward), Q value(Forward), Q value(Sarsa), Q value(arbitrator) in gData as a pickle file.
                                                  Default is not saving. 
                                                  Only possible in '--all-mode -d'

Analysis control parameters:

    --re-analysis [analysis object pickle file]   Re-run analysis functions

    --PCA-plot                                    Generate plot against PCA results. Not set by default because
                                                  previous PCA run shows MB preference gives 99% variance, so comparing 
                                                  against MB preference is good enough, instead of some principal component

    --learning-curve-plot                         Plot learning curves

    --use-confidence-interval                     When plot with error bar, use confidence interval instead of IQR

    --separate-learning-curve                     Separate learning curve plot

    --disable-auto-max                            Use fix max value on y axis when plotting learning curve in one episode     

    --to-excel [subject id]                       Generate a excel file for specific subject with detail sequence of data

    --disable-action-compare                      Use actions as feature space

    --enable-score-compare                        Use score as feature space

    --human-data-compare                          Enable comparison against the full columns of human data

    --use-selected-subjects                       Use selected subjects, defualt is min 25 50 75 max five subjects

    --head-tail-subjects                          Use head and tail subjects to emphasize the difference

    --cross-mode-plot                             Plots that compare data between modes

    --enhance-compare <boost/inhibit/cor/sep>     Only plot two modes depending on the scenario to comapre

    --cross-compare [mode]                        Extract best action sequence from subject A in a given mode. Apply to subject B.
                                                  Plot against subject B's original data

    --opposite-cross-compare [mode]               Extract worst action sequence(opposite mode optimal sequence) from specific subject in a given mode. 
                                                  Apply to that subject. Plt against the subject's original data

    --sub-A [subject A]                           Subject A for cross compare // useless for current version code

    --sub-B [subject B]                           Subject B for cross compare // useless for current version code

    --to-excel-log-Q-value                        Generate a excel file for action list, task structure, state transition(Forward), Q value(Forward), Q value(Sarsa), Q value(arbitrator) in pickle file
                                                  To use this option, '--save-log-Q-value' option should be used before 

    --to-excel-random-mode-sequence               Generate a excel file of random mode sequence after executing mixed-random mode


    --to-excel-optimal-sequence

"""


def usage():
    print(usage_str)

LOAD_PARAM_FILE   = True
NUM_PARAMETER_SET = 82
ALL_MODE          = False
TO_EXCEL          = None
SCENARIO          = None
CROSS_MODE_PLOT   = False
CROSS_COMPARE     = False
OPPOSITE_CROSS_COMPARE = False
OPPOSITE_NN_CROSS_COMPARE = False
FAIR_OPPOSITE_NN_CROSS_COMPARE = False
CROSS_COMPARE_MOD = 'min-spe'
SUBJECT_A         = 10 # Low MB->MF trans rate
SUBJECT_B         = 17 # High MB->MF trans rate
PARAMETER_FILE    = 'regdata.csv'
TO_EXCEL_LOG_Q_VALUE = False
TO_EXCEL_RANDOM_MODE_SEQUENCE = False
TO_EXCEL_OPTIMAL_SEQUENCE = False

SCENARIO_MODE_MAP = {
    'boost'   : ['min-spe', 'min-rpe'],
    'inhibit' : ['min-spe', 'max-spe'],
    'cor'     : ['min-rpe-min-spe', 'max-rpe-max-spe'],
    'sep'     : ['min-rpe-max-spe', 'max-rpe-min-spe']
}
ORIGINAL_MODE_MAP = ['min-spe', 'max-spe', 'min-rpe', 'max-rpe', 'min-rpe-min-spe', 'max-rpe-max-spe', 'max-rpe-min-spe', 'min-rpe-max-spe']
OPPOSITE_MODE_MAP = ['max-spe', 'min-spe', 'max-rpe', 'min-rpe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'min-rpe-max-spe', 'max-rpe-min-spe']

MODE_LIST = ['min-rpe', 'max-rpe', 'min-spe', 'max-spe', 'max-rpe-max-spe', 'min-rpe-min-spe', 'max-rpe-min-spe', 'min-rpe-max-spe']
MODE_LIST_INPUT = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [-1, -1], [1, -1], [-1, 1]]

# ANALYSIS_OBJ      = None # ERASEDZ
# ADDED
BEHAV_DIR = 'bhv_results/'
SIMUL_BEHAV_DIR = 'simul_results/'

MODEL_LIST_IDF = 0 # 0 : MB ONLY, 1 : MF ONLY, 2 : LEE2014, 3 : ZPE MODEL.
if MODEL_LIST_IDF == 0:
    PARAMETER_FILE='regdata_mb.csv'
elif MODEL_LIST_IDF == 1:
    PARAMETER_FILE='regdata_mf.csv'
elif MODEL_LIST_IDF == 2:
    PARAMETER_FILE='regdata_lee2014.csv'
elif MODEL_LIST_IDF == 3:
    PARAMETER_FILE='regdata_zpe.csv'
else:
    PARAMETER_FILE = PARAMETER_FILE

TASK_IDF=7
if TASK_IDF == 1:
    SIMUL_BEHAV_DIR = 'simul_results/' + 'T1/'
elif TASK_IDF == 2:
    SIMUL_BEHAV_DIR = 'simul_results/' + 'T2/'
elif TASK_IDF == 3:
    SIMUL_BEHAV_DIR = 'simul_results/' + 'T3/'
elif TASK_IDF == 4:
    SIMUL_BEHAV_DIR = 'simul_results/' + 'T4/'
elif TASK_IDF == 5:
    SIMUL_BEHAV_DIR = 'simul_results/' + 'T5/'
elif TASK_IDF == 6:
    SIMUL_BEHAV_DIR = 'simul_results/' + 'T6/'
elif TASK_IDF == 7:
    SIMUL_BEHAV_DIR = 'simul_results/' + 'T7/'
else :
    SIMUL_BEHAV_DIR = 'simul_results/'
# toggle of task idf
# 0. ladder and tree
ladder_tree_ = 1
# 1. GRW payoff or not
GRW_ = 3#1: GRW payoff 0 for legacy mode 2: for every state transition
# 2. fixed trans_prob [0.7 0.3] or not fixed (legacy)
fixed_trans_ = 0 # 1:  0.7 and 0.3, 0 for legacy mode
# 3. num_action applicable
num_action_ = 2


# PARAMETER_FILE = 'regdata_zpe.csv'



def reanalysis(analysis_object):
    with open(analysis_object, 'rb') as pkl_file:
        gData = pickle.load(pkl_file)
    with open(PARAMETER_FILE) as f:
        csv_parser = csv.reader(f)
        param_list = []
        for row in csv_parser:
            param_list.append(tuple(map(float, row[:-1])))
    if CROSS_MODE_PLOT:
        if SCENARIO is not None:
            gData.cross_mode_summary(SCENARIO_MODE_MAP[SCENARIO])
        else:
            gData.cross_mode_summary()
    elif CROSS_COMPARE:
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = [mode for mode, _ in MODE_MAP.items()]
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        for compare_mode in mode_iter_lst:
            SUBJECT_A = 81  # default change this to policy number you want to apply
            for SUBJECT_B in range(NUM_PARAMETER_SET):
                # SUBJECT_A, SUBJECT_B = choice(82, 2, replace=False)
                # set up simulation with static control sequence from subject A
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[compare_mode] = gData.get_optimal_control_sequence(compare_mode, SUBJECT_A)
                sim.CONTROL_MODE = compare_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[SUBJECT_B]), PARAMETER_SET=str(SUBJECT_B),
                                                            return_res=True)
                gData.data[analysis.MODE_IDENTIFIER] = [None]
                gData.detail[analysis.MODE_IDENTIFIER] = [None]
                gData.data[analysis.MODE_IDENTIFIER][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER][0] = res_detail_df
                gData.cross_mode_summary([analysis.MODE_IDENTIFIER, compare_mode], [0, SUBJECT_B],
                                         [SUBJECT_A, SUBJECT_B])
                gData.plot_transfer_compare_learning_curve(compare_mode, SUBJECT_B, SUBJECT_A)
    elif OPPOSITE_CROSS_COMPARE:
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = ORIGINAL_MODE_MAP
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        for current_mode in mode_iter_lst:
            compare_mode = OPPOSITE_MODE_MAP[
                ORIGINAL_MODE_MAP.index(current_mode)]  # opposite mode for worst action sequence
            for subject in range(NUM_PARAMETER_SET):
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[current_mode] = gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject),
                                                            return_res=True)
                gData.data[analysis.MODE_IDENTIFIER2] = [None]
                gData.detail[analysis.MODE_IDENTIFIER2] = [None]
                gData.data[analysis.MODE_IDENTIFIER2][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER2][0] = res_detail_df
                gData.cross_mode_summary2([analysis.MODE_IDENTIFIER2, current_mode], [0, subject], [subject, subject])
                gData.plot_opposite_compare_learning_curve(current_mode, subject)
    elif OPPOSITE_NN_CROSS_COMPARE:
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = ORIGINAL_MODE_MAP
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        # load neural network
        net = torch.load('ControlRL/Net_140output100_best')
        net.eval()
        for current_mode in mode_iter_lst:
            compare_mode = OPPOSITE_MODE_MAP[
                ORIGINAL_MODE_MAP.index(current_mode)]  # opposite mode for worst action sequence
            for subject in range(NUM_PARAMETER_SET):
                # For opposite sequence
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[current_mode] = gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject),
                                                            return_res=True)
                gData.data[analysis.MODE_IDENTIFIER2] = [None]
                gData.detail[analysis.MODE_IDENTIFIER2] = [None]
                gData.data[analysis.MODE_IDENTIFIER2][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER2][0] = res_detail_df

                # for optimal sequence from Neural Net
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                # we should make neural net function to draw figure
                idx = MODE_LIST.index(current_mode)
                net_input = Variable(torch.FloatTensor(list(param_list[subject])[:-1] + MODE_LIST_INPUT[idx]))
                output = net(net_input[None, ...])
                output = output[0]
                decoded_output = []
                for i in range(0, len(output), 4):
                    # print(torch.argmax(test_output[i:i+4]))
                    tensor = torch.argmax(output[i:i + 4])
                    decoded_output.append(tensor.item())
                print(net_input, decoded_output)
                sim.static_action_map[
                    current_mode] = decoded_output  # [0]*sim.TRIALS_PER_EPISODE #gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject),
                                                            return_res=True)
                gData.data[analysis.MODE_IDENTIFIER] = [None]
                gData.detail[analysis.MODE_IDENTIFIER] = [None]
                gData.data[analysis.MODE_IDENTIFIER][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER][0] = res_detail_df

                gData.cross_mode_summary3([analysis.MODE_IDENTIFIER, analysis.MODE_IDENTIFIER2, current_mode],
                                          [0, 0, subject], [subject, subject, subject])
                gData.plot_opposite_nn_compare_learning_curve(current_mode, subject)
    elif FAIR_OPPOSITE_NN_CROSS_COMPARE:
        if CROSS_COMPARE_MOD == 'all':
            mode_iter_lst = ORIGINAL_MODE_MAP
        else:
            mode_iter_lst = [CROSS_COMPARE_MOD]
        # load neural network
        net = torch.load('ControlRL/Net_140output100_best')
        net.eval()
        for current_mode in mode_iter_lst:
            compare_mode = OPPOSITE_MODE_MAP[
                ORIGINAL_MODE_MAP.index(current_mode)]  # opposite mode for worst action sequence
            for subject in range(NUM_PARAMETER_SET):
                # For opposite sequence
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[current_mode] = gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject),
                                                            return_res=True)
                gData.data[analysis.MODE_IDENTIFIER2] = [None]
                gData.detail[analysis.MODE_IDENTIFIER2] = [None]
                gData.data[analysis.MODE_IDENTIFIER2][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER2][0] = res_detail_df

                # For optimal sequence from Neural Net
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                # we should make neural net function to draw figure
                idx = MODE_LIST.index(current_mode)
                net_input = Variable(torch.FloatTensor(list(param_list[subject])[:-1] + MODE_LIST_INPUT[idx]))
                output = net(net_input[None, ...])
                output = output[0]
                decoded_output = []
                for i in range(0, len(output), 4):
                    # print(torch.argmax(test_output[i:i+4]))
                    tensor = torch.argmax(output[i:i + 4])
                    decoded_output.append(tensor.item())
                print(net_input, decoded_output)
                sim.static_action_map[
                    current_mode] = decoded_output  # [0]*sim.TRIALS_PER_EPISODE #gData.get_optimal_control_sequence(compare_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject),
                                                            return_res=True)
                gData.data[analysis.MODE_IDENTIFIER] = [None]
                gData.detail[analysis.MODE_IDENTIFIER] = [None]
                gData.data[analysis.MODE_IDENTIFIER][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER][0] = res_detail_df

                # For optimal sequence from simulation
                sim.STATIC_CONTROL_AGENT = True
                sim.TRIALS_PER_EPISODE = gData.trial_separation
                sim.TOTAL_EPISODES = 1
                sim.static_action_map[current_mode] = gData.get_optimal_control_sequence(current_mode, subject)
                sim.CONTROL_MODE = current_mode
                sim.ENABLE_PLOT = False
                res_data_df, res_detail_df = sim.simulation(*(param_list[subject]), PARAMETER_SET=str(subject),
                                                            return_res=True)
                gData.data[analysis.MODE_IDENTIFIER3] = [None]
                gData.detail[analysis.MODE_IDENTIFIER3] = [None]
                gData.data[analysis.MODE_IDENTIFIER3][0] = res_data_df
                gData.detail[analysis.MODE_IDENTIFIER3][0] = res_detail_df

                gData.cross_mode_summary4(
                    [analysis.MODE_IDENTIFIER, analysis.MODE_IDENTIFIER2, analysis.MODE_IDENTIFIER3], [0, 0, 0],
                    [subject, subject, subject], current_mode)
                gData.plot_opposite_nn_compare_learning_curve(current_mode, subject, True)
    else:
        for mode, _ in tqdm(MODE_MAP.items()):
            try:
                gData.set_current_mode(mode)
                gData.generate_summary(mode)
            except KeyError:
                print('mode: ' + mode + ' data not found. Skip')
        mode = 'mixed-random'
        try:
            gData.set_current_mode(mode)
            gData.generate_summary(mode)
        except KeyError:
            print('mode: ' + mode + ' data not found. Skip')
    if TO_EXCEL is not None:
        print(
            "Making excel file.... Can choose data among ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score', 'action'] by changing arg column = 'action'")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.sequence_to_excel(i, column='action')
    if TO_EXCEL is not None:
        print(
            "Making excel file.... Can choose data among ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score', 'action'] by changing arg column = 'action'")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.sequence_to_excel(i, column='ctrl_reward')
    if TO_EXCEL is not None:
        print(
            "Making excel file.... Can choose data among ['rpe', 'spe', 'mf_rel', 'mb_rel', 'p_mb', 'ctrl_reward', 'score', 'action'] by changing arg column = 'action'")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.sequence_to_excel(i, column='score')
    if TO_EXCEL_LOG_Q_VALUE:
        print(
            "Making excel file of human action list, task structure, state transition(Forward), Q value(Forward), Q value(Sarsa), Q value(arbitrator)")
        for i in tqdm(range(NUM_PARAMETER_SET)):
            gData.log_Q_value_to_excel(i)
    if TO_EXCEL_RANDOM_MODE_SEQUENCE:
        gData.random_mode_sequence_to_excel()

    if TO_EXCEL_OPTIMAL_SEQUENCE:
        for mode, _ in tqdm(MODE_MAP.items()):
            try:
                gData.optimal_sequence_to_excel(mode)
            except KeyError:
                print('mode: ' + mode + ' data not found. Skip')


if __name__ == '__main__':
    short_opt = "hdn:"
    long_opt = ["help", "mdp-stages=", "disable-control", "ctrl-mode=", "set-param-file=", "trials=", "episodes=",
                "all-mode", "enable-static-control",
                "disable-c-ext", "disable-detail-plot", "less-control-input", "re-analysis=", "PCA-plot",
                "learning-curve-plot", "use-confidence-interval",
                "to-excel=", "disable-action-compare", "enable-score-compare", "use-selected-subjects", "save-ctrl-rl",
                "head-tail-subjects",
                "human-data-compare", "disable-auto-max", "legacy-mode", "separate-learning-curve", "cross-mode-plot",
                "cross-compare=", "sub-A=", "sub-B=",
                "enhance-compare=", "no-reset", "save-log-Q-value", "to-excel-log-Q-value", "mixed-random-mode",
                "to-excel-random-mode-sequence",
                "to-excel-optimal-sequence", "opposite-cross-compare=", "opposite-nn-cross-compare=",
                "fair-opposite-nn-cross-compare="]
    '''
    try:
        opts, args = getopt.getopt(sys.argv[1:], short_opt, long_opt)
    except getopt.GetoptError as err:
        print(err)
        usage()
        sys.exit(2)
    '''
    # make it as the code without any arguemnt (scriptize)
    # TODO total epochs should be fixed as 100, and the trials should reflect the actual subject's number of trials.
    # TODO It is quite different from the optimizing framework
    sim.TOTAL_EPISODES = int(100)
    sim.TRIALS_PER_EPISODE = int(20)
    sim.ENABLE_PLOT = False
    sim.MODEL_LIST_IDF = MODEL_LIST_IDF

    gData.trial_separation = sim.TRIALS_PER_EPISODE #
    if LOAD_PARAM_FILE:
        with open(PARAMETER_FILE) as f:
            csv_parser = csv.reader(f)
            param_list = []
            for row in csv_parser:
                param_list.append(tuple(map(float, row[:-1])))

        sbj_ctxt = []
        TRIALS_PER_EPISODE = []
        for subi in range(len(param_list)):
            with open(BEHAV_DIR + 'SUB{0:03d}_BHV.csv'.format(subi+1) ) as f:
                csv_parser = csv.reader(f)
                temp = []
                for row in csv_parser:
                    temp.append([int(row[2]), int(row[17])])
                sbj_ctxt.append(temp)
                TRIALS_PER_EPISODE.append(len(temp))

        # if ALL_MODE:
        #     for mode, _ in MODE_MAP.items():
        #         gData.new_mode(sim.CONTROL_MODE)
        #         sim.CONTROL_MODE = mode
        #         print('Running mode: ' + mode)
        #         for index in range(NUM_PARAMETER_SET):
        #             print('Parameter set: ' + str(param_list[index]))
        #             sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
        #         gData.generate_summary(sim.CONTROL_MODE)
        #         gData.save_mode(sim.CONTROL_MODE)
        #         if sim.SAVE_LOG_Q_VALUE:
        #             gData.save_log_Q_value(sim.CONTROL_MODE)
        if  sim.ARBI_SIMULGEN_MODE:
            gData.new_mode('simul_gen') # just initializing
            print('Running mode: arbitration models simul-gen')
            for index in range(NUM_PARAMETER_SET):
                print('Parameter set: ' + str(param_list[index]))
                sim.TRIALS_PER_EPISODE_sbjtv = TRIALS_PER_EPISODE[index]
                # if GRW_ == 0:
                #     sim.CTXT_sbjtv = sbj_ctxt[index]
                # else:
                #     temp=np.array(sbj_ctxt[index])
                #     temp[:, 0] = 0
                #     sim.CTXT_sbjtv = temp.tolist()
                if GRW_ == 1 or GRW_ == 2:
                    temp=np.array(sbj_ctxt[index])
                    temp[:, 0] = 0
                    sim.CTXT_sbjtv = temp.tolist()
                else:
                    sim.CTXT_sbjtv = sbj_ctxt[index]

                sim.ladder_tree_ = ladder_tree_
                sim.GRW_ = GRW_
                sim.num_action_ = num_action_

                if MODEL_LIST_IDF == 3:
                    param_temp = [1, 1, *(param_list[index])]
                    behav_ = np.load('./' + SIMUL_BEHAV_DIR + 'MODEL' + str(3) + '/SUB{0:03d}_SIMUL_BHV.npy'.format(index + 1),allow_pickle=True)
                    # sim.CTXT_sbjtv = behav_[index]
                    sim.simulation(*(param_temp), PARAMETER_SET=str(index),behave_=behav_)
                else:
                    sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
                # gData.generate_summary('simul_gen')
                gData.save_mode('simul_gen')
                if sim.SAVE_LOG_Q_VALUE:
                    gData.save_log_Q_value('simul_gen')
                # Save the whole analysis object for future reference

                with open(gData.file_name('Analysis-Object') + '.pkl', 'wb') as f:
                    pickle.dump(gData, f, pickle.HIGHEST_PROTOCOL)
                dat = []
                for epi in gData.behavior_simul:
                    try:
                        os.mkdir(SIMUL_BEHAV_DIR + 'MODEL' + str(MODEL_LIST_IDF+1))
                    except:
                        ''
                    dat.append(np.array(epi))
                np.save(SIMUL_BEHAV_DIR + 'MODEL' + str(MODEL_LIST_IDF+1) + '/SUB{0:03d}_SIMUL_BHV.npy'.format(index+1) ,np.array(dat))
        else:
            IOError('loading parameter is the only allowed one')
            # gData.new_mode(sim.CONTROL_MODE)
            # print('Running mode: ' + sim.CONTROL_MODE)
            # for index in range(NUM_PARAMETER_SET):
            #     print('Parameter set: ' + str(param_list[index]))
            #     sim.simulation(*(param_list[index]), PARAMETER_SET=str(index))
            # gData.generate_summary(sim.CONTROL_MODE)
            # gData.save_mode(sim.CONTROL_MODE)
            # if sim.SAVE_LOG_Q_VALUE:
            #     gData.save_log_Q_value(sim.CONTROL_MODE)
    else:
        IOError('loading parameter is the only allowed one')


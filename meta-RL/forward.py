""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import numpy as np
import common

from collections import defaultdict

#try:

print("C++ FORWARD dynamic library found, set as default backend")
#except ImportError:
#    print("Forward C++ dynamic library not found, only pure Python version availiable")
#    USE_CFORWARD = False

class FORWARD:
    """FORWARD model-based learner

    See the algorithm description from the publication:
    States versus Rewards: Dissociable Neural Prediction Error Signals Underlying Model-Based
    and Model-Free Reinforcement Learning http://www.princeton.edu/~ndaw/gddo10.pdf

    Currently support Discreate observation and action spaces only
    """
    RANDOM_PROBABILITY       = 0.05
    TEMPORAL_DISCOUNT_FACTOR = 1.0
    LEARNING_RATE            = 0.5
    C_SIZE_TRANSITION_PROB   = 2 # C implementation requries knowing the size of transition probability
    NUM_POSSIBLE_NEXT_STATE = 4
    OUTPUT_STATE_ARRAY = [2, 1, 1, 0, 1, 0, 2, 0, 2, 3, 3, 0, 2, 0, 0, 3]
    def __init__(self, observation_space, action_space, state_reward_func, output_offset, reward_map_func, output_states,
                 epsilon=RANDOM_PROBABILITY, discount_factor=TEMPORAL_DISCOUNT_FACTOR, learning_rate=LEARNING_RATE,
                 num_possible_next_state = NUM_POSSIBLE_NEXT_STATE, disable_cforward=False, output_state_array = OUTPUT_STATE_ARRAY):
        """Args:
            observation_space (gym.spaces.Discrete)
            action_space (gym.spaces.Discrete)
            state_reward_func (closure): a reward map to initialize state-action value dict
            output_offset (int): specify the starting point of terminal reward state
            epsilon (float): thereshold to make a random action
            learning_rate (float)
            discount_factor (float)
        """
        self.num_states    = observation_space.n
        self.num_actions   = action_space.n
        self.output_offset = output_offset
        self.num_possible_next_state = num_possible_next_state
        self.output_state_array = output_state_array
        self.output_states = output_states
        self.epsilon         = epsilon
        self.discount_factor = discount_factor
        self.learning_rate   = learning_rate
        self.T               = {} # transition matrix
        self.num_levels = 3 # this is for 2-stage MDP
        self.reset(state_reward_func)
    
    def _Q_fitting(self):
        """Regenerate state-action value dictionary and put it in a closure

        Return:
            python mode: policy_fn (closure)
            C mode: None
        """

        self.Q_fwd = defaultdict(lambda: np.zeros(self.num_actions))
        for state in reversed(range(self.num_states)):
            # Do a one-step lookahead to find the best action
            for action in range(self.num_actions):
                for next_state in reversed(range(self.num_states)):
                    prob, reward = self.T[state][action][next_state]
                    if state >= self.output_offset: # terminal reward states at the bottom of the tree
                        reward = 0
                    best_action_value = np.max(self.Q_fwd[next_state])
                    self.Q_fwd[state][action] += prob * (reward + self.discount_factor * best_action_value)

        # Create a deterministic policy using the optimal value function
        self.policy_fn = common.make_epsilon_greedy_policy(self.Q_fwd, self.epsilon, self.num_actions)
        return self.policy_fn

    def action(self, state):
        return self.policy_fn(state)

    def get_Q_values(self, state):
        """Required by some arbitrition processes

        Note if state >= output_state_offset, then
        python mode: the value will be the higest value in all states times a small transition prob
        C mode: 0
        I am not sure why it is implemented like this in python mode (it is moved from legacy code, 
        so I just keep it), but it should make no much difference.

        Args:
            state (int): a discrete value representing the state
        
        Return:
            Q_values (list): a list of Q values with indices corresponds to specific action
        """
        
        return self.Q_fwd[state]

    def optimize(self, state, action, next_state):
        """Optimize state transition matrix
        
        Args:
            state (int)
            action (int)
            next_state (int)
        
        Returns:
            state_prediction_error (float)
        """
        trans_prob = self.T[state][action]
        for post_state in range(self.num_states):
            prob, reward = trans_prob[post_state]
            if post_state == next_state:
                spe = 1 - prob
                trans_prob[post_state] = (prob + self.learning_rate * spe, reward)
            else:
                trans_prob[post_state] = (prob * (1 - self.learning_rate), reward)
        self.T[state][action] = trans_prob
        self._Q_fitting()
        return spe

    def env_reset(self, state_reward_func, reward_map_func):
        """Called by the agent communication controller when environment sends a
        reset signal

        Args:
            state_reward_func (closure): as in constructor
            reward_map_func : function made in mdp.py
        """
        self.reset(state_reward_func, False)

    def reset(self, state_reward_func, reset_trans_prob=True):
        self.state_reward_func = state_reward_func
        for state in range(self.num_states):
            if reset_trans_prob:
                self.T[state] = {action: [] for action in range(self.num_actions)}
            for action in range(self.num_actions):
                for next_state in range(self.num_states):
                    self.T[state][action].append(((1./self.num_states if reset_trans_prob else self.T[state][action][next_state]), 
                                                  self.state_reward_func(next_state)))
        # build state-action value mapping
        self._Q_fitting()
        self.T_flex = self.T

    def get_Transition(self, state, action):
        return self.T[state][action]
    
    def set_Q_values(self,Q_values):
        self.Q_fwd=Q_values
        
    def set_Transition(self,T_map):
        self.T=T_map

    def bwd_update(self, bwd_idf, env):
        reward_map=[0,0,0,0,0,40,20,10,0]
        if bwd_idf != -1:
        #else:
            #self.T=self.T_flex
            # for state in range(self.num_states):
            #     for action in range(self.num_actions):
            #         for next_state in range(self.num_states):
            #             self.T[state][action][next_state][1] = self.T_flex[state][action][next_state][1]
            reward_idf = reward_map[bwd_idf-1]

        before = self.Q_fwd
        # layer 2 (second stage)
        state_ind_set = (np.int16(np.linspace(1,self.num_possible_next_state,self.num_possible_next_state)).tolist())

        states_ = [0]
        for sei in range(len(state_ind_set)):
            states_.append(0)
        bwd_output_states = []
        for ele in env.output_states_to_compare:
            if bwd_idf != -1:
                if int(reward_map[bwd_idf-1]) == int(ele):
                    states_.append(ele)
                    bwd_output_states.append(ele)
                else:
                    states_.append(0)
                    bwd_output_states.append(0)
            else:
                states_.append(ele)
                bwd_output_states.append(ele)


        for cur_st in state_ind_set:
            for cur_act in range(self.num_actions):
                tmp_sum = []
                for j in range((self.num_states)):
                    tmp_sum += self.T[cur_st][cur_act][j][0]*(states_[j]+max(self.Q_fwd[j]))
                self.Q_fwd[cur_st,cur_act]=tmp_sum

        # Create a deterministic policy using the optimal value function
        self.policy_fn = common.make_epsilon_greedy_policy(self.Q_fwd, self.epsilon, self.num_actions)

        env.output_states = bwd_output_states
        self.output_states = bwd_output_states
        self.reward_states = states_
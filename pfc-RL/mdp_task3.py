""" Shuangyi Tong <s9tong@edu.uwaterloo.ca>
    Sept 17, 2018
"""
import random
import numpy as np
import gym

from numpy.random import choice
from gym import spaces
from common import AgentCommController

class MDP(gym.Env):
    """Markov Decision Process env class, inherited from gym.Env
    Although there is not much code in gym.Env, it is just
    showing we want to support general gym API, to be able
    to easily run different environment with existing code in the future

    The MDP implemented here support arbitrary stages and arbitrary
    possible actions at any state, but each state share the same number of
    possible actions. So the decision tree is an n-tree
    """

    """MDP constants
    
    Access of observation and action space should refer
    to these indices
    """
    HUMAN_AGENT_INDEX   = 0
    CONTROL_AGENT_INDEX = 1
    GRW_ = 0
    ladder_tree_ = 1
    LEGACY_MODE           = False
    STAGES                = 2
    TRANSITON_PROBABILITY = [0.9, 0.1]
    NUM_ACTIONS           = 2
    POSSIBLE_OUTPUTS      = [0, 10, 20, 40]
    BIAS_TOGGLE_INCREMENT = 40
    REWARD_OUTPUT_STATES = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
    # REWARD_OUTPUT_STATES = [40, 0, 40, 0, 40, 0, 40, 0, 40, 0, 40, 0, 40, 0, 40, 0]
    """Control Agent Action Space
    0 - doing nothing
    1 - set stochastic: apply uniform distribution of transition probability
    2 - set deterministic: set transition probability to original one
    3 - randomize human reward: reset reward shape
    4 - reset: randomize human reward and set deterministic transition probability
    5 - reset2: randomize human reward and set stochastic transition probability
    """
    NUM_CONTROL_ACTION    = 4

    def __init__(self, stages=STAGES, trans_prob=TRANSITON_PROBABILITY, num_actions=NUM_ACTIONS,
                 outputs=POSSIBLE_OUTPUTS, more_control_input=True, legacy_mode=LEGACY_MODE, GRW=GRW_,ladder_tree=ladder_tree_):
        """
        Args:
            stages (int): stages of the MDP
            trans_prob (list): an array specifying the probability of transitions
            num_actions (int): number of actions possible to take at non-leaf state
                by player. Note total number of possible actions should be multiplied
                by the size of trans_prob
            outputs (list): an array specifying possible outputs
            more_control_input (bool): more element in control observation
            legacy_mode (bool): if use legacy implementation of MDP
        """

        self.GRW = GRW
        self.ladder_tree = ladder_tree

        # environment global variables
        self.stages            = stages # fixed at 2
        self.human_state       = 0 # start from zero
        self.legacy_mode       = legacy_mode
        if self.legacy_mode:
            self.max_rpe       = outputs[-1] + MDP.BIAS_TOGGLE_INCREMENT
            self.toggle_bit    = 0
        else:
            self.max_rpe       = outputs[-1]

        # human agent variables
        self.action_space      = [spaces.Discrete(num_actions)] # human agent action space
        self.trans_prob        = trans_prob
        self.trans_prob_init   = trans_prob
        if self.ladder_tree == 1:
            self.possible_actions  = len(self.trans_prob) * num_actions
        else:
            self.possible_actions  = num_actions

        self.outputs           = outputs # type of outputs
        self.num_output_states = pow(self.possible_actions, self.stages)
        self.reward_map = [0, 10, 20, 40] #this is corressponding to state 5, 6, 7, 8

        if self.GRW == 0:
            output_state_seed = [[20, 10],[10, 0],[10, 0],[20, 0],[20,40],[40,0],[20,0],[0,40]]
            output_state_seed = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
            if self.ladder_tree ==0:
                output_state_seed = [20, 0, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 10, 0, 40] # to enhance specific goal
        else:
            output_state_seed = [[40,0]]
            output_state_seed = [40,0]

        self.output_states = np.zeros((self.num_output_states)).astype(dtype=np.int16).tolist()

        self.output_states_to_compare =self.output_states
        # if ladder_tree == 1:
        filling_ = False
        target_loc = self.num_output_states-1
        bullet_loc = len(output_state_seed)-1
        while not filling_:
            if bullet_loc < 0:
                # reloading
                bullet_loc = len(output_state_seed)-1
                if self.ladder_tree == 1 and self.GRW != 0:
                    output_state_seed.reverse()
            if target_loc  <0:
                filling_ = True
            else:
                self.output_states[target_loc] = output_state_seed[bullet_loc]
                bullet_loc -= 1
                target_loc -= 1

        # self.output_states = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]#choice(outputs, self.num_output_states)
        # self.output_states = [40, 0, 40, 0, 40, 0, 40, 0, 40, 0, 40, 0, 40, 0, 40, 0]#choice(outputs, self.num_output_states)
        self.output_states_offset = int((pow(self.possible_actions, self.stages) - 1)
            / (self.possible_actions - 1)) # geometric series summation
        if self.legacy_mode:
            self.num_states    = self.output_states_offset + len(self.outputs)
        else:
            self.num_states    = self.output_states_offset + self.num_output_states
        self.observation_space = [spaces.Discrete(self.num_states)] # human agent can see states only
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()
        self.remember_output_states = self.output_states.copy()
        self.remember_reward_map = self.reward_map.copy()
        self.is_flexible = 1 #  1 if flexible goal condition and 0 if spicific goal condition

        # control agent variables
        self.more_control_input = more_control_input
        self.action_space.append(spaces.Discrete(MDP.NUM_CONTROL_ACTION)) # control agent action space
        if more_control_input:
            if legacy_mode:
                output_structure = spaces.Discrete(1) # one toggle bit
            else:
                output_structure = spaces.Discrete(self.num_output_states) # output states
            self.observation_space.append(spaces.Tuple((
                output_structure, # depends on if it is legacy mode or 
                spaces.Box(low=0, high=1, shape=(num_actions,), dtype=float), # transition probability 
                spaces.Box(low=0, high=1, shape=(1,), dtype=float), # rpe
                spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)))) # spe
        else:
            self.observation_space.append(spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(1,), dtype=float), # rpe
                spaces.Box(low=0, high=np.inf, shape=(1,), dtype=float)))) # spe

        # for reset reference
        self.trans_prob_reset = trans_prob

        # agent communication controller
        self.agent_comm_controller = AgentCommController()

        # ADDED % 1 means flexible and 0 means speclfc?
        self.prev_isflexible = 1
        self.prev_isflexible = 1
        self.ctxt = 3 # 1: specific low 4: flexible low 2: specific high 3: flexible high
        self.num_GRW = 0
        # pre-walking gaussian random walk
        if self.GRW != 0:
            if self.GRW == 3 or self.GRW== 4:
                for grwi in np.linspace(self.stages-1,0,self.stages).astype(dtype=np.int16)[:1]:
                    self.num_GRW += pow(self.possible_actions,grwi)
                self.GRW_offset = 1 + self.possible_actions - self.num_GRW
                self._gen_gauss_random_walk_shifts()
            else:
                for grwi in np.linspace(self.stages-1,0,self.stages).astype(dtype=np.int16)[:self.GRW]:
                    self.num_GRW += pow(self.possible_actions,grwi)
                self.GRW_offset = 1 + self.possible_actions - self.num_GRW
                self._gen_gauss_random_walk()

    def _make_state_reward_func(self):
        return lambda s: self.output_states[s - self.output_states_offset] \
               if s >= self.output_states_offset else 0


    def _make_reward_map_func(self):
        return lambda s: self.reward_map[s] \
               if s < len(self.reward_map) else 0

    def _make_control_observation(self):
        if self.more_control_input:
            if self.legacy_mode:
                return np.concatenate([[self.toggle_bit], np.array(self.trans_prob)])
            else:
                return np.concatenate([self.output_states, np.array(self.trans_prob)])
        else:
            return []

    def step(self, action):
        """"Take one step in the environment
        
        Args:
            action ([int, action]): a two element tuple, first sepcify which agent
            second is the action valid in that agent's action space

        Return (human):
            human_obs (int): an integer represent human agent's observation, which is
            equivalent to 'state' in this environment
            human_reward (float): reward received at the end of the game
            done (boolean): if the game termiate
            control_obs_frag (numpy.array): fragment of control observation, need to append reward
        Return (control):
            None, None, None, None: just match the arity
        """
        if action[0] == MDP.HUMAN_AGENT_INDEX:
            """ Human action
            Calculate the index of the n-tree node, start from 0
            Each node has possible_actions childs, the calculation is a little tricky though.
            Output_states_offset is the max index of internal nodes + 1
            Greater or equal to output_states_offset means we need to get reward from output_states
            """
            # update agent due to task context
            self._update_trans_prob()
            if self.GRW == 1 and self.human_state==0:
                self._update_trans_prob()
            elif self.GRW == 1 and self.human_state<self.output_states_offset:
                self.trans_prob=self._update_GRW()
            elif self.GRW == 2:
                self.trans_prob=self._update_GRW()

            if self.GRW == 3:
                if self.human_state == 0:
                    self._update_trans_prob()
                else:
                    self.trans_prob = self._update_GRW_shifts()
            if self.GRW==4:
                self.trans_prob = self._update_GRW_shifts()


            # if self.ladder_tree == 1:
            #     state = self.human_state * self.possible_actions + \
            #             choice(range(action[1] * len(self.trans_prob) + 1, (action[1] + 1) * len(self.trans_prob) + 1),
            #                    1, True, self.trans_prob)[0]
            # elif self.ladder_tree ==0:
            if self.ladder_tree == 1:
                state = self.human_state * self.possible_actions + \
                    choice(range(action[1] * len(self.trans_prob) + 1, (action[1] + 1) * len(self.trans_prob) + 1),
                           1, True, self.trans_prob)[0]
            else:
                seed_direct = [[0,1],[1,0]]
                state = self.human_state*self.possible_actions + 1 +\
                choice(seed_direct[action[1]], 1, True, self.trans_prob)[0]
                if not state < self.num_states:
                    state = self.output_states_offset-1 + 1
                    # choice(#range(action[1]*self.possible_actions + 1, (action[1] + 1)*self.possible_actions + 1),
                    #        1, True, self.trans_prob)[0]

            reward = self.state_reward_func(state)
            self.human_state = state
            if state < self.output_states_offset:
                done = False
            else:
                done = True
                if self.legacy_mode:
                    self.human_state = self.output_states_offset + self.outputs.index(reward)
                    reward += MDP.BIAS_TOGGLE_INCREMENT if self.toggle_bit else 0
            return self.human_state, reward, done, self._make_control_observation()
        elif action[0] == MDP.CONTROL_AGENT_INDEX:
            """ Control action
            Integrate functional and object oriented programming techniques
            to create this pythonic, compact code, similar to switch in other language
            """
            [lambda env: env, # do nothing
             lambda env: env._set_stochastic_trans_prob(), # uniform trans_prob
             lambda env: setattr(env, 'trans_prob', env.trans_prob_reset), # reset trans prob to deterministic
             lambda env: env._specific_flexible_switch(), #env._output_swap()  #env._output_reset()   # reset reward shape
             #lambda env: env._set_opposite_trans_prob()
             #lambda env: env._output_reset_with_deterministic_trans_prob(), # reset
             #lambda env: env._output_reset_with_stochastic_trans_prob() # reset2
            ][action[1]](self)
            return None, None, None, None
        else:
            raise ValueError

    def _set_stochastic_trans_prob(self):
        self.trans_prob = [1./len(self.trans_prob) for i in range(len(self.trans_prob))]


    def _set_opposite_trans_prob(self):
        self.trans_prob = [0.1, 0.9]


    def _output_swap(self):
        self.output_states = list(map((lambda x: 0 if x == 20 else
                                                 10 if x == 40 else
                                                 20 if x == 0 else
                                                 40), self.output_states))
        self.reward_map = list(map((lambda x: 0 if x == 20 else
                                                 10 if x == 40 else
                                                 20 if x == 0 else
                                                 40), self.reward_map))
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()
        self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)

    def _output_average_with_stochastic_trans_prob(self):
        self.output_states = [0.9 * (x - 20) + 20 for x in self.output_states]
        self.state_reward_func = self._make_state_reward_func()
        self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)
        self._set_stochastic_trans_prob()

    def _output_reset_with_stochastic_trans_prob(self):
        self._output_reset()
        self._set_stochastic_trans_prob()

    def _output_reset_with_deterministic_trans_prob(self):
        self._output_reset()
        self.trans_prob = self.trans_prob_reset

    def _output_reset(self):
        """Reset parameters, used as an action in control agent space
        """
        #change output_states to state number 5-8
        self.output_states = list(map((lambda x : 5 if x == self.reward_map[0] else 
                                                  6 if x == self.reward_map[1] else 
                                                  7 if x == self.reward_map[2] else 
                                                  8), self.output_states))  
        random.shuffle(self.reward_map)
        #change state number 5-8 to reward according to shuffled reward_map
        self.output_states = list(map((lambda x : self.reward_map[0] if x == 5 else 
                                                  self.reward_map[1] if x == 6 else 
                                                  self.reward_map[2] if x == 7 else 
                                                  self.reward_map[3]), self.output_states))      
        #self.output_states = choice(self.outputs, self.num_output_states)


        # refresh the closure as well
        self.state_reward_func = self._make_state_reward_func()
        self.reward_map_func = self._make_reward_map_func()
        # reset human agent
        self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)

    def _specific_flexible_switch(self):
        """Specific goal condition <-> Flexible goal condition(10 or 20 or 40 is randomly chosen)"""
        self.is_flexible = (self.is_flexible + 1) % 2 #  1 if flexible goal condition and 0 if spicific goal condition
        if self.is_flexible == 0: # speicific goal condition
            self.remember_output_states = self.output_states.copy()
            self.remember_reward_map = self.reward_map.copy()
            specific_goal = random.choice([10, 20, 40])
            self.output_states = list(map((lambda x : specific_goal if x == specific_goal else 
                                                      0), self.output_states))
            self.reward_map = list(map((lambda x : specific_goal if x == specific_goal else 
                                                      0), self.reward_map))

            # refresh the closure as well
            self.state_reward_func = self._make_state_reward_func()
            self.reward_map_func = self._make_reward_map_func()
            # reset human agent
            self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)
        else: # flexible goal condition
            self.output_states = self.remember_output_states.copy()
            self.reward_map = self.remember_reward_map.copy()

            # refresh the closure as well
            self.state_reward_func = self._make_state_reward_func()
            self.reward_map_func = self._make_reward_map_func()
            # reset human agent
            self.agent_comm_controller.reset('model-based', self.state_reward_func, self.reward_map_func)

    def _gen_gauss_random_walk(self):
        """gaussian random walk probability updates
        :return:
        """
        self.GRW_ub = 0.75
        self.GRW_lb = 0.25
        GRWList=[]
        GRW_seeds = np.linspace(self.GRW_lb,self.GRW_ub,self.num_GRW)
        random.shuffle(GRW_seeds)
        for grwi in range(self.num_GRW):
            GRWList.append([GRW_seeds[grwi], 1-GRW_seeds[grwi]])
        self.GRW_list = GRWList

    def _gen_gauss_random_walk_shifts(self):
        """gaussian random walk probability updates
        :return:
        """
        GRW_seeds = [0.9, 0.5]
        # random.shuffle(GRW_seeds)
        self.GRW_ub_list = []
        self.GRW_lb_list = []
        for i in range(2):
            self.GRW_ub_list.append(GRW_seeds[i]+.1)
            self.GRW_lb_list.append(GRW_seeds[i]-.1)

        GRWList=[]
        for grwi in range(2):
            GRWList.append([GRW_seeds[grwi], 1-GRW_seeds[grwi]])
        self.GRW_list = GRWList

    def _update_GRW_shifts(self):
        # update part

        for grwi in range(2):
            seed_ = self.GRW_list[grwi][0]
            valid_update = False
            while not valid_update:
                rw_step = (0.01* np.random.randn(1) + 0)[0]
                if (seed_+rw_step<self.GRW_ub_list[grwi]) and (seed_+rw_step>self.GRW_lb_list[grwi]):
                    valid_update = True
            seed_+= rw_step
            self.GRW_list[grwi] = [seed_, 1-seed_]


        if self.ctxt == 1:
            self.trans_prob =  self.GRW_list[0]  # 1: specific low 4: flexible low 2: specific high 3: flexible high
        elif self.ctxt == 2:
            self.trans_prob =  self.GRW_list[1]   # 1: specific low 4: flexible low 2: specific high 3: flexible high
        elif self.ctxt == 3:
            self.trans_prob =  self.GRW_list[0]   # 1: specific low 4: flexible low 2: specific high 3: flexible high
        else:
            self.trans_prob =  self.GRW_list[1]   # 1: specific low 4: flexible low 2: specific high 3: flexible high

        # wheter return the current trans prob or updated GRW trans prob
        ''
        return self.trans_prob

    def _update_trans_prob(self):
        if self.GRW == 0:
            if self.ctxt == 1:
                self.trans_prob = [.9, .1]  # 1: specific low 4: flexible low 2: specific high 3: flexible high
            elif self.ctxt == 2:
                self.trans_prob = [.5, .5]  # 1: specific low 4: flexible low 2: specific high 3: flexible high
            elif self.ctxt == 3:
                self.trans_prob = [.9, .1]  # 1: specific low 4: flexible low 2: specific high 3: flexible high
            else:
                self.trans_prob = [.5, .5]  # 1: specific low 4: flexible low 2: specific high 3: flexible high
        else:
            self.trans_prob = self.trans_prob_init # GRW 1 and GRW 2 will be dealt in step.

    def _update_GRW(self):
        # update part
        for grwi in range(self.num_GRW):
            seed_ = self.GRW_list[grwi][0]
            valid_update = False
            while not valid_update:
                rw_step = (0.025* np.random.randn(1) + 0)[0]
                if (seed_+rw_step<self.GRW_ub) and (seed_+rw_step>self.GRW_lb):
                    valid_update = True
            seed_+= rw_step
            self.GRW_list[grwi] = [seed_, 1-seed_]
        if self.human_state < self.GRW_offset: # not required
            return self.trans_prob
        else: # updates
            self.trans_prob = self.GRW_list[self.human_state-self.GRW_offset]
            return self.trans_prob
        # wheter return the current trans prob or updated GRW trans prob
        ''

    def reset_gaussian_random_walk(self):
        self._gen_gauss_random_walk()
        
    def reset(self):
        """Reset the environment before game start or after game terminates

        Return:
            human_obs (int): human agent observation
            control_obs_frag (numpy.array): control agent observation fragment, see step
        """
        self.human_state = 0
        return self.human_state, self._make_control_observation()
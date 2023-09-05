import random
import numpy as np
import gym
from math import exp

from numpy.random import choice
from gym import spaces
import torch
# https://github.com/awjuliani/Meta-RL

ACTION_SPACE = 2
N_STATE = 21
N_CTXT = 4
N_STAGE = 2
OBS_SPACE = N_STATE#+N_CTXT
OBS_SPACE_out = N_STATE+N_CTXT
TRANSITION_PROBABILITY = [[0.5, 0.5], [0.9, 0.1]]
REWARD_LIST = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]

REWARD = [20, 10, 0, 40]

class PMEnv(gym.Env):
    def __init__(self, filename):
        super(PMEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTION_SPACE)
        #self.observation_space = spaces.Box(low=0, high=1, shape=(OBS_SPACE,), dtype=int)
        self.observation_space = spaces.Discrete(OBS_SPACE)
        self.mat = np.int16(np.genfromtxt(filename,delimiter=',',usecols=(3,4,5,6,7,17)))
        # mat[, -1] => bucket color // if -1 : flexible else specific
        self.ctm = self.mat.shape[0]


    def step(self, action):

        observation = np.zeros(shape=(OBS_SPACE,), dtype=int)
        observation_out = np.zeros(shape=(OBS_SPACE_out,), dtype=int)
        state = int(self.mat[self.cti, self.stg+1])
        observation[state] = 1
        observation_out[state]=1

        cidx = (int(self.mat[self.cti, -1]) % 5)+15
        observation_out[cidx] = 1

        reward_bench = [40,20,10,0]
        if (self.stg == 1):
            if self.mat[self.cti, -1] != -1:
                if self.mat[self.cti, -1] == self.mat[self.cti, self.stg+1]:
                    reward = reward_bench[self.mat[self.cti,self.stg+1]-6]
                else:
                    reward = 0
            else:
                reward = reward_bench[self.mat[self.cti,self.stg+1]-6]
        else:
            reward = 0


        self.stg += 1
        if self.stg == 2:
            self.stg = 0
            self.cti += 1

        if (self.cti < self.ctm):
            done = False
        else:
            done = True
            self.cti = 0

        return observation, reward, done, {}

    
    def likelihood(self, _agent_val):
        res = 1
        like_a = []
        
        for i in range(len(_agent_val)):
            norm = _agent_val[i] / np.linalg.norm(_agent_val[i])
            summ = (np.sum(np.exp(norm)))
            if summ == 0:
                summ += 1e-6
            a_prob = np.exp(norm) / summ
            a = self.mat[self.cti, self.stg+i+1]-1
            likeli_ = a_prob[a]
            if likeli_ == 0:
                likeli_ += 1e-6
            res *= likeli_
            
        return res


    def reset(self):
        self.agent_a = []
        self.cti = 0
        self.stg = 0

        observation = np.zeros(shape=(OBS_SPACE,), dtype=int)
        # state = int(self.mat[self.cti, self.stg])
        self.pos = 0 #state-1
        observation[self.pos] = 1

        #cidx = (int(self.mat[self.cti, -1]) % 5) + 15
        #observation[cidx] = 1
        return observation

    def render(self):
        pass


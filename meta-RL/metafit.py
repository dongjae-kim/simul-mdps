import random
import numpy as np
import gym


from numpy.random import choice
from gym import spaces

N_DISCRETE_ACTIONS = 2
N_STATE = 16
N_CTXT = 4
N_STAGE = 2
N_OBS = N_STATE+N_CTXT

class FitEnv(gym.Env):
    def __init__(self, filename):
        super(FitEnv, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(N_OBS,), dtype=int)
        self.mat = np.genfromtxt(filename,delimiter=',',usecols=(3,4,5,6,7,17))

        self.ctm = self.mat.shape[0]

    def step(self, action):
        if (self.stg < 2):
            self.stg += 1
        else:
            self.stg = 0
            self.cti += 1

        if (self.cti < self.ctm):
            done = False
        else:
            done = True
            self.cti = 0

        observation = np.zeros(shape=(N_OBS,), dtype=int)
        state = int(self.mat[self.cti, self.stg])
        observation[state] = 1

        cidx = (int(self.mat[self.cti, -1]) % 5)+15
        observation[cidx] = 1

        if (self.stg != 0):
            reward = 2*(action+1 == self.mat[self.cti, self.stg+2])-1
        else:
            reward = 0

        return observation, reward, done, {}

    def reset(self):
        self.cti = 0
        self.stg = 0

        observation = np.zeros(shape=(N_OBS,), dtype=int)
        state = int(self.mat[self.cti, self.stg])
        observation[state] = 1

        cidx = (int(self.mat[self.cti, -1]) % 5) + 15
        observation[cidx] = 1
        return observation

    def render(self):
        pass
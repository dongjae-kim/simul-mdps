import gym
from gym import spaces
import numpy as np
import random

# Click L : Move Left First, 5:5(left/right) move
ACTION_SPACE = 2
OBS_SPACE = 21 #+ 5
OBS_SPACE_out = 21+4
REWARD_LIST = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
TRANSITION_PROBABILITY = [[0.5, 0.5], [0.9, 0.1]]

class CustomEnv(gym.Env):
    def __init__(self, filename):
        super(CustomEnv, self).__init__()
        
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Discrete(OBS_SPACE)
        
        self.mat = np.int16(np.genfromtxt(filename,delimiter=',',usecols=(3,4,5,6,7,17)))
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

    def reset(self):
        self.stg = 0
        self.pos = 0
        self.cti = 0

        observation = np.zeros(shape=(OBS_SPACE,), dtype=int)
        observation[0] = 1

        return observation


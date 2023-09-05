import gym
from gym import spaces
import numpy as np
import random

# Click L : Move Left First, 5:5(left/right) move
ACTION_SPACE = 2
OBS_SPACE = 21 #+ 5
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
        
        if self.mat[self.cti, -1] == -1:
            trans_prob = TRANSITION_PROBABILITY[0]
        else:
            trans_prob = TRANSITION_PROBABILITY[1]

        L_R = random.choices([action,1-action], weights=trans_prob)[0]
        
        self.pos = self.pos*4+1 + action*2 + L_R
        observation[self.pos] = 1 # self.pos*4+1 = next stage / action*2 = left:+0 right:+2 / L_R = transition probability
        
        self.stg += 1
        if self.stg == 2:
            self.cti += 1
            reward = REWARD_LIST[self.pos - 5]
            done = True
        elif self.stg == 1:
            reward = 0
            done = False
        
        
        if self.cti < self.ctm:
            pass
        else:
            self.cti = 0

        return observation, reward, done, {}

    def reset(self):
        self.stg = 0
        self.pos = 0
        self.cti = 0

        observation = np.zeros(shape=(OBS_SPACE,), dtype=int)
        observation[0] = 1

        return observation


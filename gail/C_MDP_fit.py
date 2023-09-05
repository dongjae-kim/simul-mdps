import gym
from gym import spaces
import numpy as np
import random

# Click L : Move Left First, 5:5(left/right) move
ACTION_SPACE = 2
OBS_SPACE = 22 #+ 5
REWARD_LIST = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
REWARD_LIST2 = [0,0,0,0,0,40,20,10,0]
# 6,7,8,-1 = 40,20,10,0
REWARD_COLOR = [7,8,8,-1,8,-1,7,-1,7,6,6,-1,7,-1,-1,6]# [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
TRANSITION_PROBABILITY = [[0.5, 0.5], [0.9, 0.1]]

class CustomEnv(gym.Env):
    def __init__(self, filename):
        super(CustomEnv, self).__init__()
        
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Discrete(OBS_SPACE)
        
        self.mat = np.int16(np.genfromtxt(filename,delimiter=',',usecols=(3,4,5,6,7,17)))
        self.ctm = self.mat.shape[0]

    def _reward(self, pos, gc = -1):

        ret = 0

        if gc == -1:
            # flexible goal condition - an agent gets an reward without no constraints
            ret = REWARD_LIST[pos] / 40
        else:
            # specific goal condition
            # - an agent gets the reward when satisfying the condition
            # : the bucket color should be equal to the ball color
            if REWARD_COLOR[pos] == gc:
                #ret = REWARD_LIST[pos]
                ret = 1

        return ret

    def step(self, action, gc=-1, unc=1):
        observation = np.zeros(shape=(OBS_SPACE,), dtype=int)

        '''
        if self.mat[self.cti, -1] == -1:
            trans_prob = TRANSITION_PROBABILITY[0]
        else:
            trans_prob = TRANSITION_PROBABILITY[1]
        '''
        # if gc == -1:
        #     trans_prob = TRANSITION_PROBABILITY[0]
        # else:
        #     trans_prob = TRANSITION_PROBABILITY[1]
        if (unc == 2) or (unc == 3):
            trans_prob = TRANSITION_PROBABILITY[0]
        else:
            trans_prob = TRANSITION_PROBABILITY[1]

        L_R = random.choices([action,1-action], weights=trans_prob)[0]
        
        #self.pos = self.pos*4+1 + action*2 + L_R
        self.pos = self.mat[self.cti,self.stg+1]-1
        observation[self.pos] = 1 # self.pos*4+1 = next stage / action*2 = left:+0 right:+2 / L_R = transition probability
        
        self.stg += 1
        if self.stg == 2:
            self.cti += 1
            # reward = self._reward(self.pos-5, gc = gc)
            reward = REWARD_LIST2[self.pos]
            #REWARD_LIST[self.pos - 5]
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


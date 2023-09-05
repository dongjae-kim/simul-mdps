
import matplotlib.pyplot as plt
import numpy as np
import srdyna_pm
import importlib
import csv
import os
import sys
import pickle as pkl


EXPLORE_STEPS = 100
POST_REWARD_TRIALS = 40

# subind = int(sys.argv[1])
# simnum = int(sys.argv[2])

simnum = 1
for simul in range(100):
    for subind in range(82):
        evt_file = './bhv_results0{0}/{1}/SUB{2:03d}_BHV.csv'.format(0,0,subind+1)
        bhv_file = './bhv_results0{0}/{1}/SUB{2:03d}_BHV.csv'.format(simnum,simul,subind+1)

        evt_mat = np.loadtxt(evt_file,delimiter=',')
        bhv_mat = np.loadtxt(bhv_file,delimiter=',')

        bhv_mat[:,17] = evt_mat[:,17]
        np.savetxt(bhv_file, bhv_mat, delimiter=',')


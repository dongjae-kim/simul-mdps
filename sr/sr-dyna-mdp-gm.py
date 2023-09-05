# %%
"""
# SR-Dyna with mdp env(gm)
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import srdyna_gm
import importlib
import csv
import os


# %%
importlib.reload(srdyna_gm)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')

# %%
# Detour Task
EXPLORE_STEPS = 10000
EXPLORE_STEPS = 10
POST_REWARD_TRIALS = 1000
bhv_file = './bhv_results/SUB001_BHV.csv'
env = srdyna_gm.MDPWorld(max_reward_locs=16, world='worlds/mdp.txt', filename=bhv_file)
S_LOC = (0, 0)
agent = srdyna_gm.SRDyna(id=0, loc=S_LOC, env=env, post_step_replays=10, exp_lambda=1/5.)


# Explore
print("Exploring...")
for i in range(EXPLORE_STEPS):
    for j in range(2):
        agent.step(random_policy=True)


# Add reward
R_LOC = (2, 0)
env.add_reward(R_LOC, 20)
R_LOC = (2, 1)
env.add_reward(R_LOC, 10)
R_LOC = (2, 2)
env.add_reward(R_LOC, 10)
R_LOC = (2, 3)
env.add_reward(R_LOC, 0)

R_LOC = (2, 4)
env.add_reward(R_LOC, 10)
R_LOC = (2, 5)
env.add_reward(R_LOC, 0)
R_LOC = (2, 6)
env.add_reward(R_LOC, 20)
R_LOC = (2, 7)
env.add_reward(R_LOC, 0)

R_LOC = (2, 8)
env.add_reward(R_LOC, 20)
R_LOC = (2, 9)
env.add_reward(R_LOC, 40)
R_LOC = (2, 10)
env.add_reward(R_LOC, 40)
R_LOC = (2, 11)
env.add_reward(R_LOC, 0)

R_LOC = (2, 12)
env.add_reward(R_LOC, 20)
R_LOC = (2, 13)
env.add_reward(R_LOC, 0)
R_LOC = (2, 14)
env.add_reward(R_LOC, 0)
R_LOC = (2, 15)
env.add_reward(R_LOC, 40)
print(env.reward_locs,"/n")

# Reset
agent.terminate_episode()
env.cti=0

# Learning reward - test
n=0
cr_train = []
cr_test = []
createFolder('./out_reward')

while(n<POST_REWARD_TRIALS):
    n += 1
    # 1 - learning reward
    for i in range(2):
        done, r=agent.step(verbose=False)
        if(done==True):
            cr_train.append(r)
    # 2 - test
    if n>0 and n%10==0:
        for i in range(2):
            done, r = agent.step(random_policy=False, verbose=True, learning=False)
            if (done == True):
                cr_test.append(r)


# Save reward
f = open('./out_reward/train.csv', 'w', newline='')
wr = csv.writer(f)
wr.writerow(cr_train)
f.close()

f = open('./out_reward/test.csv', 'w', newline='')
wr = csv.writer(f)
wr.writerow(cr_test)
f.close()

print(len(cr_train), len(cr_test))
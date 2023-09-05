# %%
"""
# SR-Dyna with mdp env(bc)
"""

# %%
import srdyna_bc
import importlib
import csv
import os


# %%
importlib.reload(srdyna_bc)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')

# %%
# Detour Task
EXPLORE_STEPS = 400
POST_REWARD_TRIALS = 40

bhv_file = './bhv_results0/SUB001_BHV.csv'
env = srdyna_bc.MDPWorld(world='worlds/mdp.txt', filename=bhv_file)
S_LOC = (0, 0)
agent = srdyna_bc.SRDyna(id=0, loc=S_LOC, env=env, post_step_replays=10, exp_lambda=1 / 5.)

# Explore
print("Exploring...")
for i in range(EXPLORE_STEPS):
    for j in range(env.ctm):
        for s in range(2):
            agent.step(random_policy=True, is_reward=False)


# Reset
agent.terminate_stg()
env.cti=0

# Learning reward - test
n=0
cr_train = []
cr_test = []
createFolder('./out_reward')

while(n<POST_REWARD_TRIALS):
    n += 1
    # 1 - learning reward
    c_r=0
    for i in range(env.ctm):
        for s in range(2):
            done, r = agent.step(is_reward=True)
            c_r += r
    cr_train.append(c_r)
    # 2 - test
    if n>0 and n%4==0:
        c_r=0
        for i in range(env.ctm):
            for s in range(2):
                done, r = agent.step(random_policy=False, verbose=True, learning=False, is_reward=True)
                c_r+=r
        cr_test.append(c_r)


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
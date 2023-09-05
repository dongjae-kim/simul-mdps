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
import sys
import pickle as pkl

# %%
importlib.reload(srdyna_gm)

def reward_job():
    # Add reward
    R_LOC = (2, 0)
    agent.env.add_reward(R_LOC, 20)
    R_LOC = (2, 1)
    agent.env.add_reward(R_LOC, 10)
    R_LOC = (2, 2)
    agent.env.add_reward(R_LOC, 10)
    R_LOC = (2, 3)
    agent.env.add_reward(R_LOC, 0)

    R_LOC = (2, 4)
    agent.env.add_reward(R_LOC, 10)
    R_LOC = (2, 5)
    agent.env.add_reward(R_LOC, 0)
    R_LOC = (2, 6)
    agent.env.add_reward(R_LOC, 20)
    R_LOC = (2, 7)
    agent.env.add_reward(R_LOC, 0)

    R_LOC = (2, 8)
    agent.env.add_reward(R_LOC, 20)
    R_LOC = (2, 9)
    agent.env.add_reward(R_LOC, 40)
    R_LOC = (2, 10)
    agent.env.add_reward(R_LOC, 40)
    R_LOC = (2, 11)
    agent.env.add_reward(R_LOC, 0)

    R_LOC = (2, 12)
    agent.env.add_reward(R_LOC, 20)
    R_LOC = (2, 13)
    agent.env.add_reward(R_LOC, 0)
    R_LOC = (2, 14)
    agent.env.add_reward(R_LOC, 0)
    R_LOC = (2, 15)
    agent.env.add_reward(R_LOC, 40)
    print(agent.env.reward_locs, "/n")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error')

# %%
# Detour Task

# subind = int(sys.argv[1])
# simnum = int(sys.argv[2])

subind = 2
simnum = 0

EXPLORE_STEPS = 1000
POST_REWARD_TRIALS = 100000
bhv_file = './bhv_results{0}/{1}/SUB{2:03d}_BHV.csv'.format(simnum,0,subind+1)

bhv_mat = np.loadtxt(bhv_file,delimiter=',')
env = srdyna_gm.MDPWorld(max_reward_locs=16, world='worlds/mdp.txt', filename=bhv_file)
S_LOC = (0, 0)
agent = srdyna_gm.SRDyna(id=0, loc=S_LOC, env=env, post_step_replays=10, exp_lambda=1/5.)

save_path = './pretrained_model/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Explore
if not os.path.exists(save_path + 'SUBS.pkl'):
    # Explore
    print("Exploring...")
    for i in range(EXPLORE_STEPS):
        print(str(i) + '/' + str(EXPLORE_STEPS))
        for j in range(env.ctm):
            for s in range(2):
                agent.step(random_policy=True)

    with open(save_path + 'SUBS.pkl', 'wb') as f:
        pkl.dump(agent, f)
else:
    with open(save_path + 'SUBS.pkl', 'rb') as f:
        agent = pkl.load(f)

# # Explore
# print("Exploring...")
# for i in range(EXPLORE_STEPS):
#     for j in range(2):
#         agent.step(random_policy=True)


REWARD_LIST = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
THIRD_STAGE = [ 7,  8,  8, 9,  8, 9,  7, 9,  7,  6,  6, 9,  7, 9, 9,  6]

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


NUM_SIMUL = 100

for i_s in range(NUM_SIMUL):
    if not os.path.exists('./bhv_results_gm{0}/{1}/'.format(simnum+1,i_s)):
        os.makedirs('./bhv_results_gm{0}/{1}/'.format(simnum+1,i_s))
    bhv_save = './bhv_results_gm{0}/{1}/SUB{2:03d}_BHV.csv'.format(simnum+1,i_s,subind+1)
    sim_mat = np.zeros((bhv_mat.shape[0],bhv_mat.shape[1]+2))

    if not os.path.exists(bhv_save):

        try:
            agent.env.mat = np.int16(np.genfromtxt(bhv_file, delimiter=',', usecols=(3, 4, 5, 6, 7, 17)))
            agent.ctm = agent.env.mat.shape[0]
            agent.cti = 0
            agent.env.cti = 0
            agent.env.ctm = agent.env.mat.shape[0]
        except:
            ''
        cr_train = []
        cr_test = []

        # while(n<POST_REWARD_TRIALS):
        #     n += 1
        # 1 - learning reward
        # agent.action = None
        # c_r=0
        # for i in range(env.ctm):
        #     for s in range(2):
        #         _,_, _, _, _, _, done, r = agent.step_likeli(is_reward=True)
        #         # _,_, _, _, _, _, done, r = agent.step()
        #         c_r += r
        # cr_train.append(c_r)

        agent.action = None
        agent.env.stg = 0
        agent.env.cti = 0
        agent.env.max_reward_locs=16
        reward_job()
        sim_mat[:,:15]= bhv_mat[:,:15]
        # 2 - test
        c_r=0
        state3 = []
        for i in range(env.ctm):
            sim_mat[i,:18]=bhv_mat[i,:18]
            if i == 50:
                print('asdf')
            for s in range(2):
                # if s == 1:
                #     sim_mat[i,3+s]=agent.state+1
                # else:
                #     sim_mat[i,3+s]=agent.state+1

                if s == 0 & agent.state != 0:
                    print('sddfa')

                last_state, last_action, last_likeli, state,  action, likeli, done, r = agent.step_likeli(random_policy=False, verbose=True, learning=False, is_reward=True)

                if s == 0 & last_state != 0:
                    print('sddfa')

                if s == 1:
                    if last_state == 16:
                        last_state_ = 1
                    elif last_state == 20:
                        last_state_ = 2
                    elif last_state == 24:
                        last_state_ = 3
                    elif last_state == 28:
                        last_state_ = 4

                    state3.append(state)
                    sim_mat[i,3+1]=last_state_+1
                    sim_mat[i,3+2]=THIRD_STAGE[state-33]
                    sim_mat[i,6+0]=last_action+1
                    sim_mat[i,6+1]=action+1
                    sim_mat[i,18]=last_likeli[last_action]
                    sim_mat[i,19]=likeli[action]
                    sim_mat[i,15]=r
                    sim_mat[i,16]=np.sum(sim_mat[:,15])
                else:
                    sim_mat[i,3]=last_state+1

                if i > 0:
                    if sim_mat[i,3] != 1:
                        print('sddfad')

                c_r+=r

            print(sim_mat[i,3:6])
            print('8'*20)
        cr_test.append(c_r)


        # # Save reward
        # f = open('./out_reward/train.csv', 'w', newline='')
        # wr = csv.writer(f)
        # wr.writerow(cr_train)
        # f.close()
        #
        # f = open('./out_reward/test.csv', 'w', newline='')
        # wr = csv.writer(f)
        # wr.writerow(cr_test)
        # f.close()

        print(len(cr_train), len(cr_test))
        np.savetxt(bhv_save, sim_mat, delimiter=',')


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
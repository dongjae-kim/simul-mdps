# %%
"""
# SR-Dyna with mdp env(bc)
"""

# %%
import numpy as np
import srdyna_bc
import importlib
import os
import pickle as pkl


# %%
importlib.reload(srdyna_bc)

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
EXPLORE_STEPS = 100
POST_REWARD_TRIALS = 40

# subind = int(sys.argv[1])
# simnum = int(sys.argv[2])

subind = 0
simnum = 0#int(sys.argv[2])
# iternum = 0#int(sys.argv[3])

save_path = './pretrained_model_gm/{0}/SUB{1:03d}'.format(simnum,subind+1)
if not os.path.exists(save_path):
    os.makedirs(save_path)

bhv_file = './bhv_results{0}/{1}/SUB{2:03d}_BHV.csv'.format(simnum,0,subind+1)

bhv_mat = np.loadtxt(bhv_file,delimiter=',')
env = srdyna_bc.MDPWorld(world='worlds/mdp.txt', filename=bhv_file)
S_LOC = (0, 0)
agent = srdyna_bc.SRDyna(id=0, loc=S_LOC, env=env, post_step_replays=10, exp_lambda=1 / 5.)



REWARD_LIST = [20, 10, 10, 0, 10, 0, 20, 0, 20, 40, 40, 0, 20, 0, 0, 40]
THIRD_STAGE = [ 7,  8,  8, 9,  8, 9,  7, 9,  7,  6,  6, 9,  7, 9, 9,  6]

# Reset
agent.terminate_stg()
env.cti=0

# Learning reward - test
n=0
# createFolder('./out_reward')

NUM_SIMUL = 100



for i_s in range(NUM_SIMUL):
    if not os.path.exists('./bhv_results_bc{0}/{1}/'.format(simnum+1,i_s)):
        os.makedirs('./bhv_results_bc{0}/{1}/'.format(simnum+1,i_s))
    bhv_save = './bhv_results_bc{0}/{1}/SUB{2:03d}_BHV.csv'.format(simnum+1,i_s,subind+1)
    sim_mat = np.zeros((bhv_mat.shape[0],bhv_mat.shape[1]+2))

    if not os.path.exists(bhv_save):
        # Explore
        if not os.path.exists(save_path + '.pkl'):
            print("Exploring...")
            for i in range(EXPLORE_STEPS):
                print(str(i)+'/'+str(EXPLORE_STEPS))
                for j in range(env.ctm):
                    for s in range(2):
                        agent.step(random_policy=True, is_reward=False)

            with open(save_path + '.pkl','wb') as f:
                pkl.dump(agent,f)
        else:
            with open(save_path+'.pkl','rb') as f:
                agent=pkl.load(f)

        cr_train = []
        cr_test = []

        # while(n<POST_REWARD_TRIALS):
        #     n += 1
        # 1 - learning reward
        c_r=0
        for i in range(env.ctm):
            for s in range(2):
                done, r = agent.step(is_reward=True)
                c_r += r
        cr_train.append(c_r)

        agent.action = None
        agent.env.stg = 0
        agent.env.cti = 0
        agent.env.max_reward_locs=16
        reward_job()
        sim_mat[:,:15]= bhv_mat[:,:15]
        # 2 - test
        c_r=0
        for i in range(env.ctm):
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
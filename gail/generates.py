import os
import gym
import pickle
import argparse
import numpy as np
from collections import deque

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from utils.utils import *
from utils.zfilter import ZFilter
from model import Actor, Critic, Discriminator
from train_model import train_actor_critic, train_discrim

from C_MDP import CustomEnv

parser = argparse.ArgumentParser(description='PyTorch GAIL')
parser.add_argument('--env_name', type=str, default="CartPole-v1",
                    help='name of the environment to run')
parser.add_argument('--load_model', type=str, default=None,
                    help='path to load the saved model')
parser.add_argument('--render', action="store_true", default=False,
                    help='if you dont want to render, set this to False')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discounted factor (default: 0.99)')
parser.add_argument('--lamda', type=float, default=0.98,
                    help='GAE hyper-parameter (default: 0.98)')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='hidden unit size of actor, critic and discrim networks (default: 100)')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='learning rate of models (default: 3e-4)')
parser.add_argument('--l2_rate', type=float, default=1e-3,
                    help='l2 regularizer coefficient (default: 1e-3)')
parser.add_argument('--clip_param', type=float, default=0.2,
                    help='clipping parameter for PPO (default: 0.2)')
parser.add_argument('--discrim_update_num', type=int, default=2,
                    help='update number of discriminator (default: 2)')
parser.add_argument('--actor_critic_update_num', type=int, default=10,
                    help='update number of actor-critic (default: 10)')
parser.add_argument('--total_sample_size', type=int, default=2048,
                    help='total sample size to collect before PPO update (default: 2048)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size to update (default: 64)')
parser.add_argument('--suspend_accu_exp', type=float, default=0.99,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
parser.add_argument('--suspend_accu_gen', type=float, default=0.99,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')
parser.add_argument('--max_iter_num', type=int, default=4000,
                    help='maximal number of main iterations (default: 4000)')
parser.add_argument('--seed', type=int, default=500,
                    help='random seed (default: 500)')
parser.add_argument('--logdir', type=str, default='logs',
                    help='tensorboardx logs directory')
parser.add_argument("--subid", type=int, default=0,
                    help="Subject's ID, default = 0 (0~81)")
parser.add_argument("--numsim", type=int, default=0,
                    help="Number of iterations, default = 0 (0~99)")
parser.add_argument("--numiter", type=int, default=0,
                    help="Number of iterations, default = 0 (0~2)")
args = parser.parse_args()
THIRD_STAGE = [ 7,  8,  8, 9,  8, 9,  7, 9,  7,  6,  6, 9,  7, 9, 9,  6]

def main():

    # args.subid  = 1
    # args.numsim = 0
    # args.numiter=  1

    print('#' * 50)
    print('#' * 50)
    print('#' * 5 + ' ' * 5 + 'SUB:' + str(args.subid + 1) + ' ' * 5 + str(args.numiter) + 'th simul')
    print('#' * 50)
    print('#' * 50)
    loading_env = './bhv_results0' + str(args.numiter) + '/' + str(0)
    env = CustomEnv(loading_env + '/SUB{0:03d}_BHV.csv'.format(args.subid + 1))  # gym.make(args.env_name)
    env.seed(args.seed)
    torch.manual_seed(args.seed)

    num_inputs = env.observation_space.n
    num_actions = env.action_space.n  # .shape[0]
    running_state = ZFilter((num_inputs,), clip=5)

    print('state size:', num_inputs)
    print('action size:', num_actions)

    actor = Actor(num_inputs, num_actions, args)
    critic = Critic(num_inputs, args)
    discrim = Discriminator(num_inputs + num_actions, args)

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(critic.parameters(), lr=args.learning_rate,
                              weight_decay=args.l2_rate)
    discrim_optim = optim.Adam(discrim.parameters(), lr=args.learning_rate)

    # load demonstrations
    # expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.p', "rb"))
    expert_demo = np.int16(np.genfromtxt(loading_env + '/SUB{0:03d}_BHV.csv'.format(args.subid + 1), delimiter=',',
                                         usecols=(3, 4, 5, 6, 7, 17)))  # 393 * [each state obs & action1,2]
    bhv_dat = np.loadtxt(loading_env + '/SUB{0:03d}_BHV.csv'.format(args.subid + 1), delimiter=',')
    expert_data = []
    # total number of games per subject
    n_games = expert_demo.shape[0]
    uncs = bhv_dat[:,2]
    for i in range(n_games):
        one_game_history = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        s0 = expert_demo[i, 0] - 1
        s1 = expert_demo[i, 1] - 1
        s2 = expert_demo[i, 2] - 1
        a1 = expert_demo[i, 3] - 1
        a2 = expert_demo[i, 4] - 1
        goal_condition = expert_demo[i, 5]

        one_game_history[s0] = 1
        one_game_history[s1] = 1
        one_game_history[s1 * 4 + (s2 - 5)]
        # one_game_history[s2] = 1
        one_game_history[21] = goal_condition
        one_game_history[22] = a1
        one_game_history[23] = a2

        expert_data.append(one_game_history)

    demonstrations = np.array(expert_data)
    print("demonstrations.shape", demonstrations.shape)

    writer = SummaryWriter(args.logdir)

    save_path = os.getcwd() + '/save_model/' + str(args.numiter)
    print('make dirs!!' * 20)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_path = save_path + '/' + 'SUB{0:03d}'.format(args.subid + 1)

    ckpt_path = model_path + '_ckpt_' + str(0) + '.pth.tar'

    # load the ckpt you need
    saved_ckpt_path = ckpt_path
    ckpt = torch.load(saved_ckpt_path)

    actor.load_state_dict(ckpt['actor'])
    critic.load_state_dict(ckpt['critic'])
    discrim.load_state_dict(ckpt['discrim'])

    running_state.rs.n = ckpt['z_filter_n']
    running_state.rs.mean = ckpt['z_filter_m']
    running_state.rs.sum_square = ckpt['z_filter_s']

    print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))
    # load ckpt ends

    episodes = 0
    train_discrim_flag = True


    for iter in range(1): #fixed number of simulation you moron
        actor.eval(), critic.eval()
        memory = deque()

        steps = 0
        scores = []

        simul_data = []
        # while steps < args.total_sample_size:
        # while steps < demonstrations.shape[0]:
        for steps in range(demonstrations.shape[0]):

            state = env.reset()
            score = 0
            simul_data_trial = bhv_dat[steps,:18].tolist()

            state = running_state(state)
            likelihood = []

            for _ in range(2):
                if args.render:
                    env.render()

                # steps += 1

                mu, std = actor(torch.Tensor(state).unsqueeze(0))
                # mu, std = actor(torch.Tensor(state,device='cuda').unsqueeze(0))
                action = get_action(mu, std)[0]

                likelihood_ = np.zeros((2)) #action/np.sum(action)
                for sim_ in range(100):
                    action_ = get_action(mu, std)[0]
                    likelihood_[int(np.argmax(action_))] += 1
                likelihood_ = likelihood_/100

                #

                print("ACTION : {}, arg = {}".format(action, np.argmax(action)))

                next_state, reward, done, _ = env.step(np.argmax(action), unc=int(uncs[steps]), gc=demonstrations[steps, 21])

                irl_reward = get_reward(discrim, state, action)

                if done:
                    mask = 0
                    steps += 1
                else:
                    # steps += 1
                    mask = 1

                memory.append([state, action, irl_reward, mask])

                next_state = running_state(next_state)
                state = next_state
                print(env.pos)
                print(env.stg)

                simul_data_trial.append(likelihood_[int(simul_data_trial[5+env.stg]-1)])

                if env.stg == 2:
                    simul_data_trial[3+env.stg] = THIRD_STAGE[env.pos-5]
                else:
                    simul_data_trial[3+env.stg] = env.pos+1
                    simul_data_trial[5+env.stg] = np.argmax(action)+1
                score += reward
                # update it
                # simul_data_trial

                if done:
                    break
            simul_data.append(simul_data_trial)
            episodes += 1
            scores.append(score)

        # if iter >= 300:

    print('save dirs!!' * 20)
    print('save dirs!!' * 20)

    saving_env = './bhv_results0' + str(args.numiter+1) + '/' + str(args.numsim)
    if not os.path.exists(saving_env):
        os.makedirs(saving_env)
    np.savetxt(saving_env + '/SUB{0:03d}_BHV.csv'.format(args.subid + 1), np.array(simul_data), delimiter=',')


    import sys

    sys.exit(0)


if __name__ == "__main__":
    main()

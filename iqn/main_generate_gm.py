import torch
from agent import IQN_Agent
import numpy as np
import random
import math
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
import gym
import argparse
import wrapper
import MultiPro
from C_MDP_Gen import CustomEnv
import os


def evaluate(eps, frame, eval_runs=5):
    """
    Makes an evaluation run with the current epsilon
    """

    reward_batch = []
    for i in range(eval_runs):
        state = eval_env.reset()
        rewards = 0
        while True:
            action = agent.act(np.expand_dims(state, axis=0), 0.001, eval=True)
            state, reward, done, _ = eval_env.step(action[0].item())
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)

    writer.add_scalar("Reward(eval)", np.mean(reward_batch), int(frame / 786))


def run(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01, eval_every=1000, eval_runs=5, worker=1):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    i_episode = 1
    state = envs.reset()
    score = 0
    for frame in range(1, frames + 1):
        action = agent.act(state, eps)
        next_state, reward, done, _ = envs.step(action)  # returns np.stack(obs), np.stack(action) ...
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d, writer)
        state = next_state
        score += np.mean(reward)
        # linear annealing to the min epsilon value (until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            # if frame < eps_frames:
            eps = max(eps_start - ((frame * d_eps) / eps_frames), min_eps)
            # else:
            #   eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)

        # evaluation runs
        if frame % eval_every == 0 or frame == 1:
            evaluate(eps, frame * worker, eval_runs)

        if done.any():
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar("Average Reward(train)", np.mean(scores_window), int(frame / 768) * worker)
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker,
                                                                             np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker,
                                                                                np.mean(scores_window)))
            i_episode += 1
            state = envs.reset()
            score = 0


def run_gen(frames=1000, eps_fixed=False, eps_frames=1e6, min_eps=0.01, eval_every=1000, eval_runs=5, worker=1,bhv_mat = []):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    bhv_mat = csv_data
    outdata = (np.zeros((bhv_mat.shape[0], bhv_mat.shape[1]+2)))

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    frame = 0
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    i_episode = 1
    # state = envs.reset()
    state = simul_env.reset()
    state = state.reshape((1,-1))
    score = 0
    for frame in range(1, frames + 1):
        if simul_env.stg == 0:
            outdata[simul_env.cti,:18]=bhv_mat[simul_env.cti,:18]
        state = state.reshape((1,-1))
        action, likeli = agent.act_likeli(state, eps)

        outdata[simul_env.cti, 3+simul_env.stg] = np.where(state)[1][0]
        outdata[simul_env.cti, 6 + simul_env.stg] = action
        outdata[simul_env.cti, 18 + simul_env.stg] = likeli

        # next_state, reward, done, _ = envs.step(action)  # returns np.stack(obs), np.stack(action) ...
        next_state, reward, done, _ = simul_env.step(action)


        next_state = next_state.reshape((1,-1))
        if simul_env.stg == 0:
            outdata[simul_env.cti,3+2] = np.where(next_state)[1][0]
            outdata[simul_env.cti, 15] = reward
            outdata[simul_env.cti, 16] = np.sum(outdata[:, 15])


        s, a, r, ns, d = state, action, reward, next_state, done
        # for s, a, r, ns, d in zip(state, action, reward, next_state, done):
        #     agent.step(s, a, r, ns, d, writer)
        # agent.step(s, a, r, ns, d, writer)

        print('['+ str(frame) +'/' +str(frames) +']  '  + '['+ str(simul_env.cti) +'/' +str(simul_env.ctm) +']')

        state = next_state
        score += np.mean(reward)
        # linear annealing to the min epsilon value (until eps_frames and from there slowly decease epsilon to 0 until the end of training
        if eps_fixed == False:
            # if frame < eps_frames:
            eps = max(eps_start - ((frame * d_eps) / eps_frames), min_eps)
            # else:
            #   eps = max(min_eps - min_eps*((frame-eps_frames)/(frames-eps_frames)), 0.001)

        # evaluation runs
        if frame % eval_every == 0 or frame == 1:
            evaluate(eps, frame * worker, eval_runs)

        if done:
            scores_window.append(score)  # save most recent score
            scores.append(score)  # save most recent score
            writer.add_scalar("Average Reward(train)", np.mean(scores_window), int(frame / 768) * worker)
            print('\rEpisode {}\tFrame {} \tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker,
                                                                             np.mean(scores_window)), end="")
            if i_episode % 100 == 0:
                print('\rEpisode {}\tFrame {}\tAverage100 Score: {:.2f}'.format(i_episode * worker, frame * worker,
                                                                                np.mean(scores_window)))
            i_episode += 1
            state = envs.reset()
            score = 0

            if not os.path.exists(save_path + '/'):
                os.makedirs(save_path + '/')
            np.savetxt(save_path + '/' + '/SUB{0:03d}_BHV.csv'.format(args.subid+1), outdata,
                       delimiter=',')

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-agent", type=str, choices=["iqn",
                                                     "iqn+per",
                                                     "noisy_iqn",
                                                     "noisy_iqn+per",
                                                     "dueling",
                                                     "dueling+per",
                                                     "noisy_dueling",
                                                     "noisy_dueling+per"
                                                     ], default="iqn",
                        help="Specify which type of IQN agent you want to train, default is IQN - baseline!")

    parser.add_argument("-env", type=str, default="CartPole-v0",
                        help="Name of the Environment, default = BreakoutNoFrameskip-v4")
    parser.add_argument("-frames", type=int, default=1000000, help="Number of frames to train, default = 10 mio")
    parser.add_argument("-eval_every", type=int, default=10000, help="Evaluate every x frames, default = 250000")
    parser.add_argument("-eval_runs", type=int, default=2, help="Number of evaluation runs, default = 2")
    parser.add_argument("-seed", type=int, default=1, help="Random seed to replicate training runs, default = 1")
    parser.add_argument("-N", type=int, default=32, help="Number of Quantiles, default = 8")
    parser.add_argument("-munchausen", type=int, default=0, choices=[0, 1],
                        help="Use Munchausen RL loss for training if set to 1 (True), default = 0")
    parser.add_argument("-bs", "--batch_size", type=int, default=32,
                        help="Batch size for updating the DQN, default = 32")
    parser.add_argument("-layer_size", type=int, default=512, help="Size of the hidden layer, default=512")
    parser.add_argument("-n_step", type=int, default=1, help="Multistep IQN, default = 1")
    parser.add_argument("-m", "--memory_size", type=int, default=int(15000), help="Replay memory size, default = 1e5")
    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate, default = 2.5e-4")
    parser.add_argument("-g", "--gamma", type=float, default=0.99, help="Discount factor gamma, default = 0.99")
    parser.add_argument("-t", "--tau", type=float, default=5e-3, help="Soft update parameter tau, default = 1e-3")
    parser.add_argument("-eps_frames", type=int, default=75000,
                        help="Linear annealed frames for Epsilon, default = 1mio")
    parser.add_argument("-min_eps", type=float, default=0.025, help="Final epsilon greedy value, default = 0.01")
    parser.add_argument("-info", type=str, default="goal_matching", help="Name of the training run")
    # parser.add_argument("-info", type=str, default="iqn_run1", help="Name of the training run")
    parser.add_argument("-save_model", type=int, choices=[0, 1], default=1,
                        help="Specify if the trained network shall be saved or not, default is 1 - save model!")
    parser.add_argument("-w", "--worker", type=int, default=1,
                        help="Number of parallel Environments. Batch size increases proportional to number of worker. not recommended to have more than 4 worker, default = 1")
    parser.add_argument("-subid", "--subid", type=int, default=0,
                        help="Subject's ID, default = 0 (0~81)")
    parser.add_argument("-numsim", "--numsim", type=int, default=0,
                        help="Number of iterations, default = 0 (0~99)")
    parser.add_argument("-numiter", "--numiter", type=int, default=0,
                        help="Number of iterations, default = 0 (0~2)")

    args = parser.parse_args()

    # bhv_file = './bhv_results' + str(args.numsim) + '/0/SUB{0:03d}_BHV.csv'.format(args.subid+1)

    writer = SummaryWriter("runs/" + str(args.numsim) + '/' + args.info + '_SUB{0:03d}'.format(args.subid+1))
    seed = args.seed
    BUFFER_SIZE = args.memory_size
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    TAU = args.tau
    LR = args.lr
    n_step = args.n_step
    env_name = args.env
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    bhv_file = './bhv_results' + str(args.numsim) + '/0/SUB{0:03d}_BHV.csv'.format(args.subid+1)

    if "noisy" in args.agent:
        eps_fixed = True
    else:
        eps_fixed = False

    envs = MultiPro.SubprocVecEnv([lambda: CustomEnv(bhv_file) for i in range(args.worker)])
    simul_env = CustomEnv(bhv_file)
    eval_env = CustomEnv(bhv_file)

    envs.seed(seed)
    eval_env.seed(seed+1)

    action_size = eval_env.action_space.n
    state_size = [eval_env.observation_space.n]

    agent = IQN_Agent(state_size=state_size,
                      action_size=action_size,
                      network=args.agent,
                      munchausen=args.munchausen,
                      layer_size=args.layer_size,
                      n_step=n_step,
                      BATCH_SIZE=BATCH_SIZE,
                      BUFFER_SIZE=BUFFER_SIZE,
                      LR=LR,
                      TAU=TAU,
                      GAMMA=GAMMA,
                      N=args.N,
                      worker=args.worker,
                      device=device,
                      seed=seed)

    # model.load_state_dict(torch.load(PATH))

    agent.qnetwork_local.load_state_dict(torch.load("./pretrained_model_gm/" + args.info+".pth",map_location=torch.device('cpu')))


    csv_data = np.int16(np.loadtxt(bhv_file, delimiter=",", dtype=np.float32))

    save_path = './bhv_results' + str(args.numsim+1) + '/' + str(args.numiter) + '/'
    # parser.add_argument("-subid", "--subid", type=int, default=0,
    #                     help="Subject's ID, default = 0 (0~81)")
    # parser.add_argument("-numsim", "--numsim", type=int, default=0,
    #                     help="Number of iterations, default = 0 (0~99)")
    # parser.add_argument("-numiter", "--numiter", type=int, default=0,
    #                     help="Number of iterations, default = 0 (0~2)")

    # eps_fixed=False
    eps_fixed=True

    t0 = time.time()
    run_gen(frames = eval_env.mat.shape[0]*2, eps_fixed=eps_fixed, eps_frames=int(eval_env.mat.shape[0]/10), min_eps=args.min_eps, eval_every=1, eval_runs=args.eval_runs, worker=args.worker, bhv_mat = csv_data)
    t1 = time.time()
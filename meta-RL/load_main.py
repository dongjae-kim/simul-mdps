import sys
from load_agent import load_agent
from metafit import FitEnv
import numpy as np
import tensorflow as tf

TRAIN = int(sys.argv[1])
TEST = int(sys.argv[2])
MAX = int(sys.argv[3])

model_path = './SUBFIT/SUBS-'+str(TRAIN)+'/FIN'
tstr = 'bhv_results/SUB'+format(TEST,'03d')+'_BHV.csv'

tf.reset_default_graph()
with tf.Session() as sess:
    env = FitEnv(tstr)
    agent = load_agent(model_path, sess)

    step = 0
    episode_count = 0

    while (step<MAX):
        episode_reward = 0
        episode_step_count = 0
        episode_likelihood = 0

        s = env.reset()
        agent.reset()
        d = False
        r = 0

        while d==False:
            action, a_dist = agent.act(state=s, reward=r)
            s, r, d, t = env.step(action)

            episode_reward += r
            step += 1

            if (r != 0):
                episode_step_count += 1
                episode_likelihood += -np.log(a_dist[0, 1-(r>0)^action])


        acc = (episode_reward/episode_step_count+1)/2
        rows = [episode_count, step, acc, episode_likelihood]
        dstr = "TEST {0} - ACC: {1:.3f}, Sum of log-Likelihood: {2:3.3f}".format(rows[0], rows[2], rows[3])
        print(dstr)

        episode_count += 1






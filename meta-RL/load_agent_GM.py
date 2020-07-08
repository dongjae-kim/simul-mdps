import os
import tensorflow as tf
import numpy as np
from module import *

MAX = 800000
gamma = .8
a_size = 2
dim_obs = 21
num_slice = 10
train = True

class load_agent:
    def __init__(self, model_path, sess):

        with tf.device("/cpu:0"):
            self.global_episodes = tf.Variable(0, dtype=tf.float32, name='global_episodes', trainable=False)
            self.trainer = tf.train.AdamOptimizer(learning_rate=1e-3)
            self.master_network = AC_Network(a_size, dim_obs, 'global', None)
            self.workers = []

            for i in range(1):
                self.workers.append(Worker([], i, a_size, dim_obs, self.trainer, model_path, self.global_episodes, MAX, num_slice))

        self.sess = sess
        saver = tf.train.Saver()
        saver.restore(sess, model_path+'/train_model.ckpt')

        self.reset()

    def reset(self):
        self.t = 0
        self.a = 0
        self.rnn_state = self.master_network.state_init

    def act(self, state, reward):
        with self.sess.as_default() as sess:
            self.s = state
            self.r = reward

            a_dist, v, rnn_state_new = sess.run(
                [self.master_network.policy, self.master_network.value, self.master_network.state_out],
                feed_dict={
                    self.master_network.state: [self.s],
                    self.master_network.prev_rewards: [[self.r]],
                    self.master_network.prev_actions: [self.a],
                    self.master_network.state_in[0]: self.rnn_state[0],
                    self.master_network.state_in[1]: self.rnn_state[1]})

            a = np.random.choice(a_dist[0], p=a_dist[0])
            self.a = np.argmax(a_dist==a)

            self.rnn_state = rnn_state_new

        return self.a, a_dist
import csv
import pickle
import scipy.signal
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper import *

class AC_Network():
    def __init__(self, a_size, dim_obs, scope, trainer):
        with tf.variable_scope(scope):
            # Input and visual encoding layers
            self.state = tf.placeholder(shape=[None, dim_obs], dtype=tf.float32)
            self.prev_rewards = tf.placeholder(shape=[None, 1], dtype=tf.float32)
            self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.prev_actions_onehot = tf.one_hot(self.prev_actions, a_size)
            hidden = tf.concat([slim.flatten(self.state), self.prev_rewards, self.prev_actions_onehot], 1)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.state_in = (c_in, h_in)

            rnn_in = tf.expand_dims(hidden, [0])
            step_size = tf.shape(self.prev_rewards)[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)

            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, 256])

            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32)

            # Output layers for policy and value estimations
            self.policy = slim.fully_connected(rnn_out, a_size,
                                               activation_fn=tf.nn.softmax,
                                               weights_initializer=normalized_columns_initializer(0.01),
                                               biases_initializer=None)
            self.value = slim.fully_connected(rnn_out, 1,
                                              activation_fn=None,
                                              weights_initializer=normalized_columns_initializer(1.0),
                                              biases_initializer=None)

            # Only the worker network need ops for loss functions and gradient updating.
            if scope != 'global':
                self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
                self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

                self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])

                # Loss functions
                self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.target_v - tf.reshape(self.value, [-1])))
                self.entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 1e-7))
                self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs + 1e-7) * self.advantages)
                self.loss = 0.5 * self.value_loss + self.policy_loss - self.entropy * 0.05

                # Get gradients from local network using local losses
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 999.0)

                # Apply local gradients to global network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.apply_grads = trainer.apply_gradients(zip(grads, global_vars))

"""### Worker Agent"""
class Worker():
    def __init__(self, game, name, a_size, dim_obs, trainer, model_path, global_episodes, MAX, slice):
        self.name = "worker_" + str(name)
        self.number = name
        self.model_path = model_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_mean_values = []

        # Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_AC = AC_Network(a_size, dim_obs, self.name, trainer)
        self.update_local_ops = update_target_graph('global', self.name)
        self.env = game
        self.MAX = MAX

        self.idx = 1
        self.slice = slice

    def train(self, rollout, sess, gamma, bootstrap_value):
        rollout = np.array(rollout)
        states = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        timesteps = rollout[:, 3]
        prev_rewards = [0] + rewards[:-1].tolist()
        prev_actions = [0] + actions[:-1].tolist()
        values = rollout[:, 5]

        self.pr = prev_rewards
        self.pa = prev_actions
        # Here we take the rewards and values from the rollout, and use them to
        # generate the advantage and discounted returns.
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus, gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = discount(advantages, gamma)

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        rnn_state = self.local_AC.state_init
        feed_dict = {self.local_AC.target_v: discounted_rewards,
                     self.local_AC.state: np.stack(states, axis=0),
                     self.local_AC.prev_rewards: np.vstack(prev_rewards),
                     self.local_AC.prev_actions: prev_actions,
                     self.local_AC.actions: actions,
                     self.local_AC.advantages: advantages,
                     self.local_AC.state_in[0]: rnn_state[0],
                     self.local_AC.state_in[1]: rnn_state[1]}
        v_l, p_l, e_l, g_n, v_n, _ = sess.run([self.local_AC.value_loss,
                                               self.local_AC.policy_loss,
                                               self.local_AC.entropy,
                                               self.local_AC.grad_norms,
                                               self.local_AC.var_norms,
                                               self.local_AC.apply_grads],
                                              feed_dict=feed_dict)
        return v_l / len(rollout), p_l / len(rollout), e_l / len(rollout), g_n, v_n

    def work(self, gamma, sess, coord, train):
        episode_count = sess.run(self.global_episodes)
        total_steps = 0
        print("Starting worker... " + str(self.number))
        with sess.as_default(), sess.graph.as_default():
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_values = []
                episode_reward = 0
                episode_step_count = 0
                episode_likelihood = 0

                d = False
                r = 0
                a = 0
                t = 0
                s = self.env.reset()
                rnn_state = self.local_AC.state_init

                while d == False:
                    # Take an action using probabilities from policy network output.
                    a_dist, v, rnn_state_new = sess.run(
                        [self.local_AC.policy, self.local_AC.value, self.local_AC.state_out],
                        feed_dict={
                            self.local_AC.state: [s],
                            self.local_AC.prev_rewards: [[r]],
                            self.local_AC.prev_actions: [a],
                            self.local_AC.state_in[0]: rnn_state[0],
                            self.local_AC.state_in[1]: rnn_state[1]})
                    a = np.random.choice(a_dist[0], p=a_dist[0])
                    a = np.argmax(a_dist == a)

                    rnn_state = rnn_state_new
                    s1, r, d, t = self.env.step(a)
                    episode_buffer.append([s, a, r, t, d, v[0, 0]])
                    episode_values.append(v[0, 0])
                    episode_reward += r
                    total_steps += 1
                    
                    episode_step_count += 1
                    s = s1

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_step_count)
                self.episode_mean_values.append(np.mean(episode_values))

                self.env.reset()

                # Update the network using the experience buffer at the end of the episode.
                if len(episode_buffer) != 0 and train == True:
                    v_l, p_l, e_l, g_n, v_n = self.train(episode_buffer, sess, gamma, 0.0)

                if self.name == 'worker_0':
                    sess.run(self.increment)
                    f = open(self.model_path+'/monitor.csv','a',encoding='utf-8',newline='')
                    wr = csv.writer(f)

                    avgrew = episode_reward/episode_step_count*3
                    rows = [episode_count, episode_step_count, avgrew]
                    wr.writerow(rows)

                    f.close()

                    dstr = "Episode {0} - Accuracy: {1:.3f}".format(rows[0],avgrew)
                    print(dstr)

                    if (total_steps > self.idx*self.MAX/self.slice):
                        saver = tf.train.Saver()
                        saver.save(sess,self.model_path+'/'+str(self.idx)+'/train_model.ckpt')
                        self.idx += 1

                    if (total_steps > self.MAX):
                        saver = tf.train.Saver()
                        saver.save(sess,self.model_path+'/FIN/train_model.ckpt')
                        coord.request_stop()

                episode_count += 1

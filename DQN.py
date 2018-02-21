# Filename: DQN.py
from collections import deque
import os
import random
import numpy as np
import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
REPLAY_SIZE = 20000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
isTest = False


class DQN():
    """ This is function foo"""
    # DQN Agent
    def __init__(self, state_dim, action_dim, hiden_dim):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        # self.state_dim = env.observation_space.shape[0]
        # self.action_dim = env.action_space.n
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hiden_dim = hiden_dim
        self.batch_step = 0

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        ''' network weights '''
        W1 = self.weight_variable([self.state_dim, 512], 'w1')
        b1 = self.bias_variable([512], 'b1')
        W2 = self.weight_variable([512, 256], 'w2')
        b2 = self.bias_variable([256], 'b2')
        W3 = self.weight_variable([256, self.action_dim], 'w3')
        b3 = self.bias_variable([self.action_dim], 'b3')
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        h_layer1 = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        h_layer = tf.nn.relu(tf.matmul(h_layer1, W2) + b2)
        # Q Value layer
        self.Q_value = tf.matmul(h_layer, W3) + b3
        self.saver = tf.train.Saver()
        if not os.path.exists(path):
            os.makedirs(path)

    def SaveData(self, name):
        self.saver.save(self.session, path + '/model-' + str(name) + '.cptk')

    def create_training_method(self):
        self.action_input = tf.placeholder(
            "float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(
            tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state,
                                   done))
        self.batch_step += 1
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()
            self.batch_step = 0

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        rbatch = [n for n in self.replay_buffer if n[2] > 0]
        if len(rbatch) > 10:
            b1 = random.sample(rbatch, 10)
            b2 = random.sample(self.replay_buffer, BATCH_SIZE - 10)
            minibatch = b1 + b2
        else:
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(
            feed_dict={self.state_input: next_state_batch})
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(
                    reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(
            feed_dict={
                self.y_input: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch
            })

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 500000

    def action(self, state):
        return np.argmax(
            self.Q_value.eval(feed_dict={self.state_input: [state]})[0])

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, name=name)


# ---------------------------------------------------------
# Hyper Parameters
# ENV_NAME = 'CartPole-v1'
# EPISODE = 10000  # Episode limitation
# STEP = 300  # Step limitation in an episode
# TEST = 10  # The number of experiment test every 100 episode
# path = "./collect_dqn"  # The path to save our model to.

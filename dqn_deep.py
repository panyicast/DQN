# _*_coding:utf-8_*_
# Filename: DQN.py
from collections import deque
import os
import random
import numpy as np
import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Hyper Parameters for DQN
GAMMA = 0.99  # discount factor for target Q
INITIAL_EPSILON = 1  # starting value of epsilon
FINAL_EPSILON = 0.1  # final value of epsilon
REPLAY_SIZE = 1000000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch
TARGET_UPDATE_FREQUENCY = 10000  # the frequency the target network is updated
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4  # actions between SGD updates
LEARNING_RATE = 0.00025  # learning rae by RMSProp
GRADIENT_MOMENTUM = 0.95  # RMSProp
SQUARED_MOMENTUM = 0.95  # RMSProp
MIN_SQUARED_GRADIENT = 0.01  # RMSProp

FINAL_EXPLOR_FRAME = 1000000
REPLAY_START_SIZE = 50000
NOOP_MAX = 30
ACTION_DIM = 5  # numbers of actions


class DQN():
    """ This is function foo"""

    # DQN Agent
    def __init__(self):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.batch_step = 0

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        ''' network weights '''
        # 输入图像为84*84 的灰度图
        self.state_input = tf.placeholder(tf.float32, shape=(None, 84, 84, 1))
        # 第一卷积层，32 filters，8*8，stride 4，输出为20*20*64
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([8, 8, 1, 32], 'W_conv1')
            b_conv1 = bias_variable([32], 'b_conv1')
            conv1 = tf.nn.conv2d(
                self.state_input,
                W_conv1,
                strides=[1, 4, 4, 1],
                padding='VALID')
            h_conv1 = tf.nn.relu(conv1 + b_conv1)
        # 第二卷积层，64 filters，4*4，stride 2，输出为9*9*64
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([4, 4, 32, 64], 'W_conv2')
            b_conv2 = bias_variable([64], 'b_conv2')
            conv2 = tf.nn.conv2d(
                h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID')
            h_conv2 = tf.nn.relu(conv2 + b_conv2)
        # 第三卷积层，64 filters，3*3，stride 1，输出为7*7*64
        with tf.name_scope('conv3'):
            W_conv3 = weight_variable([3, 3, 64, 64], 'W_conv3')
            b_conv3 = bias_variable([64], 'b_conv3')
            conv3 = tf.nn.conv2d(
                h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            h_conv3 = tf.nn.relu(conv3 + b_conv3)
        # 第四卷积层，512 filters，7*7，stride 1，输出为1*1*512
        with tf.name_scope('conv4'):
            W_conv4 = weight_variable([7, 7, 64, 512], 'W_conv4')
            b_conv4 = bias_variable([512], 'b_conv4')
            conv4 = tf.nn.conv2d(
                h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='VALID')
            h_conv4 = tf.nn.relu(conv4 + b_conv4)
        # 全连接层，输出为512
        with tf.name_scope('fc'):
            fc = tf.nn.relu(tf.contrib.layers.flatten(h_conv4))
        # 输出层，输出Q_value,维度为 ACTION_DIM
        with tf.name_scope('Q_value'):
            W_output = weight_variable([512, ACTION_DIM], 'W_output')
            b_output = bias_variable([ACTION_DIM], 'b_output')
            self.Q_value = tf.matmul(fc, W_output) + b_output

        self.saver = tf.train.Saver()

    def SaveData(self, name):
        if not os.path.exists(path):
            os.makedirs(path)
        self.saver.save(self.session, path + '/model-' + str(name) + '.cptk')

    def create_training_method(self):
        self.action_input = tf.placeholder(
            "float", [None, ACTION_DIM])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(
            tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # self.optimizer=tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.optimizer = tf.train.RMSPropOptimizer(
            LEARNING_RATE, GRADIENT_MOMENTUM).minimize(self.cost)

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
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        # rbatch = [n for n in self.replay_buffer if n[2] > 0]
        # if len(rbatch) > 10:
        #     b1 = random.sample(rbatch, 10)
        #     b2 = random.sample(self.replay_buffer, BATCH_SIZE - 10)
        #     minibatch = b1 + b2
        # else:
        #     minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
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
        print('Q value : ', Q_value)
        if random.random() <= self.epsilon:
            return random.randint(0, ACTION_DIM - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= \
            (INITIAL_EPSILON - FINAL_EPSILON) / FINAL_EXPLOR_FRAME

    def action(self, state):
        return np.argmax(
            self.Q_value.eval(feed_dict={self.state_input: [state]})[0])


def weight_variable(shape, name=''):
    initial = tf.truncated_normal(shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)


def bias_variable(shape, name=''):
    initial = tf.constant(0.001, shape=shape)
    if name:
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial)

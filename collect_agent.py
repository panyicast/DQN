from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib import point
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb
import tensorflow as tf

import time
import random
import numpy as np
import pdb
from DQN import *

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_SELECTED = features.SCREEN_FEATURES.selected.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_TERRAN_MARINE = 48
_NEUTRAL_MINERALFIELD = 341

# Parameters
_PLAYER_SELF = 1
_Neutral = 3
_SUPPLY_USED = 3
_SUPPLY_MAX = 4
_NOT_QUEUED = [0]
_QUEUED = [1]
STATE_SPACE = 12
ACTION_SPACE = 9
ACTIONS = ('NOOP', 'UP', 'DOWN', 'LEFT', 'RIGHT')
MOVE_STEP = 8


class Agent(base_agent.BaseAgent):
    selected = False
    targeted = False
    marine_flag = False
    marine_pos = np.zeros(2)
    last_state = np.zeros(42)
    new_state = np.zeros(42)
    last_action = 0
    last_score = 0
    last_reward = 0
    stepd = 0
    screensize = [0, 0]
    player_position = [0, 0]

    def __init__(self):
        super(Agent, self).__init__()
        self.dqn = DQN(42, 5, 256)

    def SelectMarine(self, obs):
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()
        if self.marine_flag:
            target = [unit_x[0], unit_y[0]]
        else:
            target = [unit_x[10], unit_y[10]]
        self.selected = True
        self.marine_flag = not self.marine_flag
        marine_pos = target
        return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

    def mineralfield(self, obs):
        unit_type = obs.observation["screen"][_PLAYER_RELATIVE]
        unit_y, unit_x = (unit_type == _Neutral).nonzero()
        # self.mineralfield = zip(unit_x, unit_y)
        return list(zip(unit_x, unit_y))

    def mfield(self, obs):
        unit_type = obs.observation["screen"][_PLAYER_RELATIVE]
        unit_y, unit_x = (unit_type == _Neutral).nonzero()
        return (unit_x, unit_y)

    def nearest(self, obs):
        mineral = self.mineralfield(obs)
        dis = [self.distance(self.marine_pos, m) for m in mineral]
        i = np.argmin(dis)
        return mineral[i]

    def distance(self, d1, d2):
        return (d1[0] - d2[0]) * (d1[0] - d2[0]) + (d1[1] - d2[1]) * (
            d1[1] - d2[1])

    def getState(self, obs):
        player_position = obs.observation['screen'][_SELECTED]
        mx, my = self.mfield(obs)
        unit_y, unit_x = player_position.nonzero()
        state = np.zeros(42) - 1
        state[0] = int(np.mean(unit_x))
        state[1] = int(np.mean(unit_y))
        n = int(len(mx) / 12)
        for i in range(n):
            state[2 * (i + 1)] = mx[5 + i * 12]
            state[2 * (i + 1) + 1] = my[5 + i * 12]
        return state

    def GetScreenSize(self, obs):
        ss = obs.observation['screen']
        x = ss.shape[-1]
        y = ss.shape[-2]
        return x, y

    def GetAction(self, id):
        target = [0, 0]
        player = self.player_position

        if id == 0:
            return actions.FunctionCall(_NOOP, [])
        elif id == 1:
            target[0] = player[0]
            target[1] =
            player[1] - MOVE_STEP if player[1] - MOVE_STEP >= 0 else 0
        elif id == 2:
            target[0] = player[0]
            target[1] =
            player[1] + MOVE_STEP if player[1] + MOVE_STEP < 84 else 83
        elif id == 3:
            target[0] =
            player[0] - MOVE_STEP if player[0] - MOVE_STEP >= 0 else 0
            target[1] = player[1]
        elif id == 4:
            target[0] =
            player[0] + MOVE_STEP if player[0] + MOVE_STEP < 84 else 83
            target[1] = player[1]
        else:
            target = player

        return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])

    def step(self, obs):
        super(Agent, self).step(obs)
        self.stepd += 1

        # time.sleep(0.01)
        s = obs.observation["single_select"]

        if not s.any():
            self.screensize = self.GetScreenSize(obs)
            return self.SelectMarine(obs)

        self.stepd = self.stepd + 1
        score = obs.observation['score_cumulative'][0]
        reward = score - self.last_score
        if score < self.last_score:
            done = True
        else:
            done = False
        self.last_score = score
        self.last_reward = reward
        state = self.getState(obs)
        # print('state = ',state)
        self.dqn.perceive(self.last_state, self.last_action, reward, state,
                          done)
        self.last_state = state
        action = self.dqn.egreedy_action(state)  # e-greedy action for train
        if self.stepd % 100 == 0:
            print('==step(%d),score(%d),reward(%d),action(%s)' %
                  (self.stepd, score, reward, ACTIONS[action]))
        self.last_action = action
        # pdb.set_trace()
        self.player_position = state[:2]
        if self.stepd % 10000 == 0:
            d = self.dqn.SaveData(self.stepd)
            print('...data saved!')

        return self.GetAction(action)

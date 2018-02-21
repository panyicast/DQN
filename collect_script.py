from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features

import time
import random
import numpy as np

# Functions
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_MOVE_SCREEN =  actions.FUNCTIONS.Move_screen.id
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

class Agent(base_agent.BaseAgent):
    selected = False
    targeted = False
    marine_flag = False
    marine_pos = np.zeros(2)


  
    def SelectMarine(self,obs):
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

    def mineralfield(self,obs):
        unit_type = obs.observation["screen"][_PLAYER_RELATIVE]
        unit_y, unit_x = (unit_type == _Neutral).nonzero()
        # self.mineralfield = zip(unit_x, unit_y)
        return list(zip(unit_x, unit_y))
        

    def mfield(self,obs):
        unit_type = obs.observation["screen"][_PLAYER_RELATIVE]
        unit_y, unit_x = (unit_type == _Neutral).nonzero()
        # self.mineralfield = zip(unit_x, unit_y)
        #return list(zip(unit_x, unit_y))
        return (unit_x, unit_y)
    
    def nearest(self,obs):
        mineral = self.mineralfield(obs)
        dis = [self.distance(self.marine_pos,m) for m in mineral]
        i = np.argmin(dis)
        return mineral[i]
        
   
    def distance(self,d1,d2):
        return (d1[0]-d2[0])*(d1[0]-d2[0]) + (d1[1]-d2[1])*(d1[1]-d2[1])
    
    def step(self, obs):
        super(Agent, self).step(obs)
        time.sleep(0.05)
        s = obs.observation["single_select"]
        
        if not self.selected:
            return self.SelectMarine(obs)

        unit_score = obs.observation['score_cumulative']
        print('score : ',unit_score)
        player_position = obs.observation['screen'][_SELECTED]
        unit_y, unit_x = player_position.nonzero()
        mx, my = self.mfield(obs)
  
        self.targeted = False
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            if self.selected:
                mtarget = self.mineralfield(obs)
                if not self.marine_flag:
                    target = mtarget[0]
                else:
                    target = mtarget[-1]

                return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])


        return actions.FunctionCall(_NOOP, [])

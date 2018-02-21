from dqn_deep import *
import numpy as np
import time


def gettime(state):
    time_start = time.time()
    b = d.egreedy_action(state)
    time_end = time.time()
    print('totally cost = ', time_end-time_start)
    return b

d = DQN()
state = np.zeros([84, 84, 1])
state = np.random.randint(0, 3, [84, 84, 1])
a = d.action(state)
print('action = ', a)

state = np.random.randint(0, 3, [84, 84, 1])
a = d.action(state)
print('action = ', a)

state = np.random.randint(0, 3, [84, 84, 1])
a = d.action(state)
print('action = ', a)

print('greed action =', gettime(state))
print('greed action =', gettime(state))
print('greed action =', gettime(state))
print('greed action =', gettime(state))
print('greed action =', gettime(state))
print('greed action =', gettime(state))

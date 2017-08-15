import tensorflow as tf
import numpy as np
from collections import namedtuple
import random

class PartialRollout(object):

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False

    def add(self, state, action, reward, value, terminal):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal

Batch = namedtuple("Batch", ["si", "actions", "advantages", "target_R", "terminal"])

def process_rollout(rollout, gamma):

    batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)
    next_values = rollout.values + [rollout.r]
    next_values = next_values[1:]

    rollout.rewards.reverse()
    rollout.values.reverse()
    next_values.reverse()

    batch_td = []
    batch_target_R = []

    for (ri, current_vi, next_vi) in zip(rollout.rewards, rollout.values, next_values):
        rollout.r = ri + 0.99 * rollout.r
        td = rollout.r - current_vi
        batch_td.append(td)
        batch_target_R.append(rollout.r)

    batch_td.reverse()
    batch_target_R.reverse()

    batch_td = np.reshape(np.asarray(batch_td), (len(batch_td), ))
    batch_target_R = np.reshape(np.asarray(batch_target_R),(len(batch_target_R), ))

    return Batch(batch_si, batch_a, batch_td, batch_target_R, rollout.terminal)


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
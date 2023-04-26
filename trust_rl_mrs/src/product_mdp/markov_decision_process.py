#! /usr/bin/env python2

import numpy as np


class MDP:
    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, state_features, state_actions, trans_rewards, state_labels, init_state, alpha=0.75, gamma=0.6):
        # fixed parameters
        self.gamma = gamma
        self.alpha = alpha

        # parameters from input
        self.states = state_actions.keys()
        self.actions = state_actions
        self.trans_rewards = trans_rewards
        self.state_features = state_features
        self.state_labels = state_labels
        self.init_state = init_state

    # obtain next state
    def get_next_state(self, curr_state, action):
        return self.trans_rewards[(curr_state, action)][0]

    # obtain the reward (s,a,s'): ONLY consider the deterministic transition
    def get_reward(self, curr_state, action, next_state):
        state_action = (curr_state, action)
        return self.trans_rewards[state_action][2]

#! /usr/bin/env python2

import numpy as np


class AbstractMap:
    def __init__(self, crop_topLeft, crop_bottomRight, cell_height=25.0, cell_width=25.0):
        """
        # image ---->col       state/location ---->x
        #      |                             |
        #      | row                         | y
        """
        self.height, self.width = crop_bottomRight[0] - crop_topLeft[0], crop_bottomRight[1] - crop_topLeft[1]  # image
        self.Rows, self.Cols = int(self.height / cell_height), int(self.width / cell_width)
        self.init_pos_y, self.init_pos_x = int(crop_topLeft[0] / cell_height), int(crop_topLeft[1] / cell_width)
        self.actions = {"E": [1, 0], "S": [0, 1], "W": [-1, 0], "N": [0, -1], "c": [0, 0]}

    def generate_states_locations(self):
        states_locations = {}
        for row in range(0, self.Rows):
            for col in range(0, self.Cols):
                state = row * self.Cols + col + 1
                state = "s" + str(state)
                states_locations[state] = (self.init_pos_x + col, self.init_pos_y + row)
                print state, ":", states_locations[state]

        return states_locations

    def generate_state_attributes(self, state_labels):
        state_features = {}
        state_actions = {}
        transitions = {}
        rewards = {}

        for row in range(0, self.Rows):
            for col in range(0, self.Cols):
                state = row * self.Cols + col + 1
                state = "s" + str(state)

                state_features[state] = [0.0, 0.0]

                if state not in state_labels.keys():
                    state_labels[state] = []

                state_actions[state] = []
                for act in self.actions:
                    new_row, new_col = self.actions[act][1] + row, self.actions[act][0] + col
                    if 0 <= new_row < self.Rows and 0 <= new_col < self.Cols:
                        state_actions[state].append(act)
                        new_state = new_row * self.Rows + new_col + 1
                        new_state = "s" + str(new_state)
                        transitions[(state, act)] = [new_state, 1.0, -np.inf]
        # print state_features, state_actions, transitions, state_labels

        return state_features, state_actions, transitions, state_labels

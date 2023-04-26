#! /usr/bin/env python2

import numpy as np


class AbstractMap:
    def __init__(self, crop_topLeft, crop_bottomRight, cell_height, cell_width, img_width=500, img_height=500,
                 env_width=500.0, env_height=500.0):
        """ Crop a target environment from a global map and define its cell size

        Args:
            crop_topLeft        (tuple) top-left pixel position of the target environment in the global map
            crop_bottomRight    (tuple) bottom-right pixel position of the target environment in the global map
            cell_height         (int) height of every cell in the target environment
            cell_width          (int) width of every cell in the target environment
            img_width (int)     image width of global map
            img_height (int)    image height of global map
            env_width (float)   width of global environment
            env_height (float)  height of global environme

        Note:
        # image ---->col       state/location ---->x
        #      |                             |
        #      | row                         | y

        """
        self.img_width = img_width
        self.img_height = img_height
        self.env_width = env_width
        self.env_height = env_height
        self.height, self.width = crop_bottomRight[0] - crop_topLeft[0], crop_bottomRight[1] - crop_topLeft[1]  # image
        self.Rows, self.Cols = int(self.height / cell_height), int(self.width / cell_width)
        self.init_pos_y, self.init_pos_x = int(crop_topLeft[0] / cell_height), int(crop_topLeft[1] / cell_width)
        self.actions = {"E": [1, 0], "S": [0, 1], "W": [-1, 0], "N": [0, -1], "c": [0, 0]}

    def init_state_attributes(self, state_labels):
        """ Initialize the abstracted states of the target environment: location, feature, label, actions,
            actions-rewards

        Args:
            state_labels (dict): atomic propositions (predefined)

        Returns:
            The dictionary pf state map to location, feature, label, actions, actions-rewards
        """
        states_locations = {}
        state_features = {}
        state_actions = {}
        state_actions_rewards = {}

        for row in range(0, self.Rows):
            for col in range(0, self.Cols):
                state = row * self.Cols + col + 1
                state = "s" + str(state)

                states_locations[state] = (self.init_pos_x + col, self.init_pos_y + row)
                state_features[state] = [0.0, 0.0]
                if state not in state_labels.keys():
                    state_labels[state] = []

                state_actions[state] = []
                state_actions_rewards[state] = {}
                for act in self.actions:
                    new_row, new_col = self.actions[act][1] + row, self.actions[act][0] + col
                    if 0 <= new_row < self.Rows and 0 <= new_col < self.Cols:
                        state_actions[state].append(act)
                        new_state = new_row * self.Rows + new_col + 1
                        new_state = "s" + str(new_state)
                        state_actions_rewards[state][act] = [new_state, 1.0, -np.inf]
        # print states_locations, state_features, state_labels, state_actions, state_actions_rewards

        state_attributes = {"state_features": state_features, "state_actions": state_actions,
                            "state_actions_rewards": state_actions_rewards, "state_labels": state_labels,
                            "states_locations": states_locations}

        return state_attributes

    def imgpos2envpos(self, imgpos):
        """convert image pixel position into environment geological position
        Args:
            imgpos (tuple)      pixel position in the global map

        Returns:
            converted geological position of image position
        """
        transMat = np.array([[self.img_width / 2.0], [self.img_height / 2.0]])
        scalMat = np.array([[self.env_width / self.img_width, 0], [0, -self.env_height / self.img_height]])
        array1 = imgpos - transMat
        envpos = np.dot(scalMat, array1)
        return envpos

    def envpos2imgpos(self, envpos):
        """convert image pixel position into environment geological position
            Args:
                envpos (tuple)      position in the global environment

            Returns:
                converted image position of geological position
        """
        transMat = np.array([[self.img_width / 2.0], [self.img_height / 2.0]])
        scalMat = np.array([[self.env_width / self.img_width, 0], [0, -self.env_height / self.img_height]])
        array1 = np.dot(np.linalg.inv(scalMat), np.array([envpos]).T)
        imgpos = array1 + transMat
        return imgpos.flatten().astype(int)

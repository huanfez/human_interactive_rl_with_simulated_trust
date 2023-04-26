#! usr/bin/env python2

import numpy as np
from scipy import ndimage
import tifffile as tf
from env_abstraction import envAbstract as envabs, setting_environment as env, setup_learning_env2 as setlearn


def update_environment(states_locations, iterat, cell_height=setlearn.cell_height, cell_width=setlearn.cell_width):
    cell_dict = {}

    traversability1 = env.gis_traversability
    r1_traversability = ndimage.maximum_filter(traversability1, (cell_height - 1, cell_width - 1))

    visibility1 = env.gis_visibility
    r1_visibility = ndimage.percentile_filter(visibility1, 50, (cell_height - 1, cell_width - 1))

    for loc in states_locations.values():
        # print state_location
        index_x, index_y = loc[0] * cell_width + cell_width // 2, loc[1] * cell_height + cell_height // 2
        cell_dict[loc] = np.array([[r1_traversability[index_y][index_x], r1_visibility[index_y][index_x]]])

    return cell_dict


if __name__ == '__main__':
    abstracted_map = envabs.AbstractMap(setlearn.crop_topLeft, setlearn.crop_bottomRight, setlearn.cell_height,
                                        setlearn.cell_width)
    env_state_location_map = abstracted_map.generate_states_locations()
    environment_dict = update_environment(env_state_location_map, 0)
    print "filtered environment information", environment_dict


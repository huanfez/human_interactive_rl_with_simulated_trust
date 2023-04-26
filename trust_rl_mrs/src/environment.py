#! /usr/bin/env python

import rospkg
import numpy as np
import tifffile as tf
import pygraphviz
from networkx.drawing import nx_agraph

from env_abstraction import map
from product_mdp import automaton as aut, mdp, productmdp as pm


def get_map_dir(pkg_name='trust_rl_mrs'):
    rospack = rospkg.RosPack()
    pkg_dir = rospack.get_path(pkg_name)  # get the file path for rospy_tutorials
    map_dir = pkg_dir + '/src/map'  # map directory

    return map_dir


def get_data_dir(pkg_name='trust_rl_mrs'):
    rospack = rospkg.RosPack()
    pkg_dir = rospack.get_path(pkg_name)  # get the file path for rospy_tutorials
    data_dir = pkg_dir + '/src/data'  # data directory

    return data_dir


def get_init_trust_map_dir(trust_file_name='/trust_temp.tif'):
    trust_temp_dir = get_data_dir() + trust_file_name
    return trust_temp_dir


def get_gis_traversability(slope_file_name='/slope_yazoo_500m.tif'):
    slope_dir = get_map_dir() + slope_file_name
    traversability_img = tf.imread(slope_dir)
    gis_traversability = np.copy(np.asarray(traversability_img))
    traversability = np.divide((1 - np.exp(gis_traversability * 100.0 - 1.8)),
                               (1 + np.exp(gis_traversability * 100.0 - 1.8)))

    return traversability


def get_gis_visibility(veg_height_file_name='/Minus_yazoo_500m.tif'):
    veg_height_dir = get_map_dir() + veg_height_file_name
    visibility_img = tf.imread(veg_height_dir)
    gis_visibility = np.copy(np.asarray(visibility_img))
    visibility = np.divide((1 - np.exp(gis_visibility - 1.5)), (1 + np.exp(gis_visibility - 1.5)))

    return visibility


def env_init_with_gis(traversability_file_name='/slope_yazoo_500m.tif', visibility_file_name='/Minus_yazoo_500m.tif'):
    init_traversability = get_gis_traversability(traversability_file_name)
    init_visibility = get_gis_visibility(visibility_file_name)
    tf.imwrite(get_data_dir() + '/traversability1_iter0.tif', init_traversability)
    tf.imwrite(get_data_dir() + '/traversability2_iter0.tif', init_traversability)
    tf.imwrite(get_data_dir() + '/traversability3_iter0.tif', init_traversability)
    tf.imwrite(get_data_dir() + '/visibility1_iter0.tif', init_visibility)
    tf.imwrite(get_data_dir() + '/visibility2_iter0.tif', init_visibility)
    tf.imwrite(get_data_dir() + '/visibility3_iter0.tif', init_visibility)


def get_prod_mdp_model(crop_topLeft, crop_bottomRight, cell_height, cell_width, state_label, mdp_init_state, dfa):
    abstracted_map = map.AbstractMap(crop_topLeft, crop_bottomRight, cell_height, cell_width)
    states_attributes = abstracted_map.init_state_attributes(state_label)

    lmdp = mdp.MDP(states_attributes["state_features"], states_attributes["state_actions"],
                   states_attributes["state_actions_rewards"], states_attributes["state_labels"], mdp_init_state)

    G = nx_agraph.from_agraph(pygraphviz.AGraph(dfa))  # buchi automaton to graph (python networkx lib)
    # nx_agraph.view_pygraphviz(G)  # visualize the graph
    buchi_automaton = aut.Automaton(G)  # to automaton (class)

    product_mdp_result = pm.product_mdp(lmdp, buchi_automaton)
    print sorted(product_mdp_result["state_actions"])

    product_mdp_result["states_locations"] = states_attributes["states_locations"]

    return product_mdp_result


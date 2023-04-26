#! /usr/bin/env python2
# coding=utf-8

import rospy
import re
import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
import random
import matplotlib.pyplot as plt
import tifffile as tf
import pygraphviz
from networkx.drawing import nx_agraph

from product_mdp import automaton as aut, ltl_dfa, markov_decision_process as mdp, product_mdp as pm
from query_strategy import cell_path_selector as cps2, q_learning as ql
from bayesian_inference import parameter4_gibbs as pmc4
import simulated_human3 as simh3
from plots import plots
from env_abstraction import envAbstract as envabs, setting_environment as env, setup_learning_env1 as setlearn


# BIC
def bic_score(Beta0_itr, Sigma0_itr, a0_itr, b0_itr, c0_itr, d0_itr, x_1toK_list, Z_1toK, y_1toK):
    mode_sigma_square1 = b0_itr / (a0_itr + 1)
    mode_sigma_square2 = d0_itr / (c0_itr + 1)
    mode_x_1toK = np.mean(x_1toK_list, axis=0)

    log_prob = 0.0
    step_num, robot_num = y_1toK.shape
    for step in range(0, step_num):
        for robot in range(0, robot_num):
            log_prob += -0.5 * np.log(2 * np.pi * mode_sigma_square2) - \
                        (y_1toK[step, robot] - mode_x_1toK[step, robot]) ** 2 / mode_sigma_square2 / 2.0

    # bic_score = -2.0*log_prob + (len(Beta0_itr) + 2.0) * np.log(len(y_1toK))
    return -log_prob


# reward function map
def reward_map(betas, itera, states_locations, Rows, Cols):
    """
        Args: dynamic_info - a tuple (125 * 125, 125 * 125) traversability & visibility matrices
              beats - an array of weights for [trarversability, visibility, bias]
              states_locations - a dictionary of mdp_state : grid_location
        Return: a 2-d array of 125 * 125 storing the value of reward r(s,a s') at s'
    """

    traversability1 = tf.imread(env.data_dir + '/traversability1_iter{0}.tif'.format(0))
    r1_traversability = ndimage.percentile_filter(traversability1, 50, (setlearn.cell_height, setlearn.cell_width))

    visibility1 = tf.imread(env.data_dir + '/visibility1_iter{0}.tif'.format(0))
    r1_visibility = ndimage.percentile_filter(visibility1, 50, (setlearn.cell_height, setlearn.cell_width))

    col_start, row_start = states_locations["s1"]
    rewards_matrix = np.zeros((Rows * setlearn.cell_height, Cols * setlearn.cell_width))
    for (col, row) in states_locations.values():
        img_row, img_col = row * setlearn.cell_height + setlearn.cell_height / 2, \
                           col * setlearn.cell_width + setlearn.cell_width / 2
        for iy in range(0, setlearn.cell_height):
            for ix in range(0, setlearn.cell_width):
                py = (row - row_start) * setlearn.cell_height + iy
                px = (col - col_start) * setlearn.cell_width + ix
                rewards_matrix[py][px] = betas[0] * r1_traversability[img_row][img_col] + \
                                                           betas[1] * r1_visibility[img_row][img_col] + betas[2]
    # print "max-min reward: ", max(rewards_matrix), min(rewards_matrix)
    return np.array(rewards_matrix)


# discrete cell info: N * N hashmap for traversability & visibility
def update_environment(states_locations, iterat, cell_height=setlearn.cell_height, cell_width=setlearn.cell_width):
    cell_dict = {}

    traversability1 = tf.imread(env.data_dir + '/traversability1_iter{0}.tif'.format(iterat))
    r1_traversability = ndimage.percentile_filter(traversability1, 50, (cell_height, cell_width))

    visibility1 = tf.imread(env.data_dir + '/visibility1_iter{0}.tif'.format(iterat))
    r1_visibility = ndimage.percentile_filter(visibility1, 50, (cell_height, cell_width))

    traversability2 = tf.imread(env.data_dir + '/traversability2_iter{0}.tif'.format(iterat))
    r2_traversability = ndimage.percentile_filter(traversability2, 50, (cell_height, cell_width))

    visibility2 = tf.imread(env.data_dir + '/visibility2_iter{0}.tif'.format(iterat))
    r2_visibility = ndimage.percentile_filter(visibility2, 50, (cell_height, cell_width))

    traversability3 = tf.imread(env.data_dir + '/traversability3_iter{0}.tif'.format(iterat))
    r3_traversability = ndimage.percentile_filter(traversability3, 50, (cell_height, cell_width))

    visibility3 = tf.imread(env.data_dir + '/visibility3_iter{0}.tif'.format(iterat))
    r3_visibility = ndimage.percentile_filter(visibility3, 50, (cell_height, cell_width))

    for loc in states_locations.values():
        # print state_location
        index_x, index_y = loc[0] * cell_width + cell_width // 2, loc[1] * cell_height + cell_height // 2
        cell_dict[loc] = np.array([[r1_traversability[index_y][index_x], r1_visibility[index_y][index_x]],
                                  [r2_traversability[index_y][index_x], r2_visibility[index_y][index_x]],
                                  [r3_traversability[index_y][index_x], r3_visibility[index_y][index_x]]])

    return cell_dict


# generate the product-MDP
def create_product_mdp(labeled_mdp, automaton_graph):
    G = nx_agraph.from_agraph(pygraphviz.AGraph(automaton_graph))  # buchi automaton to graph (python networkx lib)
    # nx_agraph.view_pygraphviz(G)  # visualize the graph
    buchi_automaton = aut.Automaton(G)  # to automaton (class)
    print "automaton initial state: ", buchi_automaton.init_state, "final states:", buchi_automaton.acc_states, "\n"

    # generate a product mdp
    product_mdp_result = pm.product_mdp(labeled_mdp, buchi_automaton)
    return product_mdp_result


# Mapping from mdp trajectory to 20 * 20 cell coordinate
def state2cell_path_converter(paths, states_locations):
    cell_path_list = [[extract_mdp_state(local_state, states_locations)[1] for local_state in path] for path in paths]
    # print "global view of path:", cell_path_list, "\n"
    return cell_path_list


# Extract mdp state from the product-mdp state
def extract_mdp_state(prod_state, states_locations):
    mdp_state = re.search(r'(.*)\|', prod_state).group(1)
    return mdp_state, states_locations[mdp_state]


if __name__ == '__main__':
    try:
        rospy.init_node('wayofpoints', anonymous=True)

        ''' List: record RL information '''
        iterations = 21  # number of iterations
        history_path_list = []
        posterior_list_beta = [np.copy(setlearn.Beta0_itr)]
        posterior_list_sigma = [np.copy(setlearn.Sigma0_itr)]
        trust_gains = []

        ''' Setup environment: MDP, ltl_f (DFA), product-MDP '''
        # env_state_location_map = setlearn.states_locations  # load an environment among 4 types terrain
        abstracted_map = envabs.AbstractMap(setlearn.crop_topLeft, setlearn.crop_bottomRight, setlearn.cell_height,
                                            setlearn.cell_width)
        env_state_location_map = abstracted_map.generate_states_locations()
        states_attributes = abstracted_map.generate_state_attributes(setlearn.states_label)

        mdp1 = mdp.MDP(states_attributes[0], states_attributes[1], states_attributes[2], states_attributes[3],
                       setlearn.mdp_init_state)  # read a labeled mdp based on the environment
        dfa_dotFormat = ltl_dfa.dfa_dotFormat  # read a buchi automaton

        product_result = create_product_mdp(mdp1, dfa_dotFormat)  # generate a product mdp

        ''' Synchronize information: environment information for RL '''
        # update environment: add traversability and visibility
        tf.imwrite(env.data_dir + '/traversability1_iter0.tif', env.r1_dynamic_traversability)
        tf.imwrite(env.data_dir + '/traversability2_iter0.tif', env.r2_dynamic_traversability)
        tf.imwrite(env.data_dir + '/traversability3_iter0.tif', env.r3_dynamic_traversability)
        tf.imwrite(env.data_dir + '/visibility1_iter0.tif', env.r1_dynamic_visibility)
        tf.imwrite(env.data_dir + '/visibility2_iter0.tif', env.r2_dynamic_visibility)
        tf.imwrite(env.data_dir + '/visibility3_iter0.tif', env.r3_dynamic_visibility)
        environment_dict = update_environment(env_state_location_map, 0)

        # update product-mdp's state: the associated environment attributes
        for pstate in product_result[0].keys():
            _, location = extract_mdp_state(pstate, env_state_location_map)
            product_result[0][pstate] = environment_dict[location][0]  # update trave. & vis. information

        ''' start an offline-like q-learning: many reward functions '''
        product_mdp1 = ql.RL(product_result[0], product_result[1], product_result[2], setlearn.alpha, setlearn.gamma)
        policy_dict, route_dict = product_mdp1.policy_route_distribution_bound(product_result[3], product_result[4],
                                                                               setlearn.Beta0_itr, setlearn.Sigma0_itr,
                                                                               0, 15000)
        # generate the true policy: because know the ture value of simulated data
        policy_true, route_true = product_mdp1.policy_route_distribution_mean(product_result[3], product_result[4],
                                                                              setlearn.beta_true3, 15000)
        rewards_mat = reward_map(setlearn.beta_true3[2:5], 0, env_state_location_map, abstracted_map.Rows,
                                 abstracted_map.Cols)
        plots.plot_policy(policy_true, rewards_mat, 0, env.data_dir)  # plot policy for mean beta

        # plot prior information of parameters
        plots.plot_distribution(setlearn.Beta0_itr, setlearn.Sigma0_itr, setlearn.beta_true3, 0, env.data_dir)

        ''' online updating: parameters & policy/trajectory '''
        policy_dict_new = {str(beta2_4): policy_dict[beta2_4] for beta2_4 in policy_dict.keys()}
        route_dict_new = {str(beta2_4): route_dict[beta2_4] for beta2_4 in route_dict.keys()}
        for iteration in range(1, iterations):
            # generate the probability of all the parameters being selected: with updated posterior
            beta2_4_list = []
            beta2_4_prob_list = []
            updated_distribution = multivariate_normal(mean=setlearn.Beta0_itr[2:], cov=setlearn.Sigma0_itr[2:, 2:])
            for beta2_4 in policy_dict.keys():
                beta2_4_list.append(str(beta2_4))
                beta2_4_prob_list.append(updated_distribution.pdf(np.array(beta2_4)))
            beta2_4_prob_list = [prob / np.sum(beta2_4_prob_list) for prob in beta2_4_prob_list]
            # print "parameter distribution: ", beta2_4_prob_list

            # sample a subset of model parameters values from the offline model with the parameters' posterior
            sample_size = 1 + int(550 * np.exp(-0.5 * iteration))
            num_nonzeros = [i for i in range(0, len(beta2_4_prob_list)) if beta2_4_prob_list[i] != 0.0]
            sample_size = min(len(policy_dict), sample_size, len(num_nonzeros))
            sampled_beta2_4 = np.random.choice(beta2_4_list, sample_size, replace=False, p=beta2_4_prob_list)
            # print "sampled beta: ", sampled_beta2_4
            sampled_policy_dict = {beta2_4: policy_dict_new[beta2_4] for beta2_4 in sampled_beta2_4}
            sampled_route_dict = {beta2_4: route_dict_new[beta2_4] for beta2_4 in sampled_beta2_4}

            # The list of unrepeated pool of policies and optimal trajectory
            route_list = [route for route in sampled_route_dict.values()]
            route_list = ql.optimal_route_set(route_list)
            policy_list = [policy for policy in sampled_policy_dict.values()]
            policy_list = ql.policy_set(policy_list)

            # plot the mean reward and policy
            rewards_mat = reward_map(setlearn.Beta0_itr[2:5], iteration, env_state_location_map, abstracted_map.Rows,
                                     abstracted_map.Cols)
            plots.plot_policy({}, rewards_mat, iteration, env.data_dir)  # plot policy for mean beta
            print "policy set size:", len(policy_list), "route set size:", len(route_list), "Rollout route list:", \
                route_list

            # 2. find the preferable path
            cell_path_array = state2cell_path_converter(route_list, env_state_location_map)  # convert state to cell
            optimo_cell_path = cps2.optimal_cell_path_mixture(cell_path_array, environment_dict)
            print 'iteration', iteration, ' human preferable path:', optimo_cell_path, "\n"
            history_path_list.append(optimo_cell_path)

            # 3. MCMC training:
            # 3.1. Read data: input and output variable of regression
            Z_1toK = cps2.cell_path_situational_awareness(optimo_cell_path, environment_dict)
            y_1toK = simh3.simulated_human_data(setlearn.beta_true3, setlearn.delta_w_true3, setlearn.delta_v_true3,
                                                Z_1toK)
            trust_gains.append(np.sum(y_1toK, axis=0))  # record for debugging

            # 2. gibbs sampler generates the posterior distribution
            samples = pmc4.iterated_sampling(Z_1toK[:, :, 2:], y_1toK, setlearn.Beta0_itr[2:],
                                             setlearn.Sigma0_itr[2:, 2:],
                                             setlearn.a0_itr, setlearn.b0_itr, iters=7000)

            # 2.1 get the hyper-parameters of posterior distribution
            """ 
            samples_x_1toK = samples[0], samples_Beta = samples[1], samples_delta_w_square = samples[2],
            samples_delta_v_square = samples[3] 
            """
            post_hyper_param = pmc4.mean_value_model_parameters(samples[0], samples[1])

            # 2.2 Set posterior to be prior
            """
            means_Beta = post_hyper_param[0], variance_Beta = post_hyper_param[1], 
            means_delta_w_square=post_hyper_param[2], variance_delta_w_square = post_hyper_param[3],
            means_delta_v_square = post_hyper_param[4], variance_delta_v_square = post_hyper_param[5]
            """
            setlearn.Beta0_itr[2:] = post_hyper_param[0]
            setlearn.Sigma0_itr[2:, 2:] = post_hyper_param[1]
            setlearn.a0_itr = 2 + post_hyper_param[2] ** 2 / post_hyper_param[3]
            setlearn.b0_itr = post_hyper_param[2] * (setlearn.a0_itr - 1)

            # collect the posterior distribution
            posterior_list_beta.append(np.copy(setlearn.Beta0_itr))
            posterior_list_sigma.append(np.copy(setlearn.Sigma0_itr))

            # update the plot of the posterior distribution
            # print "beta_posterior mean:", setlearn.Beta0_itr, "beta_posterior covariance:", setlearn.Sigma0_itr
            plots.plot_distribution(setlearn.Beta0_itr, setlearn.Sigma0_itr, setlearn.beta_true3, iteration + 1,
                                    env.data_dir)

        print "trust change:", trust_gains, "overall trust gains:", np.sum(np.array(trust_gains))

        # plot the credible interval of the posterior distribution through all the iterations
        # print "posterior list beta:", posterior_list_beta, "posterior list sigma:", posterior_list_sigma
        plots.plot_credile_interval(posterior_list_beta, posterior_list_sigma, setlearn.beta_true3, env.data_dir,
                                    iterations)

        plt.show()
    except rospy.ROSInterruptException:
        pass


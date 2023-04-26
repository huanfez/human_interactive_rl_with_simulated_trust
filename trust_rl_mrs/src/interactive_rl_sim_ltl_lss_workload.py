#! /usr/bin/env python2
# coding=utf-8

import rospy
import re
import numpy as np
from scipy import ndimage
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import tifffile as tf
import itertools
import copy

from product_mdp import ltl_dfa
from query_strategy import cell_path_selector as cps2, q_learning as ql
from bayesian_inference import parameter3_mcmc as pmc3
import simulated_human3 as simh3
from plots import plots
from env_abstraction import setting_environment as env, setup_learning_env4 as setlearn
import environment
from query_strategy import dynaq


def set_parametric_model(model, beta2_4_vals):
    param_model = copy.deepcopy(model)
    param_model["state_actions_rewards"] = {beta: param_model["state_actions_rewards"] for beta in beta2_4_vals}

    set_loc_features = {loc: [] for loc in model["states_locations"].values()}
    param_model, sensor_features = update_environment(param_model, beta2_4_vals, set_loc_features, 0)

    return param_model, sensor_features


def sample_beta2_4(interested_vals, posterior_mean, posterior_var, iteration_num):
    """
        Args:
            interested_vals(list): interested values of beta2_4
            posterior_mean(array): beta2_4 mean
            posterior_var(array): beta2_4 covariance
            iteration_num(int): interactive RL episode
        Return:
            sampled_beta2_4_vals(list): sampled beta2_4 in interested values
    """
    # generate the probability of all the parameters being selected: with updated posterior
    posterior_distribution = multivariate_normal(mean=posterior_mean, cov=posterior_var)
    beta2_4_logprob_list = [posterior_distribution.logpdf(np.array(beta2_4_val)) for beta2_4_val in interested_vals]
    # beta2_4_prob_list = [np.exp(logprob - np.log(np.sum(beta2_4_prob_list))) for logprob in beta2_4_logprob_list]
    # sum_probability = sum(np.exp(beta2_4_logprob_list))
    print "parameter distribution: ", beta2_4_logprob_list
    beta2_4_prob_list = [1.0 / sum(np.exp(np.array(beta2_4_logprob_list) - logprob)) for logprob in beta2_4_logprob_list]

    # sample a subset of model parameters values from the offline model with the parameters' posterior
    customized_sample_size = 1 + int(550 * np.exp(-0.3 * iteration_num))
    sample_size = np.min([np.count_nonzero(beta2_4_prob_list), 3])

    beta2_4_str_list = [str(beta2_4_val) for beta2_4_val in interested_vals]
    sampled_beta2_4_str = np.random.choice(beta2_4_str_list, sample_size, replace=False, p=beta2_4_prob_list)  # order

    beta2_4_dict = {str(beta2_4_val): beta2_4_val for beta2_4_val in interested_vals}
    sampled_beta2_4_vals = [beta2_4_dict[beta2_4_str] for beta2_4_str in sampled_beta2_4_str]

    return sampled_beta2_4_vals


def get_interested_beta2_4(betas_mean, betas_var):
    interval = np.array([1.96 * np.sqrt(element) for element in np.diag(betas_var)])
    betas_l = betas_mean - interval
    betas_u = betas_mean + interval
    beta0 = set(np.arange(betas_l[0], betas_u[0], 5.1))
    beta1 = set(np.arange(betas_l[1], betas_u[1], 5.1))
    beta2 = set([value for value in np.arange(betas_l[2], betas_u[2], 0.15) if -0.2 <= value <= 1.1])
    beta3 = set([value for value in np.arange(betas_l[3], betas_u[3], 0.15) if -0.2 <= value <= 1.1])
    beta4 = set([value for value in np.arange(betas_l[4], betas_u[4], 0.15) if -1.1 <= value <= 1.1])
    interested_valued = set(itertools.product(beta2, beta3, beta4))

    return list(interested_valued)


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
def update_environment(model, sampled_betas, sensor_features, iterat, cell_height=setlearn.cell_height, cell_width=setlearn.cell_width):
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

    for loc in sensor_features.keys():
        index_x, index_y = loc[0] * cell_width + cell_width // 2, loc[1] * cell_height + cell_height // 2
        sensor_features[loc] = np.array([[r1_traversability[index_y][index_x], r1_visibility[index_y][index_x]],
                                         [r2_traversability[index_y][index_x], r2_visibility[index_y][index_x]],
                                         [r3_traversability[index_y][index_x], r3_visibility[index_y][index_x]]])

    for pstate in model["state_features"].keys():
        _, location = extract_mdp_state(pstate, model["states_locations"])
        model["state_features"][pstate] = np.mean(sensor_features[location], axis=0)  # update trave. & vis. information

    for beta in sampled_betas:
        for state in model["state_actions_rewards"][beta].keys():
            for action in model["state_actions_rewards"][beta][state].keys():
                feature1 = np.array(model["state_features"][state]).T
                next_state = model["state_actions_rewards"][beta][state][action][0]
                feature2 = np.array(model["state_features"][next_state]).T
                model["state_actions_rewards"][beta][state][action][2] = np.matmul(np.array(beta[:2]), (feature1 + feature2) / 2.0) + beta[2]
                if next_state in model["acc_states"]:
                    model["state_actions_rewards"][beta][state][action][2] = 99.0 + np.matmul(np.array(beta[:2]), (feature1 + feature2) / 2.0) + beta[2]

                next_mdp_state = re.search(r'(.*)\|', next_state).group(1)
                next_mdp_state_label = setlearn.states_label.get(next_mdp_state)
                if next_mdp_state_label is not None and "obs" in next_mdp_state_label:
                    model["state_actions_rewards"][beta][state][action][2] = -99.0 + np.matmul(np.array(beta[:2]), (feature1 + feature2) / 2.0) + beta[2]
                elif next_mdp_state_label is not None and "neigh" in next_mdp_state_label:
                    model["state_actions_rewards"][beta][state][action][2] = 1.0 + np.matmul(np.array(beta[:2]), (feature1 + feature2) / 2.0) + beta[2]



    return model, sensor_features


# Extract mdp state from the product-mdp state
def extract_mdp_state(prod_state, states_locations):
    mdp_state = re.search(r'(.*)\|', prod_state).group(1)
    return mdp_state, states_locations[mdp_state]


if __name__ == '__main__':
    try:
        rospy.init_node('wayofpoints', anonymous=True)

        # List: record RL information
        episodes = 41  # number of iterations
        history_path_list = []
        posterior_list = [[np.copy(setlearn.Beta0_itr), np.copy(setlearn.Sigma0_itr)]]
        trust_gains = []

        # Setup model with gis environment: MDP, ltl_f (DFA), product-MDP
        environment.env_init_with_gis()  # get access to gis map
        init_model = environment.get_prod_mdp_model(setlearn.crop_topLeft, setlearn.crop_bottomRight,
                                                    setlearn.cell_height, setlearn.cell_width, setlearn.states_label,
                                                    setlearn.mdp_init_state, ltl_dfa.dfa0_dotFormat)
        # interested_betas = get_interested_beta2_4(setlearn.Beta0_itr, setlearn.Sigma0_itr)  # interested beta values
        interested_betas = [(0.4, 0.1, -0.6),
                            (0.1, 0.6, 0.1),
                            (0.6, 0.4, -0.13)]

        init_param_model, location_features = set_parametric_model(init_model, interested_betas)  # set parametric model

        # start q-learning: many reward functions
        agent_info = {"assumed_model": init_param_model["state_actions_rewards"], "actions": ["E", "S", "W", "N", "c"],
                      "acc_states": init_model["acc_states"],
                      "discount": 0.95, "step_size": 0.1, "epsilon": 0.1, "planning_steps": 10, 'random_seed': 35,
                      'planning_random_seed': 35}
        rl = dynaq.DynaQAgent(agent_info)  # set rl learner
        rl.planning_step(interested_betas)  # Initial multi-step planning
        # print "product-mdp: ", init_model["state_actions_rewards"]
        # print "q_values", rl.param_q_values
        print len(interested_betas), " interested beta values: ", interested_betas

        for beta2_4 in interested_betas:
            rewards_mat = reward_map(beta2_4, 0, init_model["states_locations"], 5, 5)
            plots.plot_policy({}, rewards_mat, interested_betas.index(beta2_4), env.data_dir)  # plot policy for mean beta

        # plot prior information of parameters
        plots.plot_distribution(setlearn.Beta0_itr, setlearn.Sigma0_itr, setlearn.beta_true3, interested_betas, 0, env.data_dir)

        ''' online updating: parameters & policy/trajectory '''
        acc_reward_dict = {beta2_4: [0.0] for beta2_4 in interested_betas}
        # policy_dict_new = {str(beta2_4): policy_dict[beta2_4] for beta2_4 in policy_dict.keys()}
        # route_dict_new = {str(beta2_4): route_dict[beta2_4] for beta2_4 in route_dict.keys()}
        for iteration in range(1, episodes):
            # get prior of beta, and sample one beta from interested beta
            prior_beta2_4_mean, prior_beta2_4_var = setlearn.Beta0_itr[2:], setlearn.Sigma0_itr[2:, 2:]
            sampled_beta2_4_vals = sample_beta2_4(interested_betas, prior_beta2_4_mean, prior_beta2_4_var, iteration)

            cell_path_list = []
            acc_reward = {beta2_4: 0.0 for beta2_4 in interested_betas}
            for sampled_beta2_4_ in sampled_beta2_4_vals:
                sampled_beta2_4 = [sampled_beta2_4_]

                # RL in one episode
                past_state = init_model["init_state"]  # state
                param_past_state, param_past_action = rl.agent_start(init_model["init_state"], sampled_beta2_4)  # action
                route = []
                while len(route) < 10:
                    # choose an action (randomly, but onley one action can use)
                    past_state = param_past_state.values()[0]  # past states of all params are same
                    past_action = rl.rand_generator.choice(param_past_action.values())
                    route.append(past_state)

                    # observe state & reward
                    """ real robot will need update model """
                    param_reward = {beta2_4: rl.param_model[beta2_4][past_state][past_action][2] for beta2_4 in sampled_beta2_4}
                    param_state = {beta2_4: rl.param_model[beta2_4][past_state][past_action][0] for beta2_4 in sampled_beta2_4}

                    # print "updated model:", rl.param_model[sampled_beta2_4[0]]
                    # print "state:", past_state, "candidate actions:", param_past_action.values(), "selected:", \
                    #     past_action, "next state:", param_state
                    # print "model:", rl.param_model[sampled_beta2_4[0]]
                    # print "q-values:", rl.param_q_values[sampled_beta2_4[0]]

                    # q-learning: update q-value and multiple steps of q-planning
                    if past_state in init_model["acc_states"]:
                        rl.agent_end(param_past_state, param_past_action, past_state, param_reward, sampled_beta2_4)  # q-learning
                        break
                    param_past_state, param_past_action = rl.agent_step(param_past_state, param_past_action, param_state, param_reward, sampled_beta2_4)

                    for beta2_4 in sampled_beta2_4:
                        acc_reward[beta2_4] += param_reward[beta2_4]

                # product-mdp-state route to cell path
                cell_path = [extract_mdp_state(loc_state, init_model["states_locations"])[1] for loc_state in route]
                cell_path_list.append(cell_path)
                print "epis:", iteration, " beta2_4:", sampled_beta2_4, "route:", route

            for beta2_4 in interested_betas:
                if beta2_4 not in sampled_beta2_4_vals:
                    acc_reward[beta2_4] = acc_reward_dict[beta2_4][iteration - 1]
                    # acc_reward[beta2_4] = 0.0
                acc_reward_dict[beta2_4].append(acc_reward[beta2_4])

            # 2. select the preferable path (need to modify this part)
            optimo_cell_path = cps2.optimal_cell_path_mixture(cell_path_list, location_features)
            # print 'iteration', iteration, ' human preferable path:', optimo_cell_path, "\n"
            # history_path_list.append(optimo_cell_path)

            # print "sampled beta: ", sampled_beta2_4
            # sampled_policy_dict = {beta2_4: policy_dict_new[beta2_4] for beta2_4 in sampled_beta2_4}
            # sampled_route_dict = {beta2_4: route_dict_new[beta2_4] for beta2_4 in sampled_beta2_4}

            # The list of unrepeated pool of policies and optimal trajectory
            # route_list = [route for route in sampled_route_dict.values()]
            # route_list = ql.optimal_route_set(route_list)
            # policy_list = [policy for policy in sampled_policy_dict.values()]
            # policy_list = ql.policy_set(policy_list)

            # plot the mean reward and policy
            # rewards_mat = reward_map(setlearn.Beta0_itr[2:5], iteration, env_state_location_map, abstracted_map.Rows,
            #                          abstracted_map.Cols)
            # plots.plot_policy({}, rewards_mat, iteration, env.data_dir)  # plot policy for mean beta
            # print "policy set size:", len(policy_list), "route set size:", len(route_list), "Rollout route list:", \
            #     route_list

            # 3. MCMC training:
            # 3.1. Read data: input and output variable of regression
            Z_1toK = cps2.cell_path_situational_awareness(optimo_cell_path, location_features)
            y_1toK = simh3.simulated_human_data(setlearn.beta_true3, setlearn.delta_w_true3, setlearn.delta_v_true3, Z_1toK)
            trust_gains.append(np.sum(y_1toK, axis=0))  # record for debugging

            # 2. gibbs sampler generates the posterior distribution
            samples = pmc3.iterated_sampling(y_1toK, Z_1toK, setlearn.Beta0_itr, setlearn.Sigma0_itr, setlearn.a0_itr,
                                             setlearn.b0_itr, setlearn.c0_itr, setlearn.d0_itr, setlearn.Alpha, iters=7000)

            # 2.1 get the hyper-parameters of posterior distribution
            # samples_x_1toK : samples[0], samples_Beta : samples[1], samples_delta_w_square : samples[2],
            # samples_delta_v_square : samples[3]
            post_hyper_param = pmc3.mean_value_model_parameters(samples[0], samples[1], samples[2], samples[3])

            # 2.2 Set posterior to be prior
            # means_Beta : post_hyper_param[0], variance_Beta : post_hyper_param[1],
            # means_delta_w_square: post_hyper_param[2], variance_delta_w_square : post_hyper_param[3],
            # means_delta_v_square : post_hyper_param[4], variance_delta_v_square : post_hyper_param[5]
            setlearn.Beta0_itr = post_hyper_param[0]
            setlearn.Sigma0_itr = post_hyper_param[1]
            setlearn.a0_itr = 2 + post_hyper_param[2] ** 2 / post_hyper_param[3]
            setlearn.b0_itr = post_hyper_param[2] * (setlearn.a0_itr - 1)
            setlearn.c0_itr = 2 + post_hyper_param[4] ** 2 / post_hyper_param[5]
            setlearn.d0_itr = post_hyper_param[4] * (setlearn.c0_itr - 1)

            # collect the posterior distribution
            posterior_list.append([np.copy(setlearn.Beta0_itr), np.copy(setlearn.Sigma0_itr)])

            # plot of the posterior distribution
            plots.plot_distribution(setlearn.Beta0_itr, setlearn.Sigma0_itr, setlearn.beta_true3, interested_betas, iteration,
                                    env.data_dir)

        # plot the credible interval of the posterior distribution through all the iterations
        print "posterior list beta:", posterior_list, "trust change:", trust_gains
        posterior_list_mean = [val[0] for val in posterior_list]
        posterior_list_var = [val[1] for val in posterior_list]
        plots.plot_credile_interval(posterior_list_mean, posterior_list_var, setlearn.beta_true3, env.data_dir, episodes)
        plots.plot_acc_reward(acc_reward_dict, env.data_dir)
        plt.show()
    except rospy.ROSInterruptException:
        pass

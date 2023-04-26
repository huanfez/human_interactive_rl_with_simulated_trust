#! /usr/bin/env python2
# coding=utf-8

import rospy
from gazebo_msgs.srv import GetLinkState
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped

import re
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.stats import norm

from sklearn.neighbors import KNeighborsClassifier
import time

import pygraphviz
from networkx.drawing import nx_agraph

from src.trust_rl_mrs.src.product_mdp import automaton as aut, markov_decision_process as mdp
from src.trust_rl_mrs.src.env_abstraction import setting_environment as env
import product_mdp as pm
from src.trust_rl_mrs.src.query_strategy import q_learning as ql, q_learning_environment as qle


# use gazebo info to locate robot
def update_odom():
    rospy.wait_for_service('/gazebo/get_link_state')
    try:
        get_husky_base_link_state = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        husky_base_link_state = get_husky_base_link_state("/::base_link", "world")

        # publisher: synchronize gazebo info to odom value
        set_huksy_odom_state = rospy.Publisher('/set_pose', PoseWithCovarianceStamped, queue_size=10)
        huksy_reset_state = PoseWithCovarianceStamped()
        huksy_reset_state.header.frame_id = 'odom'
        huksy_reset_state.pose.pose = husky_base_link_state.link_state.pose
        set_huksy_odom_state.publish(huksy_reset_state)
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


# reset gazebo environment
def reset_robot_simulation():
    rospy.wait_for_service('/gazebo/reset_world')
    try:
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset = reset_world()
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e

    update_odom()  # reset odometer


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
    bic_score = -log_prob
    return bic_score


# plot and save figures for the posterior distribution of beata in ith iteration
def plot_distribution(Beta0, Sigma0, beta_true, iter_th, dir):
    fig, ax = plt.subplots(1, figsize=(5, 3))
    fig.tight_layout()
    x = np.linspace(norm.ppf(0.1), norm.ppf(0.99), 1000)
    ax.plot(x, norm.pdf(x, Beta0[0], np.sqrt(Sigma0[0, 0])), 'r-', lw=1, alpha=0.9, label=r'$\beta_{-1}$')
    ax.plot(x, norm.pdf(x, Beta0[1], np.sqrt(Sigma0[1, 1])), 'g-', lw=1, alpha=0.9, label=r'$\beta_0$')
    ax.plot(x, norm.pdf(x, Beta0[2], np.sqrt(Sigma0[2, 2])), 'b-', lw=1, alpha=0.9, label=r'$\beta_1$')
    ax.plot(x, norm.pdf(x, Beta0[3], np.sqrt(Sigma0[3, 3])), 'y-', lw=1, alpha=0.9, label=r'$\beta_2$')
    ax.plot(x, norm.pdf(x, Beta0[4], np.sqrt(Sigma0[4, 4])), 'c-', lw=1, alpha=0.9, label=r'$b$')
    ax.legend(loc='upper right', frameon=False)
    # ax.set_ylim([1.26e-3, 1.265e-3])
    ax.ticklabel_format(style='sci')
    # ax[0, 0].set_xlim([-400, 400])
    ax.set_ylabel('PDF')
    ax.set_title('Prior')

    ax.axvline(x=beta_true[0], color='r', linestyle='-.', lw=1.5)
    ax.axvline(x=beta_true[1], color='g', linestyle='-.', lw=1.5)
    ax.axvline(x=beta_true[2], color='b', linestyle='-.', lw=1.5)
    ax.axvline(x=beta_true[3], color='y', linestyle='-.', lw=1.5)
    ax.axvline(x=beta_true[4], color='c', linestyle='-.', lw=1.5)

    plt.pause(1e-17)
    fig.savefig(dir + '/sim{}_posterior.tif'.format(iter_th), dpi=100, bbox_inches="tight")


# plot the credible interval of all the ith posterior distribution
def plot_credile_interval(posterior_beta_list, posterior_sigma_list, beta_true, dir, num_itrs):
    posterior_list_beta_lb = []
    posterior_list_beta_ub = []
    length = len(posterior_beta_list)
    for index in range(0, length):
        beta_lb = posterior_beta_list[index] - 1.98 * np.sqrt(posterior_sigma_list[index].diagonal())
        beta_ub = posterior_beta_list[index] + 1.98 * np.sqrt(posterior_sigma_list[index].diagonal())
        posterior_list_beta_lb.append(beta_lb)
        posterior_list_beta_ub.append(beta_ub)

    iter_num = range(0, num_itrs)
    fig2, ax2 = plt.subplots(5, 1)
    fig2.tight_layout()
    for i in range(0, 5):
        ax2[i].fill_between(iter_num, np.array(posterior_list_beta_lb)[:, i], np.array(posterior_list_beta_ub)[:, i],
                            color='g', alpha=0.6, interpolate=True)
        ax2[i].plot(iter_num, np.ones(num_itrs) * beta_true[i], color='k')
        ax2[i].set_xlabel('Trial number')
        ax2[i].set_xticks(range(0, num_itrs))
    ax2[0].set_ylabel(r'$\beta_{-1}$')
    ax2[1].set_ylabel(r'$\beta_{0}$')
    ax2[2].set_ylabel(r'$\beta_{1}$')
    ax2[3].set_ylabel(r'$\beta_{2}$')
    ax2[4].set_ylabel(r'$b$')
    fig2.savefig(dir + '/convergence.tif', dpi=300, bbox_inches="tight")


# plot and save figures for the posterior distribution of beata in ith iteration
def plot_policy(policies, iter_th, dir):
    fig, ax = plt.subplots(1, figsize=(5, 5))
    fig.tight_layout()
    plt.gca().invert_yaxis()

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 126, 25)
    # minor_ticks = np.arange(0, 126, 5)

    ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    # ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    for stat in policies.keys():
        acts_probs = policies[stat]
        mdp_state = re.search(r'(.*)\|', stat).group(1)  # regular expression to get the mdp state
        val_of_mdp_state = int(mdp_state[1:]) - 1
        local_pos_x, local_pos_y = val_of_mdp_state % 5, val_of_mdp_state / 5
        plot_pos_x, plot_pos_y = local_pos_x * 25 + 12, local_pos_y * 25 + 12

        if acts_probs[0] > 0.0:
            plt.arrow(plot_pos_x, plot_pos_y, 22, 0, length_includes_head=True, head_width=2, head_length=5, color="r")

        if acts_probs[1] > 0.0:
            plt.arrow(plot_pos_x, plot_pos_y, 0, 22, length_includes_head=True, head_width=2, head_length=5, color="g")

        if acts_probs[2] > 0.0:
            plt.arrow(plot_pos_x, plot_pos_y, -22, 0, length_includes_head=True, head_width=2, head_length=5, color="b")

        if acts_probs[3] > 0.0:
            plt.arrow(plot_pos_x, plot_pos_y, 0, -22, length_includes_head=True, head_width=2, head_length=5, color="y")

    plt.pause(1e-17)
    fig.savefig(dir + '/sim{}_policy.tif'.format(iter_th), dpi=100, bbox_inches="tight")


# discrete cell info: N * N hashmap for traversability & visibility
def update_environment(states_locations, dynamic_info_r1, dynamic_info_r2, dynamic_info_r3,
                       cell_height=env.cell_height, cell_width=env.cell_width):
    cell_dict = {}

    r1_traversability = ndimage.uniform_filter(dynamic_info_r1[0], (cell_height, cell_width))
    r1_visibility = ndimage.uniform_filter(dynamic_info_r1[1], (cell_height, cell_width))
    r2_traversability = ndimage.uniform_filter(dynamic_info_r2[0], (cell_height, cell_width))
    r2_visibility = ndimage.uniform_filter(dynamic_info_r2[1], (cell_height, cell_width))
    r3_traversability = ndimage.uniform_filter(dynamic_info_r3[0], (cell_height, cell_width))
    r3_visibility = ndimage.uniform_filter(dynamic_info_r3[1], (cell_height, cell_width))

    for location in states_locations.values():
        # print state_location
        index_x, index_y = location[0] * cell_width + cell_width // 2, location[1] * cell_height + cell_height // 2
        cell_dict[location] = np.array(
            [[r1_traversability[index_y][index_x], r1_visibility[index_y][index_x]],
             [r2_traversability[index_y][index_x], r2_visibility[index_y][index_x]],
             [r3_traversability[index_y][index_x], r3_visibility[index_y][index_x]]])

    return cell_dict


# discrete cell trust: N * N hashmap for trust
def create_trust_map(grid_height=env.grid_height, grid_width=env.grid_width):
    trust_dict = {}
    for cy in range(0, grid_height):
        for cx in range(0, grid_width):
            trust_dict[(cx, cy)] = np.array([0.0, 0.0, 0.0])
    return trust_dict


# generate the product-MDP
def create_product_mdp(labeled_mdp, automaton_graph):
    G = nx_agraph.from_agraph(pygraphviz.AGraph(automaton_graph))  # buchi automaton to graph (python networkx lib)
    nx_agraph.view_pygraphviz(G)  # visualize the graph
    buchi_automaton = aut.Automaton(G)  # to automaton (class)
    print "automaton initial state: ", buchi_automaton.init_state, "final states:", buchi_automaton.acc_states, "\n"

    # generate a product mdp
    product_mdp_result = pm.product_mdp(labeled_mdp, buchi_automaton)
    return product_mdp_result


# Mapping from mdp trajectory to 20 * 20 cell coordinate
def state2cell_path_converter(paths, states_locations):
    cell_path_list = []
    for path in paths:
        cell_path = []
        for local_state in path:
            mdp_state = re.search(r'(.*)\|', local_state).group(1)  # regular expression to get the mdp state
            cell_path.append(states_locations[mdp_state])

        cell_path_list.append(cell_path)

    print "global view of path:", cell_path_list, "\n"
    return cell_path_list


if __name__ == '__main__':
    try:
        rospy.init_node('wayofpoints', anonymous=True)

        env_state_location_map = env.states_locations3  # assign a map (in total: 4)

        # read a labeled mdp based on the environment
        mdp_init_state = "s2"
        mdp1 = mdp.MDP(qle.states_features, qle.states_actions, qle.trans_rew, qle.states_label1, mdp_init_state)

        # read a buchi automaton
        # dotFormat = """
        # digraph "Fdest & G!obs" {
        #   rankdir=LR
        #   label="\n[Büchi]"
        #   labelloc="t"
        #   node [shape="circle"]
        #   I [label="", style=invis, width=0]
        #   I -> 1
        #   0 [label="0", peripheries=2]
        #   0 -> 0 [label="!obs"]
        #   1 [label="1"]
        #   1 -> 0 [label="dest & !obs"]
        #   1 -> 1 [label="!dest & !obs"]
        # }
        # """

        # dotFormat = """
        # digraph "F(green | red)" {
        #   rankdir=LR
        #   label="\n[Büchi]"
        #   labelloc="t"
        #   node [shape="circle"]
        #   I [label="", style=invis, width=0]
        #   I -> 1
        #   0 [label="0", peripheries=2]
        #   0 -> 0 [label="1"]
        #   1 [label="1"]
        #   1 -> 0 [label="green | red"]
        #   1 -> 1 [label="!green & !red"]
        # }
        # """

        dotFormat = """
        digraph "Fmid & XFdest & G!obs" {
          rankdir=LR
          label="\n[Büchi]"
          labelloc="t"
          node [shape="circle"]
          I [label="", style=invis, width=0]
          I -> 1
          0 [label="0", peripheries=2]
          0 -> 0 [label="!obs"]
          1 [label="1"]
          1 -> 2 [label="mid & !obs"]
          1 -> 3 [label="!mid & !obs"]
          2 [label="2"]
          2 -> 0 [label="dest & !obs"]
          2 -> 2 [label="!dest & !obs"]
          3 [label="3"]
          3 -> 0 [label="dest & mid & !obs"]
          3 -> 2 [label="!dest & mid & !obs"]
          3 -> 3 [label="!dest & !mid & !obs"]
          3 -> 4 [label="dest & !mid & !obs"]
          4 [label="4"]
          4 -> 0 [label="mid & !obs"]
          4 -> 4 [label="!mid & !obs"]
        }"""

        # generate a product mdp
        product_result = create_product_mdp(mdp1, dotFormat)

        # hyper-parameters of prior
        Beta0_itr = np.asarray([0.1, 0.1, 0.1, 0.1, -0.4])
        Sigma0_itr = np.diag([1e0, 1e0, 1e0, 1e0, 1e0])
        a0_itr = 3.0
        b0_itr = 1.0
        c0_itr = 3.0
        d0_itr = 1.0

        # known parameters
        alpha1 = np.diag([1.0, 1.0, 1.0])
        alpha2 = -np.diag([1.0, 1.0, 1.0])
        Alpha = np.concatenate((alpha1, alpha2), axis=1)

        # assumed ground truth
        beta_true3 = np.asarray([0.25, 1.0, 0.5, 0.5, -0.1])
        delta_w_true3 = 0.01
        delta_v_true3 = 0.01

        # plot prior distribution
        data_dir = env.pkg_dir + '/src/data'
        plot_distribution(Beta0_itr, Sigma0_itr, beta_true3, 0, data_dir)

        # Initialization for MCMC
        env.trust_dict = create_trust_map()  # dictionary for storing the trust values
        posterior_list_beta = []
        posterior_list_sigma = []
        trust_gains = []

        # 0.1 update traversability and visibility
        info_r1 = env.r1_dynamic_traversability, env.r1_dynamic_visibility
        info_r2 = env.r2_dynamic_traversability, env.r2_dynamic_visibility
        info_r3 = env.r3_dynamic_traversability, env.r3_dynamic_visibility

        # 0.2 update environment: add traversability and visibility
        environment_dict = update_environment(env_state_location_map, info_r1, info_r2, info_r3)

        # 1. Generate multiple optimal routes: trust-based active reinforcement learning
        for pstate in product_result[0].keys():  # update traversability information
            mstate = re.search(r'(.*)\|', pstate).group(1)  # regular expression to get the mdp state
            product_result[0][pstate] = environment_dict[env_state_location_map[mstate]][0]

        product_mdp1 = ql.RL(product_result[0], product_result[1], product_result[2], qle.alpha, qle.gamma)
        policy, route_list, rout_dict = product_mdp1.policy_route_distribution(product_result[3], product_result[4],
                                                                    Beta0_itr, Sigma0_itr, 1000, sample_size=1000)
        policyt, route_listt, rout_dictt = product_mdp1.policy_route_distribution_(product_result[3], product_result[4],
                                                                    Beta0_itr, Sigma0_itr, 1000, sample_size=1000)
        print "Rollout route list:", route_list, "\n"

        # exit()
        dataX_train = []
        dataY_train = []
        figp = plt.figure(figsize=(5, 5))
        figp.tight_layout()
        axp = plt.axes(projection='3d')
        for route_key in rout_dict.keys():
            parameters_3d = np.array(rout_dict[route_key])
            axp.plot3D(parameters_3d[:, 0], parameters_3d[:, 1], parameters_3d[:, 2], 'o')
            for parameter3d in parameters_3d:
                dataX_train.append(parameter3d)
                dataY_train.append(route_key)

        dataX_test = []
        dataY_test = []
        for route_key in rout_dictt.keys():
            parameters_3d_test = np.array(rout_dictt[route_key])
            for parameter3d in parameters_3d_test:
                dataX_test.append(parameter3d)
                dataY_test.append(route_key)
        # dataX_train, dataX_test, dataY_train, dataY_test = train_test_split(dataX, dataY, train_size=0.7, test_size=0.3)

        start_time = time.time()

        neigh3 = KNeighborsClassifier(n_neighbors=3)
        modelKNN3 = neigh3.fit(dataX_train, dataY_train)
        print "test accuracy:", modelKNN3.score(dataX_test, dataY_test)
        print "train accuracy:", modelKNN3.score(dataX_train, dataY_train)

        neigh7 = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='ball_tree')
        modelKNN7 = neigh7.fit(dataX_train, dataY_train)
        print "test accuracy:", modelKNN7.score(dataX_test, dataY_test)
        print "train accuracy:", modelKNN7.score(dataX_train, dataY_train)

        neigh15 = KNeighborsClassifier(n_neighbors=15, weights='distance', algorithm='ball_tree')
        modelKNN15 = neigh15.fit(dataX_train, dataY_train)
        print "test accuracy:", modelKNN15.score(dataX_test, dataY_test)
        print "train accuracy:", modelKNN15.score(dataX_train, dataY_train)

        neigh21 = KNeighborsClassifier(n_neighbors=21, weights='distance', algorithm='ball_tree')
        modelKNN21 = neigh21.fit(dataX_train, dataY_train)
        print "test accuracy:", modelKNN21.score(dataX_test, dataY_test)
        print "train accuracy:", modelKNN21.score(dataX_train, dataY_train)

        neigh31 = KNeighborsClassifier(n_neighbors=31, weights='distance', algorithm='ball_tree')
        modelKNN31 = neigh31.fit(dataX_train, dataY_train)
        print "test accuracy:", modelKNN31.score(dataX_test, dataY_test)
        print "train accuracy:", modelKNN31.score(dataX_train, dataY_train)

        neigh41 = KNeighborsClassifier(n_neighbors=41, weights='distance', algorithm='ball_tree')
        modelKNN41 = neigh41.fit(dataX_train, dataY_train)
        print "test accuracy:", modelKNN41.score(dataX_test, dataY_test)
        print "train accuracy:", modelKNN41.score(dataX_train, dataY_train)

        neigh51 = KNeighborsClassifier(n_neighbors=51, weights='distance', algorithm='ball_tree')
        modelKNN51 = neigh51.fit(dataX_train, dataY_train)
        print "test accuracy:", modelKNN51.score(dataX_test, dataY_test)
        print "train accuracy:", modelKNN51.score(dataX_train, dataY_train)

        end_time = time.time()
        print "cost time:", end_time - start_time

        plt.show()
    except rospy.ROSInterruptException:
        pass

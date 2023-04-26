#! /usr/bin/env python2

import numpy as np
import copy
import itertools

import q_learning_environment as qle


class RL:
    # Initialize alpha, gamma, states, actions, rewards, and Q-values
    def __init__(self, state_features, state_actions, trans_rewards, alpha=0.75, gamma=0.6):

        # fixed parameters
        self.gamma = gamma
        self.alpha = alpha

        # parameters from input
        self.states = state_actions.keys()
        self.actions = state_actions
        self.trans_rewards = trans_rewards
        self.state_features = state_features

        self.Q = {}
        for state_action in self.trans_rewards.keys():
            self.Q[state_action] = 0.0

        self.Q_dist = {}
        for state_action in self.trans_rewards.keys():
            self.Q_dist[state_action] = []

    # obtain next state
    def get_next_state(self, curr_state, action):
        return self.trans_rewards[(curr_state, action)][0]

    # obtain the reward (s,a,s')
    def get_reward(self, curr_state, action, next_state):
        state_action = (curr_state, action)
        return self.trans_rewards[state_action][2]

    # ONLY consider the deterministic transition
    def update_reward_fun(self, beta):
        for state_action in self.trans_rewards:
            next_state = self.trans_rewards[state_action][0]
            concrete_feature = (self.state_features[next_state] + self.state_features[state_action[0]]) / 2.0
            self.trans_rewards[state_action][2] = np.matmul(beta[0:2], concrete_feature) + beta[2]

    # sample actions: thompson sampling
    def sample_action(self, state):
        action_length = len(self.actions[state])
        sampled_actions = np.random.multinomial(1, [1.0/action_length]*action_length)
        action_index = np.where(sampled_actions == 1)[0][0]
        # print action_index, self.actions[state], sampled_actions
        return self.actions[state][action_index]

    def max_q(self, state):
        Q_state_list = []
        for current_action in self.actions[state]:
            Q_state_list.append(self.Q[(state, current_action)])
        index = np.argmax(Q_state_list)
        return np.max(Q_state_list), self.actions[state][index]

    # initialize the q-value with reward function
    def q_init(self, ending_states):
        for state_action in self.trans_rewards.keys():
            if state_action[0] in ending_states:
                self.Q[state_action] = 0.0
                continue

            self.Q[state_action] = self.trans_rewards[state_action][2]

    # one 1000-iteration of q-learning
    def q_value(self, beta, start_state, ending_states, iterations, epsilon=0.01):
        episodic_return = [-100.0]  # collect the return
        route_list = []

        self.update_reward_fun(beta)  # need to first update reward based on betas (model parameters)

        # revise the reward for state whose next state is a terminate state
        for state_action in self.trans_rewards.keys():
            state_action_next = self.get_next_state(state_action[0], state_action[1])  # next state
            if (state_action_next in ending_states) and (state_action[0] not in ending_states):  # revise the reward with a large value
                self.trans_rewards[state_action][2] += 999.0
            elif (state_action_next in ending_states) and (state_action[0] == state_action_next):  # revise the reward with a large value
                self.trans_rewards[state_action][2] += 999.0

        self.q_init(ending_states)  # initialize the q value before the iteration

        # start iteration
        for itr in range(iterations):
            curr_state = self.states[np.random.randint(0, len(self.states))]  # sample a start state
            steps = 0  # For observing the steps of robots run

            # epsilon-greedy exploration of action
            if np.random.uniform(0, 1.0) < 0.8:  # customize a time-varying epsilon
                action = self.sample_action(curr_state)
            else:
                action = self.max_q(curr_state)[1]

            # next state and reward
            next_state = self.get_next_state(curr_state, action)
            curr_reward = self.get_reward(curr_state, action, next_state)

            # update q-table value
            max_q_next_state = self.max_q(next_state)[0]
            td = curr_reward + self.gamma * max_q_next_state - self.Q[(curr_state, action)]
            self.Q[(curr_state, action)] += self.alpha * td

            steps += 1
            # print "steps:", steps  # for debugging the iteration performance

        # when to terminate the iteration: after 100 iterations and last 80 performance is the same with last 10
        route_itr, return_itr = self.get_optimal_route_(start_state, ending_states)
        # if np.abs(np.mean(episodic_return[-80:-1]) - np.mean(episodic_return[-10:-1])) < 0.01:
        # plt.plot(episodic_return, marker='o')  # plot the return
        return self.Q, route_itr

    # Get the optimal route
    def get_optimal_route_(self, start_state, ending_state_set):
        route = [start_state]
        acc_reward = 0.0
        next_state = start_state
        while next_state not in ending_state_set:
            action = self.max_q(start_state)[1]
            next_state = self.get_next_state(start_state, action)
            acc_reward = self.get_reward(start_state, action, next_state) + acc_reward
            route.append(next_state)
            start_state = next_state

        # print "length:", len(route), route
        return route, acc_reward

    # Get the optimal route
    def get_optimal_route(self, start_state, end_state, ending_state_set):
        route = [start_state]
        acc_reward = 0.0
        next_state = start_state
        while next_state != end_state:
            if next_state in ending_state_set:  # avoid repeating visiting accepting states
                break
            action = self.max_q(start_state)[1]
            next_state = self.get_next_state(start_state, action)
            acc_reward = self.get_reward(start_state, action, next_state) + acc_reward
            route.append(next_state)
            start_state = next_state

            if len(route) > 5 * 5:
                return route, -np.inf

        # print "length:", len(route), route
        return route, acc_reward

    # routes & policies: enumerate different theta (parameter)
    def policy_route_distribution_bound(self, start_state, ending_states, betas_mean, betas_var, episode, iterations):
        action2index = {'E': 0, 'S': 1, 'W': 2, 'N': 3, 'c': 4}

        optimal_route_dict = {}
        policy_dict = {}

        # way 1 generating beta samples
        # sample_size = 1 + int(550 * np.exp(-0.5 * episode))
        # distribution_betas_2_4 = np.random.multivariate_normal(betas_mean[2:], betas_var[2:, 2:], sample_size)

        # way 2 generating beta samples
        interval = np.array([1.96 * np.sqrt(element) for element in np.diag(betas_var)])
        betas_l = betas_mean - interval
        betas_u = betas_mean + interval
        beta0 = set(np.arange(betas_l[0], betas_u[0], 5.1))
        beta1 = set(np.arange(betas_l[1], betas_u[1], 5.1))
        beta2 = set([value for value in np.arange(betas_l[2], betas_u[2], 0.15) if -0.2 <= value <= 1.1])
        beta3 = set([value for value in np.arange(betas_l[3], betas_u[3], 0.15) if -0.2 <= value <= 1.1])
        beta4 = set([value for value in np.arange(betas_l[4], betas_u[4], 0.15) if -1.1 <= value <= 1.1])
        distribution_betas_2_4 = set(itertools.product(beta2, beta3, beta4))

        # print "distribution_betas:", distribution_betas_2_4
        # distribution_betas_2_4 = [(betas_mean[2], betas_mean[3], betas_mean[4])]

        for beta_2_4 in distribution_betas_2_4:
            q_values, optimal_route = copy.deepcopy(self.q_value(beta_2_4, start_state, ending_states, iterations))
            if len(optimal_route) > 5 * 5:
                continue

            best_actions = {}  # action dictionary: equivalent to policy
            for state in self.states:
                best_actions[state] = [0.0] * 5
            for state in self.states:
                best_action = self.max_q(state)[1]
                best_actions[state][action2index[best_action]] += 1

            policy_dict[tuple(beta_2_4)] = best_actions
            optimal_route_dict[tuple(beta_2_4)] = optimal_route

        # print "policy:", policy_dict, "\n"
        # print "optimal routes:", optimal_route_dict, "\n"

        return policy_dict, optimal_route_dict

    # routes & policies: enumerate different theta (parameter)
    def policy_route_distribution_mean(self, start_state, ending_states, betas_mean, iterations=5000):
        action2index = {'E': 0, 'S': 1, 'W': 2, 'N': 3, 'c': 4}

        # print "distribution_betas:", distribution_betas_2_4
        beta_2_4 = (betas_mean[2], betas_mean[3], betas_mean[4])
        best_actions = {}

        while True:
            q_values, optimal_route = copy.deepcopy(self.q_value(beta_2_4, start_state, ending_states, iterations))
            if len(optimal_route) > 5 * 5:
                continue

            best_actions = {}  # action dictionary: equivalent to policy
            for state in self.states:
                best_actions[state] = [0.0] * 5
            for state in self.states:
                best_action = self.max_q(state)[1]
                best_actions[state][action2index[best_action]] += 1

            break

        # print "policy:", best_actions, "\n"
        # print "optimal routes:", optimal_route, "\n"

        return best_actions, optimal_route


# generate policies
def policies(best_actions, sample_size):
    policy = {}
    for action in best_actions.keys():
        policy[action] = np.array(best_actions[action]) / sample_size
    return policy


def optimal_route_set(optimal_route_list):
    optimal_routes = []
    for optimal_route in optimal_route_list:
        if optimal_route not in optimal_routes:
            optimal_routes.append(optimal_route)
    return optimal_routes


def policy_set(policy_list):
    policyList = []
    for policy in policy_list:
        if policy not in policyList:
            policyList.append(policy)
    return policyList

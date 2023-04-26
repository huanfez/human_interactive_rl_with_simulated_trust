#! /usr/bin/env python2

import numpy as np
import copy
from scipy.stats import multinomial


def cell_path_trust(cell_path, trust_dict):
    K = len(cell_path)
    y_1toK = np.ones((K, 3))
    for k in range(1, K):
        cell_k = cell_path[k]
        y_1toK[k, :] = trust_dict[cell_k]

    return y_1toK[1:K]


def cell_path_situational_awareness(cell_path, environment_dict):
    K = len(cell_path)
    Z_1toK = np.ones((K, 3, 5))
    for k in range(1, K):
        cell_k_1 = cell_path[k - 1]
        cell_k = cell_path[k]
        Z_1toK[k, :, 2:4] = (environment_dict[cell_k] + environment_dict[cell_k_1]) / 2.0

    return Z_1toK[1:K]


def cell_path_trust_prediction_mode1(Z_1toK, means_Beta):
    K = len(Z_1toK)
    Z_1toK_pred = np.copy(Z_1toK)
    xall_1toK_pred = np.zeros((K, 3))
    for k in range(0, K):
        for i in range(0, 3):
            if i == 0:
                Z_1toK_pred[k, i, 0] = 0.0
            else:
                Z_1toK_pred[k, i, 0] = xall_1toK_pred[k, i-1]

            if k == 0:
                Z_1toK_pred[k, i, 1] = 0.0
            else:
                Z_1toK_pred[k, i, 1] = xall_1toK_pred[k - 1, i]

            xall_1toK_pred[k, i] = np.matmul(Z_1toK_pred[k, i, :], means_Beta.T)

    return xall_1toK_pred


# Greedy strategy
def optimal_cell_path(betas, Sigma, cell_path_list, environment_dict):
    average_trust_list = []
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
        trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, betas)
        # print "environment info:", Z_1toK
        # print "trust of all robots:", trust_all_robots
        average_trust = np.mean(trust_all_robots)
        average_trust_list.append(average_trust)

    # print average_trust_list
    index = np.argmax(average_trust_list)
    print "greedy:, which path", index
    return cell_path_list[index]


# Thompson sampling
def optimal_cell_path_ts(betas, Sigma, cell_path_list, environment_dict):
    average_trust_list = []
    Beta_ts = np.random.multivariate_normal(betas, Sigma)
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
        trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_ts)
        # print "environment info:", Z_1toK
        # print "trust of all robots:", trust_all_robots
        average_trust = np.sum(trust_all_robots)
        average_trust_list.append(average_trust)

    index = np.argmax(average_trust_list)
    print "TS query: which path ", index, average_trust_list

    return cell_path_list[index]


# UCB
def optimal_cell_path_ucb(betas, Sigma, cell_path_list, environment_dict):
    average_trust_list = []
    Beta_ucb = np.asarray([betas[0] - 1.96 * np.sqrt(Sigma[0][0]), betas[1] - 1.96 * np.sqrt(Sigma[1][1]),
                           betas[2] - 1.96 * np.sqrt(Sigma[2][2]), betas[3] - 1.96 * np.sqrt(Sigma[3][3])])
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
        trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_ucb)
        # print "environment info:", Z_1toK
        # print "trust of all robots:", trust_all_robots
        average_trust = np.mean(trust_all_robots)
        average_trust_list.append(average_trust)

    print average_trust_list
    index = np.argmax(average_trust_list)

    return cell_path_list[index]


# Decision-field theory based probability improvement
def optimal_cell_path_pi(betas, Sigma, cell_path_list, environment_dict, f, gamma=0.0, sample_size=2000):
    J = len(cell_path_list)
    alpha = np.zeros(J)
    f_all = np.zeros((sample_size, J))
    sampled_betas = np.random.multivariate_normal(betas, Sigma, sample_size)

    sequence = 0
    for Beta_pi in sampled_betas:
        average_trust_list = []
        for cell_path in cell_path_list:
            Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
            trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_pi)
            # print "environment info:", Z_1toK
            # print "trust of all robots:", trust_all_robots
            average_trust = np.mean(trust_all_robots)
            average_trust_list.append(average_trust)

        for j in range(0, J):
            delta_trust_j = (average_trust_list[j] * J - np.sum(average_trust_list)) / (J - 1)
            f_all[sequence, j] = gamma * f[j] + delta_trust_j
            alpha[j] += f_all[sequence, j] / float(sample_size)

        sequence += 1

    print "alpha:", alpha, sequence
    index = np.argmax(alpha)

    return cell_path_list[index], np.mean(f_all, axis=0)


# Decision-field theory based probability improvement - Thompson sampling
def optimal_cell_path_pi2(betas, Sigma, cell_path_list, environment_dict, f, gamma=0.0, sample_size=2000):
    J = len(cell_path_list)
    alpha = np.zeros(J)
    f_all = np.zeros((sample_size, J))
    sampled_betas = np.random.multivariate_normal(betas, Sigma, sample_size)

    sequence = 0
    for Beta_pi in sampled_betas:
        average_trust_list = []
        for cell_path in cell_path_list:
            Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
            trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_pi)
            # print "environment info:", Z_1toK
            # print "trust of all robots:", trust_all_robots
            average_trust = np.mean(trust_all_robots)
            average_trust_list.append(average_trust)

        for j in range(0, J):
            delta_trust_j = (average_trust_list[j] * J - np.sum(average_trust_list)) / (J - 1)
            f_all[sequence, j] = gamma * f[j] + delta_trust_j
            alpha[j] += (f_all[sequence, j] > 0.0) / float(sample_size)

        sequence += 1

    if len(alpha) == 1:
        return cell_path_list[0], np.mean(f_all, axis=0)

    alpha = alpha / np.sum(alpha)
    index = np.where(multinomial.rvs(1, alpha) == 1)[0]
    # index = np.argmax(alpha)
    print "alpha:", alpha, sequence, "which path:", index.item()
    # index = np.argmax(alpha)

    return cell_path_list[index.item()], np.mean(f_all, axis=0)


# acquisition function for lca entropy
def optimal_cell_path_mix3(cell_path_list, environment_dict, history_path_list, betas, Sigma, f,
                           coefficients=np.array([0.2, 0.2, 0.6]), gamma=0.0, sample_size=2000):
    """Return the trajectory that has the largest mixture utility"""

    '''candidate trajectory entropy and workload'''
    log_workload_list = []
    entropy_list = []
    for cell_path in cell_path_list:
        # each candidate trajectory data
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)

        # Add candidate trajectory data
        apen = 0.0
        workload = 0.0
        for atr in range(2, 4, 1):
            Z_1toK_data_atr = np.concatenate((Z_1toK[:, 0, atr], Z_1toK[:, 1, atr], Z_1toK[:, 2, atr]), axis=0)
            # for workload
            mean_Z_1toK_data_atr = np.mean(Z_1toK_data_atr)
            workload += 1.0 / (1.0 + np.exp(-mean_Z_1toK_data_atr))
            # for approximate entropy
            apen += ApEn_new(Z_1toK_data_atr)

        entropy_list.append(apen)
        log_workload_list.append(np.log(workload))

    J = len(cell_path_list)
    alpha = np.zeros(J)
    f_all = np.zeros((sample_size, J))
    sampled_betas = np.random.multivariate_normal(betas, Sigma, sample_size)
    sequence = 0
    for Beta_pi in sampled_betas:
        average_trust_list = []
        for cell_path in cell_path_list:
            Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
            trust_all_robots = cell_path_trust_prediction_mode1(Z_1toK, Beta_pi)
            average_trust = np.mean(trust_all_robots)
            average_trust_list.append(average_trust)

        for j in range(0, J):
            delta_trust_j = (average_trust_list[j] * J - np.sum(average_trust_list)) / (J - 1)
            f_all[sequence, j] = gamma * f[j] + delta_trust_j
            alpha[j] += (f_all[sequence, j] > 0.0) / float(sample_size)

        sequence += 1
    alpha = alpha / np.sum(alpha)
    if len(alpha) == 1:
        return cell_path_list[0], np.mean(f_all, axis=0)

    # mixture of three
    mixture_utility_list = []
    for j in range(0, J):
        mixture_utility = coefficients[0] * alpha[j] + coefficients[1] * log_workload_list[j] - \
                          coefficients[2] * entropy_list[j]
        mixture_utility_list.append(mixture_utility)

    index = np.argmax(mixture_utility_list)
    print "mixture utility:", mixture_utility_list, "which path:", index.item(), mixture_utility_list[index.item()], \
        log_workload_list[index.item()], cell_path_list[index.item()]

    return cell_path_list[index.item()], np.mean(f_all, axis=0)


# Entropy of input variable
def optimal_cell_path_entropy(cell_path_list, environment_dict, history_path_list):
    if not history_path_list:
        entropy_list = []
        for cell_path in cell_path_list:
            Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
            Z_1toK_data = np.concatenate((Z_1toK[:, 0], Z_1toK[:, 1], Z_1toK[:, 2]), axis=0)
            cov_Z_1toK_data = np.cov(Z_1toK_data[:, 2:4].T)
            entropy_new = 0.5 * np.log(np.linalg.det(cov_Z_1toK_data)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))
            entropy_list.append(entropy_new)

        index = np.argmax(entropy_list)
        return cell_path_list[index.item()]

    Z_history_data = np.empty((1, 5))
    for history_path in history_path_list:
        Z_1toK_history = cell_path_situational_awareness(history_path, environment_dict)
        Z_history_data = np.concatenate((Z_history_data, Z_1toK_history[:, 0], Z_1toK_history[:, 1], Z_1toK_history[:, 2]), axis=0)

    cov_Z_history_data = np.cov(copy.deepcopy(Z_history_data[1:, 2:4]).T)
    entropy_history = 0.5 * np.log(np.linalg.det(cov_Z_history_data)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))

    entropy_change_list = []
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)

        Z_history_data_ = copy.deepcopy(Z_history_data)
        Z_history_data_ = np.concatenate((Z_history_data_, Z_1toK[:, 0], Z_1toK[:, 1], Z_1toK[:, 2]), axis=0)
        cov_Z_history_data_ = np.cov(copy.deepcopy(Z_history_data_[1:, 2:4]).T)
        entropy_new = 0.5 * np.log(np.linalg.det(cov_Z_history_data_)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))
        if cell_path in history_path_list:
            entropy_change_list.append(entropy_new - 1.0)
        else:
            entropy_change_list.append(entropy_new)

    index = np.argmax(entropy_change_list)
    print "entropy:", entropy_change_list, "which path:", index.item(), entropy_change_list[index.item()], cell_path_list[index.item()]

    return cell_path_list[index.item()]


# Entropy of input variable mean
def optimal_cell_path_entropy2(cell_path_list, environment_dict, history_path_list):
    if not history_path_list:  # can also use mean beta value to elect a trajectory
        return cell_path_list[0]

    # Use mean/cov value of each trajectory
    mean_Z_history_data = np.empty((1, 5))
    for history_path in history_path_list:
        Z_1toK_history = cell_path_situational_awareness(history_path, environment_dict)
        Z_history_data = np.concatenate((Z_1toK_history[:, 0], Z_1toK_history[:, 1], Z_1toK_history[:, 2]), axis=0)
        # print "np.mean(Z_history_data, axis=0):", np.mean(Z_history_data, axis=0)
        mean_Z_history_data = np.concatenate((mean_Z_history_data, np.array([np.mean(Z_history_data, axis=0)])), axis=0)

    cov_Z_history_data = np.cov(copy.deepcopy(mean_Z_history_data[1:, 2:4]).T)
    entropy_history = 0.5 * np.log(np.linalg.det(cov_Z_history_data)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))

    entropy_change_list = []
    for cell_path in cell_path_list:
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
        Z_1toK_data = np.concatenate((Z_1toK[:, 0], Z_1toK[:, 1], Z_1toK[:, 2]), axis=0)
        mean_Z_history_data_ = copy.deepcopy(mean_Z_history_data)

        mean_Z_history_data_ = np.concatenate((mean_Z_history_data_, np.array([np.mean(Z_1toK_data, axis=0)])), axis=0)
        cov_Z_history_data_ = np.cov(copy.deepcopy(mean_Z_history_data_[1:, 2:4]).T)
        entropy_new = 0.5 * np.log(np.linalg.det(cov_Z_history_data_)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))
        entropy_change_list.append(entropy_new - entropy_history)

    print "entropy:", entropy_change_list
    index = np.argmin(entropy_change_list)
    print "which path:", index.item()

    return cell_path_list[index.item()]


# acquisition function for mixture
def optimal_cell_path_mix(cell_path_list, environment_dict, history_path_list, coefficient=0.5):
    """Return the trajectory that has the largest mixture utility"""
    '''If it is the first iteration'''
    if not history_path_list:
        mixture_utility_list = []
        entropy_list = []
        log_workload_list = []
        for cell_path in cell_path_list:
            Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
            Z_1toK_data = np.concatenate((Z_1toK[:, 0], Z_1toK[:, 1], Z_1toK[:, 2]), axis=0)
            cov_Z_1toK_data = np.cov(Z_1toK_data[:, 2:4].T)
            entropy_new = 0.5 * np.log(np.linalg.det(cov_Z_1toK_data)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))
            entropy_list.append(entropy_new)

            workload = 0.0
            for atr in range(2, 4, 1):
                Z_1toK_data_atr = np.concatenate((Z_1toK[:, 0, atr], Z_1toK[:, 1, atr], Z_1toK[:, 2, atr]), axis=0)
                mean_Z_1toK_data_atr = np.sum(Z_1toK_data_atr)
                workload += 1.0 / (1.0 + np.exp(-mean_Z_1toK_data_atr))
            log_workload = np.log(workload)
            log_workload_list.append(log_workload)

            # mixture of two
            mixture_utility = coefficient * entropy_new + (1 - coefficient) * log_workload
            mixture_utility_list.append(mixture_utility)

        index = np.argmax(mixture_utility_list)
        return cell_path_list[index.item()]

    '''Historical trajectory entropy'''
    Z_history_data = np.empty((1, 5))
    for history_path in history_path_list:
        Z_1toK_history = cell_path_situational_awareness(history_path, environment_dict)
        Z_history_data = np.concatenate((Z_history_data, Z_1toK_history[:, 0], Z_1toK_history[:, 1], Z_1toK_history[:, 2]), axis=0)
    cov_Z_history_data = np.cov(copy.deepcopy(Z_history_data[1:, 2:4]).T)
    entropy_history = 0.5 * np.log(np.linalg.det(cov_Z_history_data)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))

    '''previous trajectory workload'''
    Z_1toK_prev = cell_path_situational_awareness(history_path_list[-1], environment_dict)
    workload_prev = 0.0
    for atr in range(2, 4, 1):
        Z_1toK_data_atr = np.concatenate((Z_1toK_prev[:, 0, atr], Z_1toK_prev[:, 1, atr], Z_1toK_prev[:, 2, atr]), axis=0)
        mean_Z_1toK_data_atr = np.sum(Z_1toK_data_atr)
        workload_prev += 1.0 / (1.0 + np.exp(-mean_Z_1toK_data_atr))
    log_workload_prev = np.log(workload_prev)

    '''candidate trajectory entropy and workload'''
    mixture_utility_list = []
    entropy_change_list = []
    log_workload_list = []
    for cell_path in cell_path_list:
        # each candidate trajectory data
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)

        # Add candidate trajectory data into historical data: calculate new entropy
        Z_history_data_ = copy.deepcopy(Z_history_data)
        Z_history_data_ = np.concatenate((Z_history_data_, Z_1toK[:, 0], Z_1toK[:, 1], Z_1toK[:, 2]), axis=0)
        cov_Z_history_data_ = np.cov(copy.deepcopy(Z_history_data_[1:, 2:4]).T)
        entropy_new = 0.5 * np.log(np.linalg.det(cov_Z_history_data_)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))
        if cell_path in history_path_list:
            entropy_new = entropy_new - 1.0
        entropy_change_list.append(entropy_new)

        # calculate candidate trajectory log workload
        workload = 0.0
        for atr in range(2, 4, 1):
            Z_1toK_data_atr = np.concatenate((Z_1toK[:, 0, atr], Z_1toK[:, 1, atr], Z_1toK[:, 2, atr]), axis=0)
            mean_Z_1toK_data_atr = np.sum(Z_1toK_data_atr)
            workload += 1.0 / (1.0 + np.exp(-mean_Z_1toK_data_atr))
        log_workload = np.log(workload)
        log_workload_list.append(log_workload)

        # mixture of two
        mixture_utility = coefficient * entropy_new + (1 - coefficient) * log_workload
        mixture_utility_list.append(mixture_utility)

    index = np.argmax(mixture_utility_list)
    print "mixture utility:", mixture_utility_list, "which path:", index.item(), entropy_change_list[index.item()], \
        log_workload_list[index.item()], cell_path_list[index.item()]

    return cell_path_list[index.item()]


# acquisition function for mixture
def optimal_cell_path_lca(cell_path_list, environment_dict, history_path_list, coefficient=0.5):
    """Return the trajectory that has the largest lca utility"""
    '''If it is the first iteration'''
    if not history_path_list:
        lca_utility_list = []
        entropy_list = []
        log_workload_list = []
        for cell_path in cell_path_list:
            Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
            Z_1toK_data = np.concatenate((Z_1toK[:, 0], Z_1toK[:, 1], Z_1toK[:, 2]), axis=0)
            cov_Z_1toK_data = np.cov(Z_1toK_data[:, 2:4].T)
            entropy_new = 0.5 * np.log(np.linalg.det(cov_Z_1toK_data)) + 0.5 * 2.0 * (1.0 + np.log(2.0 * np.pi))
            entropy_list.append(entropy_new)

            workload = 0.0
            for atr in range(2, 4, 1):
                Z_1toK_data_atr = np.concatenate((Z_1toK[:, 0, atr], Z_1toK[:, 1, atr], Z_1toK[:, 2, atr]), axis=0)
                mean_Z_1toK_data_atr = np.sum(Z_1toK_data_atr)
                workload += 1.0 / (1.0 + np.exp(-mean_Z_1toK_data_atr))
            log_workload = np.log(workload)
            log_workload_list.append(log_workload)

            # lca of two
            lca_utility = lca(entropy_new, log_workload)
            lca_utility_list.append(lca_utility)

        index = np.argmax(lca_utility_list)
        return cell_path_list[index.item()]

    '''Historical trajectory entropy'''
    Z_history_data = np.empty((1, 5))
    for history_path in history_path_list:
        Z_1toK_history = cell_path_situational_awareness(history_path, environment_dict)
        Z_history_data = np.concatenate((Z_history_data, Z_1toK_history[:, 0], Z_1toK_history[:, 1], Z_1toK_history[:, 2]), axis=0)
    cov_Z_history_data = np.cov(copy.deepcopy(Z_history_data[1:, 2:4]).T)
    entropy_history = 0.5 * np.log(np.linalg.det(cov_Z_history_data)) + 0.5 * 2.0 * (1.0 + np.log(2.0 * np.pi))

    '''previous trajectory workload'''
    Z_1toK_prev = cell_path_situational_awareness(history_path_list[-1], environment_dict)
    workload_prev = 0.0
    for atr in range(2, 4, 1):
        Z_1toK_data_atr = np.concatenate((Z_1toK_prev[:, 0, atr], Z_1toK_prev[:, 1, atr], Z_1toK_prev[:, 2, atr]), axis=0)
        mean_Z_1toK_data_atr = np.sum(Z_1toK_data_atr)
        workload_prev += 1.0 / (1.0 + np.exp(-mean_Z_1toK_data_atr))
    log_workload_prev = np.log(workload_prev)

    '''candidate trajectory entropy and workload'''
    lca_utility_list = []
    entropy_change_list = []
    log_workload_list = []
    for cell_path in cell_path_list:
        # each candidate trajectory data
        Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)

        # Add candidate trajectory data into historical data: calculate new entropy
        Z_history_data_ = copy.deepcopy(Z_history_data)
        Z_history_data_ = np.concatenate((Z_history_data_, Z_1toK[:, 0], Z_1toK[:, 1], Z_1toK[:, 2]), axis=0)
        cov_Z_history_data_ = np.cov(copy.deepcopy(Z_history_data_[1:, 2:4]).T)
        entropy_new = 0.5 * np.log(np.linalg.det(cov_Z_history_data_)) + 0.5 * 2.0 * (1 + np.log(2.0 * np.pi))
        if cell_path in history_path_list:
            entropy_new = entropy_new - 1.0
        entropy_change_list.append(entropy_new)

        # calculate candidate trajectory log workload
        workload = 0.0
        for atr in range(2, 4, 1):
            Z_1toK_data_atr = np.concatenate((Z_1toK[:, 0, atr], Z_1toK[:, 1, atr], Z_1toK[:, 2, atr]), axis=0)
            mean_Z_1toK_data_atr = np.sum(Z_1toK_data_atr)
            workload += 1.0 / (1.0 + np.exp(-mean_Z_1toK_data_atr))
        log_workload = np.log(workload)
        log_workload_list.append(log_workload)

        # lca of two
        lca_utility = lca((entropy_new - entropy_history) * 100.0, (log_workload - log_workload_prev) * 100.0)
        lca_utility_list.append(lca_utility)

    index = np.argmax(lca_utility_list)
    print "lca utility:", lca_utility_list, "which path:", index.item(), entropy_change_list[index.item()], \
        log_workload_list[index.item()], cell_path_list[index.item()]

    return cell_path_list[index.item()]


# LCA function
def lca(entropy_change, workload_change):
    if entropy_change >= 0.0:
        V_e = np.log(1.0 + entropy_change)
    else:
        V_e = -np.log(1.0 + np.abs(entropy_change)) - (np.log(1.0 + np.abs(entropy_change)))**2

    if workload_change >= 0.0:
        V_w = np.log(1.0 + workload_change)
    else:
        V_w = -np.log(1.0 + np.abs(workload_change)) - (np.log(1.0 + np.abs(workload_change))) ** 2

    preference = V_e + V_w
    return preference


# Approximate entropy of time-series
def ApEn_new(U, m=3, r=0.5):
    U = np.array(U)
    N = U.shape[0]

    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i + m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z

    return abs(_phi(m + 1) - _phi(m))


# apen of a path
def ApEn_path(cell_path, environment_dict):
    Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)

    apen = 0.0
    for atr in range(2, 4, 1):
        Z_1toK_data_atr = np.concatenate((Z_1toK[:, 0, atr], Z_1toK[:, 1, atr], Z_1toK[:, 2, atr]), axis=0)
        apen += ApEn_new(Z_1toK_data_atr)

    return apen


# Approximate entropy of input variable
def optimal_cell_path_apen(cell_path_list, environment_dict):
    apen_list = []
    for cell_path in cell_path_list:
        apen = ApEn_path(cell_path, environment_dict)
        apen_list.append(apen)

    index = np.argmin(apen_list)
    print "Approximate Entropy:", apen_list, "which path:", index

    return cell_path_list[index]


# teleoperation workload
def workload_teleop(cell_path, environment_dict):
    workload_teleop_1toK = 0.0
    Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
    for k in range(len(Z_1toK)):
        Z_k_1toI_mean = np.mean(Z_1toK[k], axis=0)
        complexity = np.linalg.norm(2.5 - Z_k_1toI_mean[2:4], 2)
        workload_teleop_k = 1.0 - np.exp(-complexity * k)
        workload_teleop_1toK += workload_teleop_k

    return workload_teleop_1toK


def optimal_cell_path_teleop_workload(cell_path_list, environment_dict):
    workload_teleop_list = []
    for cell_path in cell_path_list:
        workload_j = workload_teleop(cell_path, environment_dict)
        workload_teleop_list.append(workload_j)

    index = np.argmin(workload_teleop_list)
    print "cell path index:", index

    return cell_path_list[index]


# teleoperation workload
def workload_labelling(cell_path, environment_dict):
    workload_label_1toK = 0
    Z_1toK = cell_path_situational_awareness(cell_path, environment_dict)
    for k in range(len(Z_1toK)):
        for i in range(len(Z_1toK[k])):
            if k == 0:
                difference = np.linalg.norm(Z_1toK[k, i, 2:4], 2)
            else:
                difference = np.linalg.norm(Z_1toK[k, i, 2:4] - Z_1toK[k - 1, i, 2:4], 2)

            workload_k_i = 1.0 / (1.0 + np.exp(-difference))
            workload_label_1toK += workload_k_i

    return workload_label_1toK


def optimal_cell_path_label_workload(cell_path_list, environment_dict):
    workload_label_list = []
    for cell_path in cell_path_list:
        workload_j = workload_labelling(cell_path, environment_dict)
        workload_label_list.append(workload_j)

    index = np.argmin(workload_label_list)
    print "cell path index:", index

    return cell_path_list[index]


def optimal_cell_path_mixture(cell_path_list, environment_dict):
    mix_utility_list = []
    for cell_path in cell_path_list:
        label_workload_j = workload_labelling(cell_path, environment_dict)
        tele_workload_j = workload_teleop(cell_path, environment_dict)
        # apen_j = ApEn_path(cell_path, environment_dict)

        mix_utility = label_workload_j * 0.5 + tele_workload_j * 0.5
        mix_utility_list.append(mix_utility)

    index = np.argmin(mix_utility_list)
    print "cell path index:", index

    return cell_path_list[index]


# prob sampling
def optimal_cell_path_prob(cell_path_dict):
    sorted_dict = sorted(cell_path_dict)
    print "sorted path dict", sorted_dict
    probability = [cell_path_dict[path] for path in sorted_dict]
    index = np.where(multinomial.rvs(1, probability) == 1)[0]
    route = sorted_dict[index.item()]
    return route.split(' ')

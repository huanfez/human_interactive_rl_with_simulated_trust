#! /usr/bin/env python2
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import multivariate_normal, norm
import re


def plot_acc_reward(acc_reward_dict, dir):
    print "accumulated reward: ", acc_reward_dict
    interested_betas = acc_reward_dict.keys()
    num_plots = len(acc_reward_dict)
    fig, ax = plt.subplots(num_plots, 1)
    fig.tight_layout()
    for i in range(0, num_plots):
        interested_beta = interested_betas[i]
        ax[i].plot(acc_reward_dict[interested_beta])
        ax[i].set_xlabel('Episode number')
        # ax2[i].set_xticks(range(0, num_itrs))
        ax[0].set_ylabel('Accumulated reward')
    fig.savefig(dir + '/acc_reward.tif', dpi=300, bbox_inches="tight")


# plot the credible interval of all the ith posterior distribution
def plot_credile_interval(posterior_beta_list, posterior_sigma_list, beta_true, dir, num_itrs):
    posterior_list_beta_lb = []
    posterior_list_beta_ub = []
    length = len(posterior_beta_list)
    # print "posterior beta lists:", posterior_beta_list
    # print "posterior sigma lists:", posterior_sigma_list

    print "posterior beta log-determinant:"
    for index in range(0, length):
        beta_lb = posterior_beta_list[index] - 1.98 * np.sqrt(posterior_sigma_list[index].diagonal())
        beta_ub = posterior_beta_list[index] + 1.98 * np.sqrt(posterior_sigma_list[index].diagonal())
        posterior_list_beta_lb.append(beta_lb)
        posterior_list_beta_ub.append(beta_ub)
        print np.log(np.linalg.det(posterior_sigma_list[index]))

    # print "posterior credible lower intervals:", posterior_list_beta_lb
    # print "posterior credible upper intervals:", posterior_list_beta_ub

    iter_num = range(0, num_itrs)
    fig2, ax2 = plt.subplots(5, 1)
    fig2.tight_layout()
    for i in range(0, 5):
        ax2[i].fill_between(iter_num, np.array(posterior_list_beta_lb)[:, i], np.array(posterior_list_beta_ub)[:, i],
                            color='g', alpha=0.6, interpolate=True)
        # ax2[i].plot(iter_num, np.ones(num_itrs) * beta_true[i], color='k')
        ax2[i].set_xlabel('Episode number')
        ax2[i].set_xticks(range(0, num_itrs))
    ax2[0].set_ylabel(r'$\beta_{-1}$')
    ax2[1].set_ylabel(r'$\beta_{0}$')
    ax2[2].set_ylabel(r'$\beta_{1}$')
    ax2[3].set_ylabel(r'$\beta_{2}$')
    ax2[4].set_ylabel(r'$b$')
    fig2.savefig(dir + '/convergence.tif', dpi=300, bbox_inches="tight")


# plot the credible interval of all the ith posterior distribution
def plot_credile_interval_lr(posterior_beta_list, posterior_sigma_list, beta_true, dir, num_itrs):
    posterior_list_beta_lb = []
    posterior_list_beta_ub = []
    length = len(posterior_beta_list)
    print "posterior beta log-determinant:,"
    for index in range(0, length):
        beta_lb = posterior_beta_list[index] - 1.98 * np.sqrt(posterior_sigma_list[index].diagonal())
        beta_ub = posterior_beta_list[index] + 1.98 * np.sqrt(posterior_sigma_list[index].diagonal())
        posterior_list_beta_lb.append(beta_lb)
        posterior_list_beta_ub.append(beta_ub)
        print np.log(np.linalg.det(posterior_sigma_list[index][2:, 2:]))

    iter_num = range(0, num_itrs)
    fig2, ax2 = plt.subplots(5, 1)
    fig2.tight_layout()
    for i in range(0, 5):
        ax2[i].fill_between(iter_num, np.array(posterior_list_beta_lb)[:, i], np.array(posterior_list_beta_ub)[:, i],
                            color='g', alpha=0.6, interpolate=True)
        # ax2[i].plot(iter_num, np.ones(num_itrs) * beta_true[i], color='k')
        ax2[i].set_xlabel('Trial number')
        ax2[i].set_xticks(range(0, num_itrs))
    ax2[0].set_ylabel(r'$\beta_{-1}$')
    ax2[1].set_ylabel(r'$\beta_{0}$')
    ax2[2].set_ylabel(r'$\beta_{1}$')
    ax2[3].set_ylabel(r'$\beta_{2}$')
    ax2[4].set_ylabel(r'$b$')
    fig2.savefig(dir + '/convergence.tif', dpi=300, bbox_inches="tight")


# plot and save figures for the posterior distribution of beata in ith iteration
def plot_policy(policies, rewards_map, iter_th, dir):
    fig, ax = plt.subplots(1, figsize=(5, 5))
    fig.tight_layout()
    plt.gca().invert_yaxis()

    # colormap
    # color_map = colors.ListedColormap(np.array([np.arange(0.2, 1.0, 0.08), np.arange(0.0, 1.0, 0.1),
    #                                             np.arange(0.0, 0.5, 0.05)]).T)
    color_map = colors.ListedColormap(np.array([np.arange(1.0, 0.0, -.05), np.arange(0.0, 0.74, 0.037),
                                                np.arange(0.0, 1.0, 0.05)]).T)
    bound_val = np.arange(-2.0, 2.0, 0.2)
    norm_val = colors.BoundaryNorm(bound_val, color_map.N)
    # data = np.random.rand(5, 5) * 2.0
    # img = ax.imshow(rewards_map, cmap=color_map, norm=norm_val)
    img = ax.imshow(rewards_map, cmap='Spectral', interpolation='nearest')

    # Major ticks every 20, minor ticks every 5
    major_ticks = np.arange(0, 126, 25)
    # minor_ticks = np.arange(0, 126, 5)

    ax.set_xticks(major_ticks)
    ax.tick_params(axis='x', labelsize=20)
    # ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.tick_params(axis='y', labelsize=20)
    # ax.set_yticks(minor_ticks, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    # ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', linestyle='-', linewidth=2, color='k', alpha=0.7)

    # plot arrows based on the optimal policy
    for stat, acts_probs in policies.items():
        # acts_probs = policies[stat]
        mdp_state = re.search(r'(.*)\|', stat).group(1)  # regular expression to get the mdp state
        val_of_mdp_state = int(mdp_state[1:]) - 1  # mdp state in numerical val
        local_pos_x, local_pos_y = val_of_mdp_state % 5, val_of_mdp_state / 5  # grid pos
        plot_pos_x, plot_pos_y = local_pos_x * 25 + 12, local_pos_y * 25 + 12  # image plot pos

        arrows = [(22, 0), (0, 22), (-22, 0), (0, -22), (5, 5)]  # four direction arrows
        arrows_indices = [index for index, ele in enumerate(acts_probs) if ele > 0.0]  # locate arrow direction
        for arrow_index in arrows_indices:
            if arrow_index == 4:
                continue
            arr_x, arr_y = arrows[arrow_index][0], arrows[arrow_index][1]
            plt.arrow(plot_pos_x, plot_pos_y, arr_x, arr_y, width=1.5, length_includes_head=True, head_width=5,
                      head_length=6, color="k")

    # cax = ax.axes([5, 5, 5, 100])
    cbar = plt.colorbar(img, orientation='vertical')
    cbar.ax.tick_params(labelsize=20)
    fig.savefig(dir + '/sim{}_policy.tif'.format(iter_th), dpi=200, bbox_inches="tight")
    plt.close(fig)


# plot and save figures for the posterior distribution of beata in ith iteration
def plot_distribution(Beta0, Sigma0, beta_true, interested_beta, iter_th, dir):
    plt.ioff()
    fig, ax = plt.subplots(len(Beta0) - 2, 1, figsize=(5, 6))
    fig.tight_layout()
    x = np.linspace(norm.ppf(0.1), norm.ppf(0.99), 1000)
    # ax.plot(x, norm.pdf(x, Beta0[0], np.sqrt(Sigma0[0, 0])), 'r-', lw=1, alpha=0.9, label=r'$\beta_{-1}$')
    # ax.plot(x, norm.pdf(x, Beta0[1], np.sqrt(Sigma0[1, 1])), 'g-', lw=1, alpha=0.9, label=r'$\beta_0$')
    ax[0].plot(x, norm.pdf(x, Beta0[2], np.sqrt(Sigma0[2, 2])), '-', lw=1, alpha=0.9, label=r'$\beta_1$')
    ax[1].plot(x, norm.pdf(x, Beta0[3], np.sqrt(Sigma0[3, 3])), '-', lw=1, alpha=0.9, label=r'$\beta_2$')
    ax[2].plot(x, norm.pdf(x, Beta0[4], np.sqrt(Sigma0[4, 4])), '-', lw=1, alpha=0.9, label=r'$b$')
    ax[0].legend(loc='upper right', frameon=False)
    ax[1].legend(loc='upper right', frameon=False)
    ax[2].legend(loc='upper right', frameon=False)
    # ax.set_ylim([1.26e-3, 1.265e-3])
    # ax.ticklabel_format(style='sci')
    # ax[0, 0].set_xlim([-400, 400])
    # ax.set_ylabel('PDF')
    # ax.set_title('Prior')

    # ax.axvline(x=beta_true[0], color='r', linestyle='-.', lw=1.5)
    # ax.axvline(x=beta_true[1], color='g', linestyle='-.', lw=1.5)
    ax[0].axvline(x=beta_true[2], linestyle='-.', lw=1.5)
    ax[1].axvline(x=beta_true[3], linestyle='-.', lw=1.5)
    ax[2].axvline(x=beta_true[4], linestyle='-.', lw=1.5)

    beta_arr= np.array(interested_beta)
    ax[0].plot(sorted(beta_arr[:, 0]), norm.pdf(sorted(beta_arr[:, 0]), Beta0[2], np.sqrt(Sigma0[2, 2])), 'o', lw=1, alpha=0.9, label=r'$\beta_1$')
    ax[1].plot(sorted(beta_arr[:, 1]), norm.pdf(sorted(beta_arr[:, 1]), Beta0[3], np.sqrt(Sigma0[3, 3])), 'o', lw=1, alpha=0.9, label=r'$\beta_2$')
    ax[2].plot(sorted(beta_arr[:, 2]), norm.pdf(sorted(beta_arr[:, 2]), Beta0[4], np.sqrt(Sigma0[4, 4])), 'o', lw=1, alpha=0.9, label=r'$b$')

    # plt.pause(1e-17)
    fig.savefig(dir + '/sim{}_posterior.tif'.format(iter_th), dpi=100, bbox_inches="tight")
    plt.close(fig)

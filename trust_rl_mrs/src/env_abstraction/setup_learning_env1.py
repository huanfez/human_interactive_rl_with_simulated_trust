#! /usr/bin/env python2

import numpy as np


crop_topLeft = (375, 125)  # image position
crop_bottomRight = (495, 245)

cell_height, cell_width = 12, 12

# states : labels
# states_label = {"s1":["neigh"], "s2":["dest"], "s3":["neigh"], "s4":["neigh"], "s5":["neigh"],
#                  "s6":[], "s7":["obs"], "s8":["obs"], "s9":["neigh"], "s10":[],
#                  "s11":[], "s12":["neigh"], "s13":[], "s14":["neigh"], "s15":["neigh"],
#                  "s16":[], "s17":[], "s18":["neigh"], "s19":[], "s20":[],
#                  "s21":[], "s22":[], "s23":[], "s24":[], "s25":[]}

states_label = {"s2": ["neigh"], "s3": ["dest"], "s4": ["neigh"], "s8": ["neigh"], "s9": ["neigh"],
                "s11": ["neigh"], "s17": ["neigh"], "s20": ["neigh"],
                "s21": ["neigh"], "s24": ["neigh"], "s25": ["neigh"], "s27": ["neigh"],
                "s31": ["obs"], "s33": ["obs"], "s34": ["obs"], "s36": ["obs"], "s37": ["obs"], "s40": ["neigh"],
                "s41": ["obs"], "s42": ["neigh"], "s43": ["obs"], "s47": ["neigh"], "s49": ["neigh"], "s50": ["neigh"],
                "s53": ["neigh"], "s55": ["neigh"], "s56": ["neigh"],
                "s69": ["neigh"]
                }

# states_label = {"s39": ["obs"], "s40": ["obs"], "s42": ["obs"], "s43": ["obs"], "s49": ["obs"],
#                 "s51": ["obs"], "s52": ["obs"], "s126": ["dest"]}

# mdp_init_state = "s23"
mdp_init_state = "s86"
# mdp_init_state = "s15"

# # environment: no trees & even
# states_locations = {"s1": (5, 15), "s2": (6, 15), "s3": (7, 15), "s4": (8, 15), "s5": (9, 15),
#                      "s6": (5, 16), "s7": (6, 16), "s8": (7, 16), "s9": (8, 16), "s10": (9, 16),
#                      "s11": (5, 17), "s12": (6, 17), "s13": (7, 17), "s14": (8, 17), "s15": (9, 17),
#                      "s16": (5, 18), "s17": (6, 18), "s18": (7, 18), "s19": (8, 18), "s20": (9, 18),
#                      "s21": (5, 19), "s22": (6, 19), "s23": (7, 19), "s24": (8, 19), "s25": (9, 19)}

# q-learning parameters
gamma = 0.75  # Discount factor
alpha = 0.6  # Learning rate

""" 
active learning: 
Bayesian inference parameters
time-series known and unknown model parameters
"""
# hyper-parameters of prior distribution of trust model
Beta0_itr = np.asarray([0.1, 0.1, 0.2, 0.0, -0.4])
Sigma0_itr = np.diag([1e3, 1e3, 1e3, 1e3, 1e3])
a0_itr = 3.0
b0_itr = 1.0
c0_itr = 3.0
d0_itr = 1.0

# known parameters of trust model
alpha1 = np.diag([1.0, 1.0, 1.0])
alpha2 = -np.diag([1.0, 1.0, 1.0])
Alpha = np.concatenate((alpha1, alpha2), axis=1)

# assumed ground truth of trust model
beta_true3 = np.asarray([0.0125, 0.5, 0.85, 0.13, -0.3])
delta_w_true3 = 0.01
delta_v_true3 = 0.01

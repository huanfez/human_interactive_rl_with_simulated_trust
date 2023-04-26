#! /usr/bin/env python2

import numpy as np


# states : features
states_features = {"s1":[0.2, 0.0], "s2":[0.3, 0.0], "s3":[0.4, 0.0], "s4":[0.0, 0.6], "s5":[0.5, 0.0], "s6":[0.5, 0.0], "s7":[0.3, 0.0], "s8":[0.0, 0.6], "s9":[0.0, 0.6], "s10":[0.3, 0.0],
          "s11":[0.4, 0.0], "s12":[0.8, 0.0], "s13":[0.6, 0.0], "s14":[0.0, 0.7], "s15":[0.7, 0.0], "s16":[0.0, 0.4], "s17":[0.0, 0.8], "s18":[0.0, 0.6], "s19":[0.6, 0.0], "s20":[0.7, 0.0],
          "s21":[0.0, 0.8], "s22":[0.0, 0.7], "s23":[0.0, 0.0], "s24":[0.0, 0.5], "s25":[0.0, 0.6]}

# state : actions
states_actions = {"s1":["E", "S", "c"], "s2":["E", "S", "W", "c"], "s3":["E", "S", "W", "c"], "s4":["E", "S", "W", "c"], "s5":["S", "W", "c"],
          "s6":["E", "S", "N", "c"], "s7":["E", "S", "W", "N", "c"], "s8":["E", "S", "W", "N", "c"], "s9":["E", "S", "W", "N", "c"], "s10":["S", "W", "N", "c"],
          "s11":["E", "S", "N", "c"], "s12":["E", "S", "W", "N", "c"], "s13":["E", "S", "W", "N", "c"], "s14":["E", "S", "W", "N", "c"], "s15":["S", "W", "N", "c"],
          "s16":["E", "S", "N", "c"], "s17":["E", "S", "W", "N", "c"], "s18":["E", "S", "W", "N", "c"], "s19":["E", "S", "W", "N", "c"], "s20":["S", "W", "N", "c"],
          "s21":["E", "N", "c"], "s22":["E", "W", "N", "c"], "s23":["E", "W", "N", "c"], "s24":["E", "W", "N", "c"], "s25":["W", "N", "c"]}

# state1, action : state2, probability, reward
trans_rew = {("s1", "E"): ["s2", 1.0, -np.inf], ("s1", "S"): ["s6", 1.0, -np.inf], ("s1", "c"): ["s1", 1.0, -np.inf],
          ("s2", "E"): ["s3", 1.0, -np.inf], ("s2", "S"): ["s7", 1.0, -np.inf], ("s2", "W"): ["s1", 1.0, -np.inf], ("s2", "c"): ["s2", 1.0, -np.inf],
          ("s3", "E"): ["s4", 1.0, -np.inf], ("s3", "S"): ["s8", 1.0, -np.inf], ("s3", "W"): ["s2", 1.0, -np.inf], ("s3", "c"): ["s3", 1.0, -np.inf],
          ("s4", "E"): ["s5", 1.0, -np.inf], ("s4", "S"): ["s9", 1.0, -np.inf], ("s4", "W"): ["s3", 1.0, -np.inf], ("s4", "c"): ["s4", 1.0, -np.inf],
          ("s5", "S"): ["s10", 1.0, -np.inf], ("s5", "W"): ["s4", 1.0, -np.inf], ("s5", "c"): ["s5", 1.0, -np.inf],
          ("s6", "E"): ["s7", 1.0, -np.inf], ("s6", "S"): ["s11", 1.0, -np.inf], ("s6", "N"): ["s1", 1.0, -np.inf], ("s6", "c"): ["s6", 1.0, -np.inf],
          ("s7", "E"): ["s8", 1.0, -np.inf], ("s7", "S"): ["s12", 1.0, -np.inf], ("s7", "W"): ["s6", 1.0, -np.inf],
          ("s7", "N"): ["s2", 1.0, -np.inf], ("s7", "c"): ["s7", 1.0, -np.inf],
          ("s8", "E"): ["s9", 1.0, -np.inf], ("s8", "S"): ["s13", 1.0, -np.inf], ("s8", "W"): ["s7", 1.0, -np.inf],
          ("s8", "N"): ["s3", 1.0, -np.inf], ("s8", "c"): ["s8", 1.0, -np.inf],
          ("s9", "E"): ["s10", 1.0, -np.inf], ("s9", "S"): ["s14", 1.0, -np.inf], ("s9", "W"): ["s8", 1.0, -np.inf],
          ("s9", "N"): ["s4", 1.0, -np.inf], ("s9", "c"): ["s9", 1.0, -np.inf],
          ("s10", "S"): ["s15", 1.0, -np.inf], ("s10", "W"): ["s9", 1.0, -np.inf], ("s10", "N"): ["s5", 1.0, -np.inf], ("s10", "c"): ["s10", 1.0, -np.inf],
          ("s11", "E"): ["s12", 1.0, -np.inf], ("s11", "S"): ["s16", 1.0, -np.inf], ("s11", "N"): ["s6", 1.0, -np.inf], ("s11", "c"): ["s11", 1.0, -np.inf],
          ("s12", "E"): ["s13", 1.0, -np.inf], ("s12", "S"): ["s17", 1.0, -np.inf], ("s12", "W"): ["s11", 1.0, -np.inf],
          ("s12", "N"): ["s7", 1.0, -np.inf], ("s12", "c"): ["s12", 1.0, -np.inf],
          ("s13", "E"): ["s14", 1.0, -np.inf], ("s13", "S"): ["s18", 1.0, -np.inf], ("s13", "W"): ["s12", 1.0, -np.inf],
          ("s13", "N"): ["s8", 1.0, -np.inf], ("s13", "c"): ["s13", 1.0, -np.inf],
          ("s14", "E"): ["s15", 1.0, -np.inf], ("s14", "S"): ["s19", 1.0, -np.inf], ("s14", "W"): ["s13", 1.0, -np.inf],
          ("s14", "N"): ["s9", 1.0, -np.inf], ("s14", "c"): ["s14", 1.0, -np.inf],
          ("s15", "S"): ["s20", 1.0, -np.inf], ("s15", "W"): ["s14", 1.0, -np.inf], ("s15", "N"): ["s10", 1.0, -np.inf], ("s15", "c"): ["s15", 1.0, -np.inf],
          ("s16", "E"): ["s17", 1.0, -np.inf], ("s16", "S"): ["s21", 1.0, -np.inf], ("s16", "N"): ["s11", 1.0, -np.inf], ("s16", "c"): ["s16", 1.0, -np.inf],
          ("s17", "E"): ["s18", 1.0, -np.inf], ("s17", "S"): ["s22", 1.0, -np.inf], ("s17", "W"): ["s16", 1.0, -np.inf],
          ("s17", "N"): ["s12", 1.0, -np.inf], ("s17", "c"): ["s17", 1.0, -np.inf],
          ("s18", "E"): ["s19", 1.0, -np.inf], ("s18", "S"): ["s23", 1.0, -np.inf], ("s18", "W"): ["s17", 1.0, -np.inf],
          ("s18", "N"): ["s13", 1.0, -np.inf], ("s18", "c"): ["s18", 1.0, -np.inf],
          ("s19", "E"): ["s20", 1.0, -np.inf], ("s19", "S"): ["s24", 1.0, -np.inf], ("s19", "W"): ["s18", 1.0, -np.inf],
          ("s19", "N"): ["s14", 1.0, -np.inf], ("s19", "c"): ["s19", 1.0, -np.inf],
          ("s20", "S"): ["s25", 1.0, -np.inf], ("s20", "W"): ["s19", 1.0, -np.inf], ("s20", "N"): ["s15", 1.0, -np.inf], ("s20", "c"): ["s20", 1.0, -np.inf],
          ("s21", "E"): ["s22", 1.0, -np.inf], ("s21", "N"): ["s16", 1.0, -np.inf], ("s21", "c"): ["s21", 1.0, -np.inf],
          ("s22", "E"): ["s23", 1.0, -np.inf], ("s22", "W"): ["s21", 1.0, -np.inf], ("s22", "N"): ["s17", 1.0, -np.inf], ("s22", "c"): ["s22", 1.0, -np.inf],
          ("s23", "E"): ["s24", 1.0, -np.inf], ("s23", "W"): ["s22", 1.0, -np.inf], ("s23", "N"): ["s18", 1.0, -np.inf], ("s23", "c"): ["s23", 1.0, -np.inf],
          ("s24", "E"): ["s25", 1.0, -np.inf], ("s24", "W"): ["s23", 1.0, -np.inf], ("s24", "N"): ["s19", 1.0, -np.inf], ("s24", "c"): ["s24", 1.0, -np.inf],
          ("s25", "W"): ["s24", 1.0, -np.inf], ("s25", "N"): ["s20", 1.0, -np.inf], ("s25", "c"): ["s25", 1.0, -np.inf]}

# states : labels
states_label = {"s1":[], "s2":[], "s3":[], "s4":[], "s5":[],
                 "s6":[], "s7":["obs"], "s8":["obs"], "s9":[], "s10":[],
                 "s11":[], "s12":[], "s13":[], "s14":[], "s15":[],
                 "s16":[], "s17":[], "s18":[], "s19":[], "s20":[],
                 "s21":[], "s22":[], "s23":["dest"], "s24":[], "s25":[]}

mdp_init_state = "s2"

# environment: no trees & even
states_locations = {"s1": (5, 15), "s2": (6, 15), "s3": (7, 15), "s4": (8, 15), "s5": (9, 15),
                     "s6": (5, 16), "s7": (6, 16), "s8": (7, 16), "s9": (8, 16), "s10": (9, 16),
                     "s11": (5, 17), "s12": (6, 17), "s13": (7, 17), "s14": (8, 17), "s15": (9, 17),
                     "s16": (5, 18), "s17": (6, 18), "s18": (7, 18), "s19": (8, 18), "s20": (9, 18),
                     "s21": (5, 19), "s22": (6, 19), "s23": (7, 19), "s24": (8, 19), "s25": (9, 19)}

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
Sigma0_itr = np.diag([1e0, 1e0, 1e0, 1e0, 1e0])
a0_itr = 3.0
b0_itr = 1.0
c0_itr = 3.0
d0_itr = 1.0

# known parameters of trust model
alpha1 = np.diag([1.0, 1.0, 1.0])
alpha2 = -np.diag([1.0, 1.0, 1.0])
Alpha = np.concatenate((alpha1, alpha2), axis=1)

# assumed ground truth of trust model
beta_true3 = np.asarray([0.0125, 1.0, 0.5, 0.5, -0.1])
delta_w_true3 = 0.0001
delta_v_true3 = 0.0001

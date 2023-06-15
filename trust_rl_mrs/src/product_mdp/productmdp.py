# coding=utf-8
# !/usr/bin/env python2

import queue
import re


def product_mdp(lmdp, automaton):
    state_features = {}
    state_actions = {}
    trans_rewards = {}
    pm_init_state = ""
    pm_acc_states = []

    aut_acc_states = automaton.get_acc_states(automaton.graph)
    state_list = queue.Queue()  # queue for temporary state

    label_of_init_state = lmdp.state_labels[lmdp.init_state]
    init_next_aut_states = get_aut_successors(automaton.init_state, automaton)
    # print "neighbors:,", init_next_aut_states
    for next_aut_state in init_next_aut_states:
        cur_aut_transition = automaton.graph[automaton.init_state][next_aut_state][0]['label']
        trans_flag = parse(cur_aut_transition, label_of_init_state)
        # print "flag:,", trans_flag, cur_aut_transition, label_of_init_state
        if trans_flag:
            pm_init_state = lmdp.init_state + "|" + next_aut_state  # initial state
            print "product automaton initial state:", pm_init_state
            state_list.put(pm_init_state)  # push queue
            state_features[pm_init_state] = lmdp.state_features[lmdp.init_state]  # visited states

    while not state_list.empty():
        curr_state = state_list.get()  # pop out state
        cur_mdp_state = re.search(r'(.*)\|', curr_state).group(1)  # break to get the mdp current state
        cur_aut_state = re.search(r'\|(.*)', curr_state).group(1)  # break to get the automaton current state

        if cur_aut_state in aut_acc_states:
            pm_acc_states.append(curr_state)

        # get successors of current state
        next_mdp_states = get_mdp_successors(cur_mdp_state, lmdp)
        next_aut_states = get_aut_successors(cur_aut_state, automaton)
        if next_mdp_states == [] or next_aut_states == []:
            continue

        # push successors into queue: BFS
        for next_mdp_state in next_mdp_states:
            for next_aut_state in next_aut_states:
                label_of_next_state = lmdp.state_labels[next_mdp_state]
                cur_aut_transition = automaton.graph[cur_aut_state][next_aut_state][0]['label']
                trans_flag = parse(cur_aut_transition, label_of_next_state)

                if not trans_flag:
                    continue

                new_state = next_mdp_state + "|" + next_aut_state
                if new_state not in state_features.keys():
                    state_list.put(new_state)
                    state_features[new_state] = lmdp.state_features[next_mdp_state]  # add the state features

                state_actions, trans_rewards = update_action_trans_rewards(cur_mdp_state, cur_aut_state, next_mdp_state,
                                                                           next_aut_state, lmdp, automaton, curr_state,
                                                                           new_state, state_actions, trans_rewards)

    product_mdp_results = {"state_features": state_features, "state_actions": state_actions,
                           "state_actions_rewards": trans_rewards, "init_state": pm_init_state, "acc_states": pm_acc_states}

    return product_mdp_results


def get_mdp_successors(mdp_cur_state, lmdp):
    mdp_next_states = []
    # obtain the neighbor states for the mdp
    for action in lmdp.actions[mdp_cur_state]:
        mdp_next_state = lmdp.get_next_state(mdp_cur_state, action)
        mdp_next_states.append(mdp_next_state)

    return mdp_next_states


def get_aut_successors(aut_cur_state, automaton):
    # aut_next_states = []
    #
    # # obtain the neighbor states for the automaton
    # print "current state:,", aut_cur_state, "neighbors:", list(automaton.graph.neighbors(aut_cur_state))
    # for aut_next_state in automaton.graph.neighbors(aut_cur_state):
    #     aut_next_states.append(aut_next_state)

    return list(automaton.graph.neighbors(aut_cur_state))


def parse(bool_expression, labels):
    if bool_expression == '1':
        return True

    proposition_state = False
    propositions = []
    lor_pos = [m.start() for m in re.finditer(r'\|', bool_expression)]
    if len(lor_pos):
        lor_pos.insert(0, -1)
        lor_pos.append(len(bool_expression))
        for index in range(len(lor_pos) - 1):
            substring = bool_expression[lor_pos[index]+1: lor_pos[index+1]]
            propositions.append(substring)
    else:
        propositions = [bool_expression]

    for prop in propositions:
        aps = re.findall(r'[!]*[a-z]+[0-9]*', prop)
        proposition_state = parser_aps(aps, labels) or proposition_state

    return proposition_state


def parser_aps(aps, labels):
    for ap in aps:
        if ap[0] == '!' and ap[1:] in labels:  # ap of automaton in negation format, but label is true
            return False
        elif ap[0] != '!' and ap not in labels:  # ap of automaton in positive format, but label is false
            return False

    return True


def update_action_trans_rewards(mdp_curr_state, aut_curr_state, mdp_next_state, aut_next_state, lmdp, automaton,
                                curr_state, new_state, prod_mdp_state_actions, prod_mdp_trans_rewards):
    if curr_state not in prod_mdp_trans_rewards.keys():
        prod_mdp_trans_rewards[curr_state] = {}

    for action in lmdp.actions[mdp_curr_state]:
        if lmdp.get_next_state(mdp_curr_state, action) != mdp_next_state:
            continue

        if curr_state in prod_mdp_state_actions.keys():
            prod_mdp_state_actions[curr_state].append(action)
        else:
            prod_mdp_state_actions[curr_state] = [action]

        prod_mdp_trans_rewards[curr_state][action] = [new_state, lmdp.trans_rewards[mdp_curr_state][action][1],
                                                      lmdp.trans_rewards[mdp_curr_state][action][2]]

    return prod_mdp_state_actions, prod_mdp_trans_rewards

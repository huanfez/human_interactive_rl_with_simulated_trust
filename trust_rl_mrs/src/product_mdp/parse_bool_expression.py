#! /usr/bin/env python2

import re


def parse2(bool_expression, labels):
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

        print "cleaned string:", propositions
    else:
        propositions = [bool_expression]

    for prop in propositions:
        aps = re.findall(r'[!]*[a-z]+[0-9]*', prop)
        print "aps:", aps
        proposition_state = parser_aps(aps, labels) or proposition_state

    return proposition_state


def parse1(bool_expression, labels):
    proposition_state = False
    propositions = re.findall(r'\((.*?)\)', bool_expression)
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


aut_labels = ['p1', 'p0']
bool_expr = '(p0 & p2)'
print "viable:", parse2(bool_expr, aut_labels)

mdp = {'s1':{'label':['red', 'stop'], 'traver':0.9, 'visa':0.3, 'trans':{'a1':{'s2':{'p':0.4, 'r':10.0},
                                                                               's3':{'p':0.6, 'r':12.0}},
                                                                         'a2':{'s2':[0.4, 10.0], 's3':[0.6, 12.0]}}},
       's2':{'label':['green'], 'traver':0.8, 'visa':0.6, 'trans':{'a1':{'s1':[0.4, 10.0], 's2':[0.6, 12.0]},
                                                                   'a2':{'s1':[0.4, 10.0], 's2':[0.6, 12.0]}}}
       }
print "mdp:", mdp
print "states:", mdp.keys()
print "edge:", mdp['s1']['trans'].keys(), mdp['s1']['trans']['a1'], mdp['s1']['trans']['a1']['s2']
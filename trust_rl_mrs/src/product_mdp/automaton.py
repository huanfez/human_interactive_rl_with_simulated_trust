# !/usr/bin/env python2

import networkx as nx


class Automaton:
    def __init__(self, graph):
        self.init_state = graph['I'].keys()[0]
        self.acc_states = self.get_acc_states(graph)

        graph.remove_node('I')
        self.graph = graph

    def get_acc_states(self, graph):
        # Find the states labelled with peripheries=2
        acc_states = []
        for node in list(graph.nodes):
            labels = graph.node[node].keys()
            if 'peripheries' in labels:
                acc_states.append(node)
        return acc_states

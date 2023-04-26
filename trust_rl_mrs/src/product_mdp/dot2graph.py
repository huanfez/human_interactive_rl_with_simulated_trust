# coding=utf-8
# !/usr/bin/env python2

import pygraphviz
from networkx.drawing import nx_agraph
import networkx as nx

from src.trust_rl_mrs.src.product_mdp import automaton

dotFormat = """
digraph "(!p2 U G!p1) | (!p0 & (!p0 W !p1)) | (p0 & (p0 M p1))" {
  rankdir=LR
  label="\n[Büchi]"
  labelloc="t"
  node [shape="circle"]
  I [label="", style=invis, width=0]
  I -> 0
  0 [label="0"]
  0 -> 1 [label="(!p0 & !p1) | (p0 & p1)"]
  0 -> 2 [label="p0 & !p1"]
  0 -> 3 [label="(p0 & !p1 & !p2) | (!p0 & p1 & !p2)"]
  0 -> 4 [label="!p0 & p1"]
  1 [label="1", peripheries=2]
  1 -> 1 [label="1"]
  2 [label="2", peripheries=2]
  2 -> 2 [label="!p1"]
  3 [label="3"]
  3 -> 2 [label="!p1"]
  3 -> 3 [label="!p2"]
  4 [label="4", peripheries=2]
  4 -> 1 [label="!p1"]
  4 -> 4 [label="!p0 & p1"]
}
"""

G = nx_agraph.from_agraph(pygraphviz.AGraph(dotFormat))
nx_agraph.view_pygraphviz(G)
print "nodes:", list(G.nodes)
print "edges:", list(G.edges)

aut1 = automaton.Automaton(G)
print "initial state: ", aut1.init_state, "final states:", aut1.acc_states
print "nodes:", list(aut1.graph.nodes)

print "edge between 0 and 3:", G['0']['3']
print "edge label between 0 and 3:", G['0']['3'][0]['label']
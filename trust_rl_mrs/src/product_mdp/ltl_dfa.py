#! /usr/bin/env python2
# coding=utf-8

dfa0_dotFormat = """
    digraph "Fdest" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="circle"]
      I [label="", style=invis, width=0]
      I -> 1
      0 [label="0", peripheries=2]
      0 -> 0 [label="1"]
      1 [label="1"]
      1 -> 0 [label="dest"]
      1 -> 1 [label="!dest"]
    }
"""

dfa_dotFormat = """
    digraph "Fneigh & XFdest & G(!obs & (!dest | G!neigh))" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="circle"]
      I [label="", style=invis, width=0]
      I -> 2
      0 [label="0", peripheries=2]
      0 -> 0 [label="!neigh & !obs"]
      1 [label="1"]
      1 -> 0 [label="dest & !neigh & !obs"]
      1 -> 1 [label="!dest & !obs"]
      2 [label="2"]
      2 -> 1 [label="!dest & neigh & !obs"]
      2 -> 2 [label="!dest & !neigh & !obs"]
    }
"""

dfa1_dotFormat = """
    digraph "Fdest & G!obs" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="circle"]
      I [label="", style=invis, width=0]
      I -> 1
      0 [label="0", peripheries=2]
      0 -> 0 [label="!obs"]
      1 [label="1"]
      1 -> 0 [label="dest & !obs"]
      1 -> 1 [label="!dest & !obs"]
    }
"""

dfa2_dotFormat = """digraph "Fdest & G(!obs & (!peak | Xpeak) & (peak | X!peak))" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="circle"]
      I [label="", style=invis, width=0]
      I -> 2
      0 [label="0", peripheries=2]
      0 -> 0 [label="!obs & peak"]
      1 [label="1", peripheries=2]
      1 -> 1 [label="!obs & !peak"]
      2 [label="2"]
      2 -> 0 [label="dest & !obs & peak"]
      2 -> 1 [label="dest & !obs & !peak"]
      2 -> 3 [label="!dest & !obs & !peak"]
      2 -> 4 [label="!dest & !obs & peak"]
      3 [label="3"]
      3 -> 1 [label="dest & !obs & !peak"]
      3 -> 3 [label="!dest & !obs & !peak"]
      4 [label="4"]
      4 -> 0 [label="dest & !obs & peak"]
      4 -> 4 [label="!dest & !obs & peak"]
    }
"""

dfa3_dotFormat = """
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
    }
"""

dfa4_dotFormat = """
    digraph "Fmid & XFdest & G(!obs & (peak | X!peak) & (!peak | Xpeak))" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="circle"]
      I [label="", style=invis, width=0]
      I -> 2
      0 [label="0", peripheries=2]
      0 -> 0 [label="!obs & !peak"]
      1 [label="1", peripheries=2]
      1 -> 1 [label="!obs & peak"]
      2 [label="2"]
      2 -> 3 [label="mid & !obs & !peak"]
      2 -> 4 [label="!mid & !obs & !peak"]
      2 -> 5 [label="mid & !obs & peak"]
      2 -> 6 [label="!mid & !obs & peak"]
      3 [label="3"]
      3 -> 0 [label="dest & !obs & !peak"]
      3 -> 3 [label="!dest & !obs & !peak"]
      4 [label="4"]
      4 -> 0 [label="dest & mid & !obs & !peak"]
      4 -> 3 [label="!dest & mid & !obs & !peak"]
      4 -> 4 [label="!dest & !mid & !obs & !peak"]
      4 -> 7 [label="dest & !mid & !obs & !peak"]
      5 [label="5"]
      5 -> 1 [label="dest & !obs & peak"]
      5 -> 5 [label="!dest & !obs & peak"]
      6 [label="6"]
      6 -> 1 [label="dest & mid & !obs & peak"]
      6 -> 5 [label="!dest & mid & !obs & peak"]
      6 -> 6 [label="!dest & !mid & !obs & peak"]
      6 -> 8 [label="dest & !mid & !obs & peak"]
      7 [label="7"]
      7 -> 0 [label="mid & !obs & !peak"]
      7 -> 7 [label="!mid & !obs & !peak"]
      8 [label="8"]
      8 -> 1 [label="mid & !obs & peak"]
      8 -> 8 [label="!mid & !obs & peak"]
    }
"""

dfa5_dotFormat = """
  digraph "Fmid & XFdest & G(!obs & !peak)" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="circle"]
      I [label="", style=invis, width=0]
      I -> 1
      0 [label="0", peripheries=2]
      0 -> 0 [label="!obs & !peak"]
      1 [label="1"]
      1 -> 2 [label="mid & !obs & !peak"]
      1 -> 3 [label="!mid & !obs & !peak"]
      2 [label="2"]
      2 -> 0 [label="dest & !obs & !peak"]
      2 -> 2 [label="!dest & !obs & !peak"]
      3 [label="3"]
      3 -> 0 [label="dest & mid & !obs & !peak"]
      3 -> 2 [label="!dest & mid & !obs & !peak"]
      3 -> 3 [label="!dest & !mid & !obs & !peak"]
      3 -> 4 [label="dest & !mid & !obs & !peak"]
      4 [label="4"]
      4 -> 0 [label="mid & !obs & !peak"]
      4 -> 4 [label="!mid & !obs & !peak"]
}
"""

dfa_tr_dotFormat = """
    digraph "F(green | red)" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="circle"]
      I [label="", style=invis, width=0]
      I -> 1
      0 [label="0", peripheries=2]
      0 -> 0 [label="1"]
      1 [label="1"]
      1 -> 0 [label="green | red"]
      1 -> 1 [label="!green & !red"]
    }
"""

dfa_demo = """
    digraph "a & X(b & X(F(d & Xg) & F(e & Xg) & F(f & Xg)))" {
      rankdir=LR
      label="\n[Büchi]"
      labelloc="t"
      node [shape="ellipse",width="0.5",height="0.5"]
      I [label="", style=invis, width=0]
      I -> 0
      0 [label="0"]
      0 -> 1 [label="a"]
      1 [label="1"]
      1 -> 2 [label="b"]
      2 [label="2"]
      2 -> 2 [label="1"]
      2 -> 3 [label="d & e & f"]
      2 -> 4 [label="!d & !e & f"]
      2 -> 5 [label="!d & e & !f"]
      2 -> 6 [label="!d & e & f"]
      2 -> 7 [label="d & !e & !f"]
      2 -> 8 [label="d & !e & f"]
      2 -> 9 [label="d & e & !f"]
      3 [label="3"]
      3 -> 10 [label="g"]
      4 [label="4"]
      4 -> 3 [label="d & e & !f & g"]
      4 -> 6 [label="!d & e & !f & g"]
      4 -> 8 [label="d & !e & !f & g"]
      4 -> 11 [label="g"]
      5 [label="5"]
      5 -> 3 [label="d & !e & f & g"]
      5 -> 6 [label="!d & !e & f & g"]
      5 -> 9 [label="d & !e & !f & g"]
      5 -> 12 [label="g"]
      6 [label="6"]
      6 -> 3 [label="(d & !e & g) | (d & !f & g)"]
      6 -> 13 [label="g"]
      7 [label="7"]
      7 -> 3 [label="!d & e & f & g"]
      7 -> 8 [label="!d & !e & f & g"]
      7 -> 9 [label="!d & e & !f & g"]
      7 -> 14 [label="g"]
      8 [label="8"]
      8 -> 3 [label="(e & !f & g) | (!d & e & g)"]
      8 -> 15 [label="g"]
      9 [label="9"]
      9 -> 3 [label="(!e & f & g) | (!d & f & g)"]
      9 -> 16 [label="g"]
      10 [label="10", peripheries=2]
      10 -> 10 [label="1"]
      11 [label="11"]
      11 -> 3 [label="d & e & !f"]
      11 -> 6 [label="!d & e & !f"]
      11 -> 8 [label="d & !e & !f"]
      11 -> 11 [label="1"]
      12 [label="12"]
      12 -> 3 [label="d & !e & f"]
      12 -> 6 [label="!d & !e & f"]
      12 -> 9 [label="d & !e & !f"]
      12 -> 12 [label="1"]
      13 [label="13"]
      13 -> 3 [label="(d & !e) | (d & !f)"]
      13 -> 13 [label="1"]
      14 [label="14"]
      14 -> 3 [label="!d & e & f"]
      14 -> 8 [label="!d & !e & f"]
      14 -> 9 [label="!d & e & !f"]
      14 -> 14 [label="1"]
      15 [label="15"]
      15 -> 3 [label="(e & !f) | (!d & e)"]
      15 -> 15 [label="1"]
      16 [label="16"]
      16 -> 3 [label="(!e & f) | (!d & f)"]
      16 -> 16 [label="1"]
    }
"""

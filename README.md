# human_interactive_rl_with_simulated_trust

Multi-robot boudning overwatch

## 0. Bounding overwatch task

(1) "select the easy way to go" makes it more like a traveling overwatch rather than a bounding overwatch

![alt text](https://github.com/huanfez/robotsBoundingOverwatch/blob/main/default_gzclient_camera(1)-2022-04-22T16_08_11.130434.jpg?raw=true)

## 1. temporal logic task

(1) prepare the color map to decide where are the obstacles

LTL formula 1: `F dest & G !obs`

![alt text](https://github.com/huanfez/robotsBoundingOverwatch/blob/main/buchi_automaton1.png?raw=true)

(2) prepare the visibility map to decide how to move

(3) may have other specifications?

(4) use spot in Ubuntu to convert LTL to buchi automaton 

```
ltl2tgba -b 'F mid && X F dest && G ! obs' -d
```

(5) commands for extracting the proposition on the edge of automaton `aut` (from spot):

```
aut.graph[node1][node2][0]['label']
```


## 2. active reinforcement learning

(1) first to solve the reinforcement learning with Bayesian optimization (decision field theory) "no opponent"

(2) then consider whether to join the opponent

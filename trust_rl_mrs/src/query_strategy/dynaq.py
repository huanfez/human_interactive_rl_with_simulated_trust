#! /usr/bin/env python2

import numpy as np
import copy


class DynaQAgent():
    def __init__(self, agent_info):
        """Setup for the agent called when the experiment first starts.

        Args:
            agent_info (dict), the parameters used to initialize the agent. The dictionary contains:
            {
                assumed_model (dict): assumed (state, action, next state, probability, reward)
                actions (list): action list
                epsilon (float): The parameter for epsilon-greedy exploration,
                step_size (float): The step-size,
                discount (float): The discount factor,
                planning_steps (int): The number of planning steps per environmental interaction

                random_seed (int): the seed for the RNG used in epsilon-greedy
                planning_random_seed (int): the seed for the RNG used in the planner
            }
        """

        # First, we get the relevant information from agent_info
        # NOTE: we use np.random.RandomState(seed) to set the two different RNGs
        # for the planner and the rest of the code
        self.gamma = agent_info.get("discount", 0.95)
        self.step_size = agent_info.get("step_size", 0.1)
        self.epsilon = agent_info.get("epsilon", 0.1)
        self.planning_steps = agent_info.get("planning_steps", 100)

        self.rand_generator = np.random.RandomState(agent_info.get('random_seed', 30))
        self.planning_rand_generator = np.random.RandomState(agent_info.get('planning_random_seed', 30))

        # Next, we initialize the attributes required by the agent, e.g., q_values, model, etc.
        # A simple way to implement the model is to have a dictionary of dictionaries,
        # mapping each state to a dictionary which maps actions to (reward, next state) tuples.
        self.actions = agent_info.get("actions")
        self.acc_states = agent_info.get("acc_states")

        # self.param_past_action = {param: "" for param in agent_info.get("assumed_model").keys()}
        # self.param_past_state = {param: "" for param in agent_info.get("assumed_model").keys()}
        self.param_model = agent_info.get("assumed_model")  # model is a dictionary of dictionaries which maps states to
        # actions to (next_state, reward) tuples

        self.param_q_values = self.init_param_q_values()

    def init_param_q_values(self):
        param_model = copy.deepcopy(self.param_model)
        param_q_values = {}
        for param in param_model.keys():
            param_q_values[param] = {}
            for state in param_model[param].keys():
                param_q_values[param][state] = {}
                for action in param_model[param][state].keys():
                    param_q_values[param][state][action] = 0.0

        return param_q_values

    def update_model(self, past_state, past_action, state, param_reward):
        """updates the model

        Args:
            past_state       (string): s
            past_action      (string): a
            state            (string): s'
            param_reward     (dict): beta: r
        Returns:
            Nothing
        """
        # Update the model with the (s,a,s',r) tuple (1~4 lines)
        for param in param_reward.keys():
            if past_state not in self.param_model[param].keys():
                self.param_model[param][past_state] = {}

            self.param_model[param][past_state][past_action] = [copy.deepcopy(state), 1.0, copy.deepcopy(param_reward[param])]

    def planning_step(self, sampled_params):
        """performs planning, i.e. indirect RL.

        Args:
            sampled_params   (list): beta
        Returns:
            Nothing
        """

        # The indirect RL step:
        # - Choose a state and action from the set of experiences that are stored in the model.
        # - Query the model with this state-action pair for the predicted next state and reward.
        # - Update the action values with this simulated experience.
        # - Repeat for the required number of planning steps.
        #
        # Note that the update equation is different for terminal and non-terminal transitions.
        #
        # Important: remember you have a random number generator 'planning_rand_generator' as
        #     a part of the class which you need to use as self.planning_rand_generator.choice()
        for _ in range(self.planning_steps):
            for param in sampled_params:
                past_state = self.planning_rand_generator.choice(list(self.param_model[param].keys()))
                past_action = self.planning_rand_generator.choice(list(self.param_model[param][past_state].keys()))
                state_prob_reward = self.param_model[param][past_state][past_action]
                state, reward = state_prob_reward[0], state_prob_reward[2]
                if state not in self.acc_states:
                    self.param_q_values[param][past_state][past_action] += self.step_size * (reward + self.gamma *
                        np.max(self.param_q_values[param][state].values()) - self.param_q_values[param][past_state][past_action])
                else:
                    self.param_q_values[param][past_state][past_action] += self.step_size * (reward - self.param_q_values[param][past_state][past_action])

    def argmax(self, q_values):
        """argmax with random tie-breaking
        Args:
            q_values (dict): the array of action values
        Returns:
            action (string): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for act in q_values.keys():
            if q_values[act] > top:
                top = q_values[act]
                ties = []

            if q_values[act] == top:
                ties.append(act)

        return self.rand_generator.choice(ties)

    def choose_action_egreedy(self, state, q_values):
        """returns an action using an epsilon-greedy policy w.r.t. the current action-value function.

        Important: assume you have a random number generator 'rand_generator' as a part of the class
                    which you can use as self.rand_generator.choice() or self.rand_generator.rand()

        Args:
            state (string): state of the agent
            q_values (dict): a dict of dict, state action values
        Returns:
            The action taken w.r.t. the aforementioned epsilon-greedy policy
        """
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.choice(q_values[state].keys())
        else:
            values = q_values[state]
            action = self.argmax(values)

        return action

    def agent_start(self, state, params):
        """The first method called when the experiment starts, called after the environment starts.
        Args:
            state (string): the state from the environment's env_start function.
            params (list): parameters of reward function
        Returns:
            actions (dict): the first action the agent takes with respect to different params.
        """
        # given the state, select the action using self.choose_action_egreedy()),
        # and save current state and action
        param_past_state = {}
        param_past_action = {}
        for param in params:
            action = self.choose_action_egreedy(state, self.param_q_values[param])
            param_past_state[param] = state
            param_past_action[param] = action

        return param_past_state, param_past_action

    def agent_step(self, param_past_state, param_past_action, param_state, param_reward, params):
        """A step taken by the agent.

        Args:
            param_reward (dict): the reward received for taking the last action taken
            param_state (dict): the state from the environment's step based on where the agent ended up after the
                last step
            params (list): parameters of reward function
        Returns:
            (dict) The action the agent takes given this state with respect to different params.
        """

        # - Direct-RL step (~1-3 lines)
        # - Model Update step (~1 line)
        # - `planning_step` (~1 line)
        # - Action Selection step (~1 line)
        # Save the current state and action before returning the action to be performed. (~2 lines)
        for param in params:
            past_state = copy.deepcopy(param_past_state[param])
            past_action = copy.deepcopy(param_past_action[param])
            state = param_state[param]
            self.param_q_values[param][past_state][past_action] += self.step_size * (param_reward[param] + self.gamma *
                np.max(self.param_q_values[param][state].values()) - self.param_q_values[param][past_state][past_action])

            # self.update_model(past_state, past_action, state, param_reward)

        self.planning_step(params)

        param_past_state = {}
        param_past_action = {}
        for param in params:
            state = param_state[param]
            action = self.choose_action_egreedy(state, self.param_q_values[param])

            param_past_state[param] = copy.deepcopy(state)
            param_past_action[param] = copy.deepcopy(action)

        return param_past_state, param_past_action

    def agent_end(self, param_past_state, param_past_action, accept_state, param_reward, params):
        """Called when the agent terminates.

        Args:
            param_reward (dict): the reward the agent received for entering the
                terminal state.
            accept_state (string): the accepting state
            params (list): parameters of reward function
        """

        # - Direct RL update with this final transition (1~2 lines)
        # - Model Update step with this final transition (~1 line)
        # - One final `planning_step` (~1 line)
        #
        # Note: the final transition needs to be handled carefully. Since there is no next state,
        #       you will have to pass a dummy state (like -1), which you will be using in the planning_step() to
        #       differentiate between updates with usual terminal and non-terminal transitions.
        for param in params:
            past_state = param_past_state[param]
            past_action = param_past_action[param]
            self.param_q_values[param][past_state][past_action] += self.step_size * (
                    param_reward[param] - self.param_q_values[param][past_state][past_action])

            # self.update_model(past_state, past_action, accept_state, param_reward)

        self.planning_step(params)


# def run_experiment(env, agent, env_parameters, agent_parameters, exp_parameters):
#     # Experiment settings
#     num_runs = exp_parameters['num_runs']
#     num_episodes = exp_parameters['num_episodes']
#     planning_steps_all = agent_parameters['planning_steps']
#
#     env_info = env_parameters
#     agent_info = {"num_states": agent_parameters["num_states"],  # We pass the agent the information it needs.
#                   "num_actions": agent_parameters["num_actions"],
#                   "epsilon": agent_parameters["epsilon"],
#                   "discount": env_parameters["discount"],
#                   "step_size": agent_parameters["step_size"]}
#
#     all_averages = np.zeros((len(planning_steps_all), num_runs, num_episodes))  # for collecting metrics
#     log_data = {'planning_steps_all': planning_steps_all}  # that shall be plotted later
#
#     for idx, planning_steps in enumerate(planning_steps_all):
#
#         print('Planning steps : ', planning_steps)
#         os.system('sleep 0.5')  # to prevent tqdm printing out-of-order before the above print()
#         agent_info["planning_steps"] = planning_steps
#
#         for i in tqdm(range(num_runs)):
#
#             agent_info['random_seed'] = i
#             agent_info['planning_random_seed'] = i
#
#             rl_glue = RLGlue(env, agent)  # Creates a new RLGlue experiment with the env and agent we chose above
#             rl_glue.rl_init(agent_info,
#                             env_info)  # We pass RLGlue what it needs to initialize the agent and environment
#
#             for j in range(num_episodes):
#
#                 rl_glue.rl_start()  # We start an episode. Here we aren't using rl_glue.rl_episode()
#                 # like the other assessments because we'll be requiring some
#                 is_terminal = False  # data from within the episodes in some of the experiments here
#                 num_steps = 0
#                 while not is_terminal:
#                     reward, _, action, is_terminal = rl_glue.rl_step()  # The environment and agent take a step
#                     num_steps += 1  # and return the reward and action taken.
#
#                 all_averages[idx][i][j] = num_steps
#
#     log_data['all_averages'] = all_averages
#     np.save("results/Dyna-Q_planning_steps", log_data)


# def plot_steps_per_episode(file_path):
#     data = np.load(file_path).item()
#     all_averages = data['all_averages']
#     planning_steps_all = data['planning_steps_all']
#
#     for i, planning_steps in enumerate(planning_steps_all):
#         plt.plot(np.mean(all_averages[i], axis=0), label='Planning steps = ' + str(planning_steps))
#
#     plt.legend(loc='upper right')
#     plt.xlabel('Episodes')
#     plt.ylabel('Steps\nper\nepisode', rotation=0, labelpad=40)
#     plt.axhline(y=16, linestyle='--', color='grey', alpha=0.4)
#     plt.show()
#
#
# # Experiment parameters
# experiment_parameters = {
#     "num_runs": 30,  # The number of times we run the experiment
#     "num_episodes": 40,  # The number of episodes per experiment
# }
#
# # Environment parameters
# environment_parameters = {
#     "discount": 0.95,
# }
#
# # Agent parameters
# agent_parameters = {
#     "num_states": 54,
#     "num_actions": 4,
#     "epsilon": 0.1,
#     "step_size": 0.125,
#     "planning_steps": [0, 5, 50]  # The list of planning_steps we want to try
# }
#
# current_env = ShortcutMazeEnvironment  # The environment
# current_agent = DynaQAgent  # The agent
#
# run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)
# plot_steps_per_episode('results/Dyna-Q_planning_steps.npy')
# shutil.make_archive('results', 'zip', 'results');

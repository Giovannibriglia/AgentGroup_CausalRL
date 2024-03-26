import numpy as np
from gymnasium.spaces import Discrete
import random
import global_variables


# TODO: the problem for the causality is on the deltaX and deltaY of the actions, check this aspect in the causal table
class RandomAgent:
    def __init__(self, dict_env_parameters: dict, dict_learning_parameters: dict, dict_other_params: dict,
                 kind_of_alg: str, exploration_strategy: str):

        self.rows = dict_env_parameters['rows']
        self.cols = dict_env_parameters['cols']
        self.n_actions = int(dict_env_parameters['n_actions'])
        self.action_space = Discrete(self.n_actions, start=0)
        self.seed_value = dict_env_parameters['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

        self.kind_of_alg = global_variables.LABEL_RANDOM_AGENT

    def update_Q_or_memory(self, state, action, reward, next_state):
        pass

    def update_exp_fact(self, episode):
        pass

    def select_action(self, current_state: np.ndarray, enemies_nearby_agent: np.ndarray = None, goals_nearby_agent: np.ndarray = None) -> int:
        action = np.random.randint(0, self.n_actions, size=1)[0]
        return action


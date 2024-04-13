import math
import random
import time

import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete
from scripts.utils import exploration_strategies
import global_variables
import warnings


class QLearningAgent:
    def __init__(self, dict_env_parameters: dict, dict_learning_parameters: dict, dict_other_params: dict,
                 kind_of_alg: str, exploration_strategy: str):
        # TODO: predefined q-table
        self.rows = dict_env_parameters['rows']
        self.cols = dict_env_parameters['cols']
        self.n_actions = int(dict_env_parameters['n_actions'])
        self.action_space = Discrete(self.n_actions, start=0)
        """self.if_maze = dict_env_parameters['if_maze']
        self.reward_alive = dict_env_parameters['value_reward_alive']
        self.reward_winner = dict_env_parameters['value_reward_winner']
        self.reward_loser = dict_env_parameters['value_reward_loser']"""
        self.seed_value = dict_env_parameters['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

        self.n_episodes = dict_other_params['N_EPISODES']

        self.kind_of_alg = kind_of_alg
        if self.kind_of_alg not in global_variables.LIST_IMPLEMENTED_ALGORITHMS:
            raise AssertionError(f'{self.kind_of_alg} chosen not implemented')

        self.exploration_strategy = exploration_strategy
        if self.exploration_strategy in global_variables.LIST_IMPLEMENTED_EXPLORATION_STRATEGIES:
            if self.exploration_strategy == global_variables.LABEL_EPSILON_GREEDY:
                self.agent = exploration_strategies.EpsilonGreedyQAgent(dict_env_parameters, dict_learning_parameters,
                                                                        self.n_episodes)
            elif self.exploration_strategy == global_variables.LABEL_SOFTMAX_ANNEALING:
                self.agent = exploration_strategies.SoftmaxAnnealingQAgent(dict_env_parameters,
                                                                           dict_learning_parameters, self.n_episodes)
            elif self.exploration_strategy == global_variables.LABEL_THOMPSON_SAMPLING:
                self.agent = exploration_strategies.ThompsonSamplingQAgent(dict_env_parameters,
                                                                           dict_learning_parameters, self.n_episodes)
            elif self.exploration_strategy == global_variables.LABEL_BOLTZMANN_MACHINE:
                self.agent = exploration_strategies.BoltzmannQAgent(dict_env_parameters,
                                                                    dict_learning_parameters, self.n_episodes)
            else:
                raise AssertionError(f'{self.exploration_strategy} chosen not implemented')
        else:
            raise AssertionError(f'{self.exploration_strategy} chosen not implemented')

    def update_Q_or_memory(self, state, action, reward, next_state):
        self.agent.update_Q_or_memory(state, action, reward, next_state)

    def update_exp_fact(self, episode):
        if (self.exploration_strategy == global_variables.LABEL_EPSILON_GREEDY or
                self.exploration_strategy == global_variables.LABEL_SOFTMAX_ANNEALING):
            self.agent.update_exp_fact(episode)

    def select_action(self, current_state: np.ndarray, enemies_nearby_agent: np.ndarray = None,
                      goals_nearby_agent: np.ndarray = None, causal_table: pd.DataFrame = None) -> int:
        state = current_state.copy()

        if global_variables.LABEL_CAUSAL in self.kind_of_alg and causal_table is not None:

            possible_actions = self._get_possible_actions(causal_table, enemies_nearby_agent, goals_nearby_agent)

            action = self.agent.choose_action(state, possible_actions)

            if action not in possible_actions and action not in goals_nearby_agent:
                print(
                    f'enemies nearby: {enemies_nearby_agent} - possible actions: {possible_actions} - action chosen {action}')
                warnings.warn("Wrong causal model, enemies nearby", UserWarning)

            if action not in possible_actions and action in enemies_nearby_agent:
                print(
                    f'goals nearby {goals_nearby_agent} - possible actions: {possible_actions} - action chosen {action}')
                warnings.warn("Wrong causal model, goals nearby", UserWarning)
        else:
            possible_actions = list(np.arange(0, self.n_actions, 1))
            action = self.agent.choose_action(state, possible_actions)

        return action

    def _get_possible_actions(self, causal_table: pd.DataFrame,
                              enemies_nearby: np.ndarray = None, goals_nearby: np.ndarray = None) -> list:

        if enemies_nearby is not None:
            enemies_nearby = list(set(enemies_nearby))
        if goals_nearby is not None:
            goals_nearby = list(set(goals_nearby))

        col_action = next(s for s in causal_table.columns if global_variables.LABEL_COL_ACTION in s)
        col_reward = next(s for s in causal_table.columns if global_variables.LABEL_COL_REWARD in s)
        col_enemy_nearby = next(s for s in causal_table.columns if global_variables.LABEL_ENEMY_CAUSAL_TABLE in s and
                                global_variables.LABEL_NEARBY_CAUSAL_TABLE in s)
        col_goal_nearby = next(s for s in causal_table.columns if
                               global_variables.LABEL_GOAL_CAUSAL_TABLE in s and global_variables.LABEL_NEARBY_CAUSAL_TABLE in s)

        if enemies_nearby is not None and goals_nearby is not None:
            filtered_rows = causal_table[(causal_table[col_goal_nearby].isin(goals_nearby)) &
                                         (causal_table[col_enemy_nearby].isin(enemies_nearby))]
        elif enemies_nearby is not None:
            filtered_rows = causal_table[(causal_table[col_goal_nearby].isin([50])) &
                                         (causal_table[col_enemy_nearby].isin(enemies_nearby))]
        elif goals_nearby is not None:
            filtered_rows = causal_table[(causal_table[col_goal_nearby].isin(goals_nearby)) &
                                         (causal_table[col_enemy_nearby].isin([50]))]
        else:
            filtered_rows = causal_table

        max_achievable_reward = filtered_rows[col_reward].max()
        filtered_max_reward = filtered_rows[filtered_rows[col_reward] == max_achievable_reward]
        # Group by action and calculate average rewards
        grouped = filtered_max_reward.groupby([col_reward, col_enemy_nearby, col_goal_nearby])[col_action]
        # Initialize a variable to hold the common values
        possible_actions = None
        # Iterate over the groups
        for name, group in grouped:
            # If it's the first group, initialize common_values with the values of the first group
            if possible_actions is None:
                possible_actions = set(group)
            # Otherwise, take the intersection of common_values and the values of the current group
            else:
                possible_actions = possible_actions.intersection(group)
        if possible_actions is not None:
            possible_actions = list(possible_actions)
        else:
            possible_actions = []

        return possible_actions

    def return_q_table(self):
        if self.exploration_strategy != global_variables.LABEL_THOMPSON_SAMPLING:
            return self.agent.q_table
        else:
            return [self.agent.alpha, self.agent.beta]

import math
import random
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

        if global_variables.LABEL_CAUSAL in self.kind_of_alg:

            possible_actions = self._get_possible_actions(enemies_nearby_agent, goals_nearby_agent, causal_table)

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

    def _get_possible_actions(self, enemies_nearby: np.ndarray, goals_nearby: np.ndarray,
                              causal_table: pd.DataFrame) -> list:
        # TODO: RIFARE
        possible_actions = list(np.arange(0, self.n_actions, 1))

        possible_actions_goals = [element for element in possible_actions if element in goals_nearby]
        possible_actions_enemies = [element for element in possible_actions if element not in enemies_nearby]
        if len(possible_actions_goals) > 0:
            possible_actions = possible_actions_goals
        elif len(possible_actions_enemies) > 0:
            possible_actions = possible_actions_enemies

        """# Get column names containing action, reward, deltaX, and deltaY
        col_action = next(s for s in causal_table.columns if global_variables.LABEL_COL_ACTION in s)
        col_reward = next(s for s in causal_table.columns if global_variables.LABEL_COL_REWARD in s)
        # col_deltaX = next(s for s in causal_table.columns if global_variables.LABEL_COL_DELTAX in s)
        # col_deltaY = next(s for s in causal_table.columns if global_variables.LABEL_COL_DELTAY in s)

        # Goals model
        cond_reward_goals = causal_table[col_reward] == global_variables.VALUE_REWARD_WINNER_PAPER
        selected_rows_goals = causal_table[causal_table[col_action].isin(goals_nearby) & cond_reward_goals]
        selected_actions_goals = selected_rows_goals[col_action].unique()

        if len(selected_actions_goals) == 0:
            # Enemies model
            cond_reward_enemies = causal_table[col_reward] == global_variables.VALUE_REWARD_LOSER_PAPER
            selected_rows_enemies = causal_table[causal_table[col_action].isin(enemies_nearby) & cond_reward_enemies]
            selected_actions_enemies = selected_rows_enemies[col_action].unique()

            if len(selected_actions_enemies) > 0:
                possible_actions = [x for x in possible_actions if x not in selected_actions_enemies]
        else:
            possible_actions = selected_actions_goals"""
        # print(possible_actions)
        return possible_actions

    def return_q_table(self):
        if self.exploration_strategy != global_variables.LABEL_THOMPSON_SAMPLING:
            return self.agent.q_table
        else:
            return [self.agent.alpha, self.agent.beta]

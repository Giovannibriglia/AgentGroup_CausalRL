import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete
import random
import global_variables
from scripts.algorithms import exploration_strategies
from scripts.environment import CustomEnv


# TODO: the problem for the causality is on the deltaX and deltaY of the actions, check this aspect in the causal table
class QLearning:
    def __init__(self, dict_env_parameters: dict, dict_learning_parameters: dict, dict_other_params: dict,
                 kind_of_alg: str, exploration_strategy: str):
        # TODO: if_deep and predefined env

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
                self.agent = exploration_strategies.EpsilonGreedyQAgent(dict_env_parameters, dict_learning_parameters, self.n_episodes)
            elif self.exploration_strategy == global_variables.LABEL_SOFTMAX_ANNEALING:
                self.agent = exploration_strategies.SoftmaxAnnealingQAgent(dict_env_parameters, dict_learning_parameters, self.n_episodes)
            elif self.exploration_strategy == global_variables.LABEL_THOMPSON_SAMPLING:
                self.agent = exploration_strategies.ThompsonSamplingQAgent(dict_env_parameters, dict_learning_parameters, self.n_episodes)
            # TODO: implement others
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

    def select_action(self, current_state: np.ndarray, enemies_nearby_agent: np.ndarray = None, goals_nearby_agent: np.ndarray = None) -> int:
        state = current_state.copy()
        if global_variables.LABEL_CAUSAL in self.kind_of_alg:
            if self.kind_of_alg == global_variables.LABEL_CAUSAL_OFFLINE:
                causal_table = global_variables.CAUSAL_TABLE_OFFLINE
            else:
                # TODO: implement online casual table
                pass
            possible_actions = self._get_possible_actions(enemies_nearby_agent, goals_nearby_agent, causal_table)

            action = self.agent.choose_action(state, possible_actions)

            if action in enemies_nearby_agent:
                raise AssertionError(f'Wrong causal model, enemies nearby')
            if action not in goals_nearby_agent:
                raise AssertionError(f'Wrong causal model, goals nearby')
        else:
            action = self.agent.choose_action(state)
        return action

    def _get_possible_actions(self, enemies_nearby: np.ndarray, goals_nearby: np.ndarray,
                              causal_table: pd.DataFrame) -> list:

        possible_actions = list(np.arange(0, self.n_actions, 1))

        check_goal = False
        possible_actions_for_goal = []
        for nearby_goal in goals_nearby:
            action_to_do = causal_table[
                (causal_table[global_variables.COL_REWARD] == 1) & (
                        causal_table[global_variables.COL_NEARBY_GOAL] == nearby_goal)].reset_index(drop=True)
            if not action_to_do.empty:
                action_to_do = action_to_do.loc[0, global_variables.COL_ACTION]
                if action_to_do in possible_actions:
                    possible_actions_for_goal.append(action_to_do)
                    check_goal = True

        if not check_goal:
            for nearby_enemy in enemies_nearby:
                action_to_remove = causal_table[
                    (causal_table[global_variables.COL_REWARD] == -1) & (
                            causal_table[global_variables.COL_NEARBY_ENEMY] == nearby_enemy)].reset_index(
                    drop=True)
                if not action_to_remove.empty:
                    action_to_remove = action_to_remove.loc[0, global_variables.COL_ACTION]
                    if action_to_remove in possible_actions:
                        possible_actions.remove(action_to_remove)
            # print(f'Enemies nearby: {enemies_nearby} -- Possible actions: {possible_actions}')
        else:
            possible_actions = possible_actions_for_goal
            # print(f'Goals nearby: {goals_nearby} -- Possible actions: {possible_actions}')

        return possible_actions

import os
import time
import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete
import random
from tqdm.auto import tqdm
import global_variables
from scripts.algorithms.q_learning import QLearning
from scripts.algorithms.random_agent import RandomAgent
from scripts.environment import CustomEnv


class Training:
    def __init__(self, dict_env_parameters: dict, dict_learning_parameters: dict, dict_other_params: dict,
                 kind_of_alg: str,
                 exploration_strategy: str):

        self.dict_learning_parameters = dict_learning_parameters
        self.dict_env_parameters = dict_env_parameters
        self.dict_other_params = dict_other_params

        self.kind_of_alg = kind_of_alg
        self.exploration_strategy = exploration_strategy

        self.key_metric_rewards_for_episodes = global_variables.KEY_METRIC_REWARDS_EPISODE
        self.key_metric_steps_for_episodes = global_variables.KEY_METRICS_STEPS_EPISODE
        self.key_metric_average_time_for_episodes = global_variables.KEY_METRIC_TIME_EPISODE

        self.rows = dict_env_parameters['rows']
        self.cols = dict_env_parameters['cols']
        self.n_actions = int(dict_env_parameters['n_actions'])
        self.action_space = Discrete(self.n_actions, start=0)
        self.if_maze = dict_env_parameters['if_maze']
        self.reward_alive = dict_env_parameters['value_reward_alive']
        self.reward_winner = dict_env_parameters['value_reward_winner']
        self.reward_loser = dict_env_parameters['value_reward_loser']
        self.seed_value = dict_env_parameters['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

        self.who_moves_first = dict_other_params['WHO_MOVES_FIRST']
        self.timeout_in_hours = dict_other_params['TIMEOUT_IN_HOURS']
        self.kind_th_CI = dict_other_params['KIND_TH_CHECKS_CAUSAL_INFERENCE']
        self.th_CI = dict_other_params['TH_CHECKS_CAUSAL_INFERENCE']
        self.n_episodes = dict_other_params['N_EPISODES']

        if global_variables.LABEL_Q_LEARNING in kind_of_alg:
            self.algorithm = QLearning(dict_env_parameters, dict_learning_parameters, dict_other_params, kind_of_alg,
                                       exploration_strategy)
        elif global_variables.LABEL_RANDOM_AGENT in kind_of_alg:
            self.algorithm = RandomAgent(dict_env_parameters, dict_learning_parameters, dict_other_params, kind_of_alg,
                                         exploration_strategy)
        elif global_variables.LABEL_DQN in kind_of_alg:
            # TODO: DQN implementation
            pass

    def start_train(self, env, dir_save_metrics: str = None, name_sav_metrics: str = None, df_track: bool = False,
                    episodes_to_visualize: list = None, dir_save_video: str = None,
                    name_save_video: str = None):

        dict_metrics = {f'{self.key_metric_rewards_for_episodes}': [],
                        f'{self.key_metric_steps_for_episodes}': [],
                        f'{self.key_metric_average_time_for_episodes}': []}

        first_visit = True
        initial_time_game = time.time()
        if df_track:
            self.df_track = pd.DataFrame(columns=[])

        pbar = tqdm(range(self.n_episodes))
        for episode in pbar:
            # TODO: multi-agents settings
            if episode == 0:
                if first_visit:
                    current_states = env.reset(if_reset_n_time_loser=True)
                    first_visit = False
                else:
                    current_states = env.reset(if_reset_n_time_loser=False)
            else:
                current_states = env.reset(if_reset_n_time_loser=False)

            if episode in episodes_to_visualize:
                if_visualization = True
                env.init_gui(f'{self.kind_of_alg}_{self.exploration_strategy}', self.exploration_strategy,
                             self.n_episodes, global_variables.PATH_IMAGES_FOR_RENDER)
            else:
                if_visualization = False

            total_episode_reward = 0
            step_for_episode = 0
            done = False

            initial_time_episode = time.time()
            while not done:
                agent_n = 0
                actions = []

                if (time.time() - initial_time_game) <= self.timeout_in_hours * 3600:
                    if self.who_moves_first == 'enemy':
                        env.step_enemies()
                        if df_track:
                            enemies_nearby_agents, goals_nearby_agents = env.get_nearby_agent()
                        if global_variables.LABEL_CAUSAL in self.kind_of_alg:
                            enemies_nearby_agents, goals_nearby_agents = env.get_nearby_agent()
                            action = self.algorithm.select_action(current_states[agent_n],
                                                                  enemies_nearby_agents[agent_n],
                                                                  goals_nearby_agents[agent_n])
                        else:
                            action = self.algorithm.select_action(current_states[agent_n])
                        actions.append(action)

                    elif self.who_moves_first == 'agent':
                        if df_track:
                            enemies_nearby_agents, goals_nearby_agents = env.get_nearby_agent()
                        if global_variables.LABEL_CAUSAL in self.kind_of_alg:
                            enemies_nearby_agents, goals_nearby_agents = env.get_nearby_agent()
                            action = self.algorithm.select_action(current_states[agent_n],
                                                                  enemies_nearby_agents[agent_n],
                                                                  goals_nearby_agents[agent_n])
                        else:
                            action = self.algorithm.select_action(current_states[agent_n])
                        actions.append(action)
                        env.step_enemies()

                    current_states = current_states.copy()
                    # print('Current', current_states)
                    next_states = env.step_agents(actions)
                    # print('Next', next_states, 'Current', current_states)
                    if if_visualization:
                        env.movement_gui(episode, step_for_episode)

                    rewards, dones, if_loses = env.check_winner_gameover_agents()
                    if_lose = if_loses[agent_n]
                    done = dones[agent_n]
                    self.algorithm.update_Q_or_memory(current_states[agent_n], actions[agent_n], rewards[agent_n],
                                                      next_states[agent_n])
                    # TODO: add these for DQN
                    """
                    agent.optimize_model()
                    agent.update_target_net()"""

                    total_episode_reward += rewards[agent_n]
                    step_for_episode += 1

                    if df_track:
                        self._update_df(actions, current_states, next_states, rewards, enemies_nearby_agents,
                                        goals_nearby_agents)

                    if if_lose:
                        current_states = env.reset(if_reset_n_time_loser=False)
                    else:
                        current_states = next_states
                else:
                    pass
                    # TODO: implement here the timeout condition

            self.algorithm.update_exp_fact(episode)

            if if_visualization and name_save_video is not None:
                dir_save = f'{global_variables.GLOBAL_PATH_REPO}/Videos/{dir_save_video}'
                os.makedirs(dir_save, exist_ok=True)
                env.save_video(f'{dir_save}/{name_save_video}_episode{episode}.mp4')

            dict_metrics[f'{self.key_metric_rewards_for_episodes}'].append(total_episode_reward)
            dict_metrics[f'{self.key_metric_steps_for_episodes}'].append(step_for_episode)
            dict_metrics[f'{self.key_metric_average_time_for_episodes}'].append(
                round(time.time() - initial_time_episode, 3))

            mean = round(np.mean(dict_metrics[f'{self.key_metric_rewards_for_episodes}']), 2)
            pbar.set_postfix_str(
                f'{self.kind_of_alg} {self.exploration_strategy}, Average reward: {mean}, #Defeats: {env.n_times_loser}')

            # TODO: saving metrics

    def _update_df(self, actions, current_states, next_states, rewards, enemies_nearby_agents, goals_nearby_agents):
        n_agents = len(actions)
        n_enemies = len(enemies_nearby_agents[0])
        n_goals = len(goals_nearby_agents[0])
        keys = global_variables.define_columns_causal_table(n_agents, n_enemies, n_goals)
        dict_for_df = {key: None for key in keys}

        sub = next_states - current_states
        # print('C_U ', current_states, 'Next U ', next_states, '\n')
        for agent in range(n_agents):
            deltaY = sub[agent][0]
            deltaX = sub[agent][1]
            dict_for_df[
                f'{global_variables.LABEL_COL_DELTAX}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'] = deltaX
            dict_for_df[
                f'{global_variables.LABEL_COL_DELTAY}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'] = deltaY
            dict_for_df[f'{global_variables.LABEL_COL_REWARD}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'] = \
                rewards[agent]
            dict_for_df[f'{global_variables.LABEL_COL_ACTION}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'] = \
                actions[agent]
            for enemy in range(n_enemies):
                dict_for_df[
                    f'{global_variables.LABEL_ENEMY_CAUSAL_TABLE}{enemy}_Nearby_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'] = \
                    enemies_nearby_agents[agent][enemy]
            for goal in range(n_goals):
                dict_for_df[
                    f'{global_variables.LABEL_GOAL_CAUSAL_TABLE}{goal}_Nearby_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'] = \
                    goals_nearby_agents[agent][goal]

        self.df_track = self.df_track.append(dict_for_df, ignore_index=True)

    def get_df_track(self):
        return self.df_track


if __name__ == '__main__':
    dict_env_params = {'rows': 5, 'cols': 5, 'n_agents': 1, 'n_enemies': 2, 'n_goals': 1,
                       'n_actions': global_variables.N_ACTIONS_PAPER,
                       'if_maze': False,
                       'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                       'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                       'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                       'seed_value': 4, 'enemies_actions': 'random', 'env_type': 'numpy',
                       'predefined_env': None}
    dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
    dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

    # Create an environment
    env = CustomEnv(dict_env_params, False)

    # TODO: implement random agent and "get_table" method
    for label_kind_of_alg in [global_variables.LABEL_RANDOM_AGENT, global_variables.LABEL_Q_LEARNING]:

        if label_kind_of_alg == global_variables.LABEL_RANDOM_AGENT:
            label_exploration_strategy = 'random'
            class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                   f'{label_kind_of_alg}_{global_variables.LABEL_VANILLA}',
                                   f'{label_exploration_strategy}')
            # Train the agent
            class_train.start_train(env, [], [],
                                    False, [],
                                    'Comparison123'
                                    f'{label_kind_of_alg}_{global_variables.LABEL_VANILLA}_{label_exploration_strategy}')

            class_train.get_df_track().to_excel(f'{global_variables.GLOBAL_PATH_REPO}/mario.xlsx')
        else:
            for label_exploration_strategy in [global_variables.LABEL_SOFTMAX_ANNEALING,
                                               global_variables.LABEL_THOMPSON_SAMPLING,
                                               global_variables.LABEL_EPSILON_GREEDY]:
                class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                       f'{label_kind_of_alg}_{global_variables.LABEL_VANILLA}',
                                       f'{label_exploration_strategy}')
                # Train the agent
                class_train.start_train(env, [], [],
                                        True, [],
                                        'Comparison123'
                                        f'{label_kind_of_alg}_{global_variables.LABEL_VANILLA}_{label_exploration_strategy}')

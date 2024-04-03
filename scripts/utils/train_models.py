import json
import os
import random
import time
import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete
from tqdm.auto import tqdm
import global_variables
from scripts.algorithms.causal_discovery import CausalDiscovery
from scripts.algorithms.dqn_agent import DQNAgent
from scripts.algorithms.q_learning_agent import QLearningAgent
from scripts.algorithms.random_agent import RandomAgent
from scripts.utils.environment import CustomEnv
from scripts.utils.others import NumpyEncoder, compare_causal_graphs


with open(f'{global_variables.PATH_CAUSAL_GRAPH_OFFLINE}', 'r') as file:
    GROUND_TRUTH_CAUSAL_GRAPH = json.load(file)

OFFLINE_CAUSAL_TABLE = pd.read_pickle(f'{global_variables.PATH_CAUSAL_TABLE_OFFLINE}')


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
        self.key_metric_timeout_condition = global_variables.KEY_METRIC_TIMEOUT_CONDITION
        self.key_metric_q_table = global_variables.KEY_METRIC_Q_TABLE

        self.n_agents = dict_env_parameters['n_agents']
        self.n_enemies = dict_env_parameters['n_enemies']
        self.n_goals = dict_env_parameters['n_goals']
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

        if global_variables.LABEL_Q_LEARNING in self.kind_of_alg:
            self.algorithm = QLearningAgent(dict_env_parameters, dict_learning_parameters, dict_other_params,
                                            kind_of_alg,
                                            exploration_strategy)
        elif global_variables.LABEL_RANDOM_AGENT in self.kind_of_alg:
            self.algorithm = RandomAgent(dict_env_parameters, dict_learning_parameters, dict_other_params, kind_of_alg,
                                         exploration_strategy)
        elif global_variables.LABEL_DQN in self.kind_of_alg:
            self.algorithm = DQNAgent(dict_env_parameters, dict_learning_parameters, dict_other_params, kind_of_alg,
                                      exploration_strategy)
        else:
            raise AssertionError(f'{self.kind_of_alg} chosen not implemented')

    def start_train(self, env, dir_save_metrics: str = None, name_save_metrics: str = None,
                    batch_update_df_track: int = None,
                    episodes_to_visualize: list = None, dir_save_videos: str = None,
                    name_save_videos: str = None):

        if episodes_to_visualize is None:
            episodes_to_visualize = []

        if batch_update_df_track is not None:
            self.cols_df_track = global_variables.define_columns_causal_table(self.n_agents, self.n_enemies,
                                                                              self.n_goals)
            self.dict_df_track = {key: [] for key in self.cols_df_track}
            self.df_track = pd.DataFrame(columns=self.cols_df_track)

            if global_variables.LABEL_CAUSAL_ONLINE in self.kind_of_alg:
                self.CAUSAL_TABLE_ONLINE = None
                self.n_checks_causal_table_online = 0

        dict_metrics = {f'{self.key_metric_rewards_for_episodes}': [],
                        f'{self.key_metric_steps_for_episodes}': [],
                        f'{self.key_metric_average_time_for_episodes}': [],
                        f'{self.key_metric_timeout_condition}': False,
                        f'{self.key_metric_q_table}': None}

        first_visit = True
        initial_time_game = time.time()

        timeout_condition = (time.time() - initial_time_game) >= self.timeout_in_hours * 3600

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
                save_video = True if (name_save_videos is not None and dir_save_videos is not None) else False
                env.init_gui(f'{self.kind_of_alg}_{self.exploration_strategy}', self.exploration_strategy,
                             self.n_episodes, global_variables.PATH_IMAGES_FOR_RENDER, save_video)
            else:
                if_visualization = False

            total_episode_reward = 0
            step_for_episode = 0
            done = False

            initial_time_episode = time.time()
            while not done and not timeout_condition:
                agent_n = 0

                if not timeout_condition:
                    if self.who_moves_first == 'enemy':
                        env.step_enemies()

                        if batch_update_df_track is not None or global_variables.LABEL_CAUSAL in self.kind_of_alg:
                            enemies_nearby_agents, goals_nearby_agents = env.get_nearby_agent()
                        else:
                            enemies_nearby_agents, goals_nearby_agents = None, None

                        # current_states = current_states.copy()  # needed, pay attention when you remove it
                        actions = self._step_agents_inside_train(current_states, enemies_nearby_agents,
                                                                 goals_nearby_agents)

                        current_states = current_states.copy()  # needed, pay attention when you remove it
                        next_states = env.step_agents(actions)

                    elif self.who_moves_first == 'agent':
                        if batch_update_df_track is not None or global_variables.LABEL_CAUSAL in self.kind_of_alg:
                            enemies_nearby_agents, goals_nearby_agents = env.get_nearby_agent()
                        else:
                            enemies_nearby_agents, goals_nearby_agents = None, None

                        # current_states = current_states.copy()  # needed, pay attention when you remove it
                        actions = self._step_agents_inside_train(current_states, enemies_nearby_agents,
                                                                 goals_nearby_agents)

                        current_states = current_states.copy()  # needed, pay attention when you remove it
                        next_states = env.step_agents(actions)

                        env.step_enemies()

                    else:
                        raise AssertionError('parameter "who_moves_first" required is not already implemented')

                    if if_visualization:
                        env.movement_gui(episode, step_for_episode)

                    rewards, dones, if_loses = env.check_winner_gameover_agents()
                    if_lose = if_loses[agent_n]
                    done = dones[agent_n]

                    if global_variables.LABEL_DQN in self.kind_of_alg:
                        general_state = np.zeros(self.rows * self.cols)
                        general_state[current_states[agent_n][0] * self.rows + current_states[agent_n][1]] = 1
                        general_next_state = np.zeros(self.rows * self.cols)
                        general_next_state[next_states[agent_n][0] * self.rows + next_states[agent_n][1]] = 1
                        self.algorithm.update_Q_or_memory(general_state, actions[agent_n], rewards[agent_n],
                                                          general_next_state)
                    else:
                        self.algorithm.update_Q_or_memory(current_states[agent_n], actions[agent_n], rewards[agent_n],
                                                          next_states[agent_n])

                    total_episode_reward += rewards[agent_n]
                    step_for_episode += 1

                    if batch_update_df_track is not None:
                        self._update_dict_track(actions, current_states, next_states, rewards, enemies_nearby_agents,
                                                goals_nearby_agents)

                    if if_lose:
                        current_states = env.reset(if_reset_n_time_loser=False)
                    else:
                        current_states = next_states
                else:
                    print('***timeout condition***')
                    dict_metrics[f'{self.key_metric_timeout_condition}'] = True

            self.algorithm.update_exp_fact(episode)

            if batch_update_df_track is not None and (episode >= batch_update_df_track):
                self._define_df_track()
                self.dict_df_track = {key: [] for key in self.cols_df_track}

            if if_visualization and name_save_videos is not None and dir_save_videos is not None:
                dir_save = f'{global_variables.GLOBAL_PATH_REPO}/Videos/{dir_save_videos}'
                os.makedirs(dir_save, exist_ok=True)
                env.video_saving(f'{dir_save}/{name_save_videos}_episode{episode}.mp4')

            dict_metrics[f'{self.key_metric_rewards_for_episodes}'].append(total_episode_reward)
            dict_metrics[f'{self.key_metric_steps_for_episodes}'].append(step_for_episode)
            dict_metrics[f'{self.key_metric_average_time_for_episodes}'].append(
                round(time.time() - initial_time_episode, 3))

            mean = round(np.mean(dict_metrics[f'{self.key_metric_rewards_for_episodes}']), 2)
            pbar.set_postfix_str(
                f'{self.kind_of_alg} {self.exploration_strategy}, Average reward: {mean}, #Defeats: {env.n_times_loser}')

        # saving q_table if the agent is not DQN
        if global_variables.LABEL_Q_LEARNING in self.kind_of_alg:
            dict_metrics[f'{self.key_metric_q_table}'] = self.algorithm.return_q_table()
        else:
            dict_metrics[f'{self.key_metric_q_table}'] = None

        if name_save_metrics is not None and dir_save_metrics is not None:
            # os.makedirs(f'{global_variables.GLOBAL_PATH_REPO}/Results', exist_ok=True)
            dir_save = f'{global_variables.GLOBAL_PATH_REPO}/Results/{dir_save_metrics}'
            os.makedirs(dir_save, exist_ok=True)

            with open(f'{dir_save}/{name_save_metrics}.json', 'w') as f:
                json.dump(dict_metrics, f, cls=NumpyEncoder)

        if batch_update_df_track is not None:
            self._define_df_track()

            if global_variables.LABEL_CAUSAL_ONLINE in self.kind_of_alg:

                if self.n_checks_causal_table_online < self.th_CI:
                    cd = CausalDiscovery(self.df_track, self.n_agents, self.n_enemies, self.n_goals)

                    self.CAUSAL_TABLE_ONLINE = cd.return_causal_table()

                    out_causal_graph = cd.return_causal_graph()

                    if compare_causal_graphs(out_causal_graph, GROUND_TRUTH_CAUSAL_GRAPH):
                        self.n_checks_causal_table_online += 1
                    elif self.kind_th_CI == 'consecutive':
                        self.n_checks_causal_table_online = 0

    def _update_dict_track(self, actions, current_states, next_states, rewards, enemies_nearby_agents,
                           goals_nearby_agents):
        n_agents = len(actions)
        n_enemies = len(enemies_nearby_agents[0])
        n_goals = len(goals_nearby_agents[0])

        sub = next_states - current_states
        for agent in range(n_agents):
            deltaY = sub[agent][0]
            deltaX = sub[agent][1]
            self.dict_df_track[
                f'{global_variables.LABEL_COL_DELTAX}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'].append(
                deltaX)
            self.dict_df_track[
                f'{global_variables.LABEL_COL_DELTAY}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'].append(
                deltaY)
            self.dict_df_track[
                f'{global_variables.LABEL_COL_REWARD}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'].append(
                rewards[agent])
            self.dict_df_track[
                f'{global_variables.LABEL_COL_ACTION}_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'].append(
                actions[agent])
            for enemy in range(n_enemies):
                self.dict_df_track[
                    f'{global_variables.LABEL_ENEMY_CAUSAL_TABLE}{enemy}_Nearby_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'].append(
                    enemies_nearby_agents[agent][enemy])
            for goal in range(n_goals):
                self.dict_df_track[
                    f'{global_variables.LABEL_GOAL_CAUSAL_TABLE}{goal}_Nearby_{global_variables.LABEL_AGENT_CAUSAL_TABLE}{agent}'].append(
                    goals_nearby_agents[agent][goal])

    def _define_df_track(self):
        new_df_track = pd.DataFrame(self.dict_df_track)
        new_df_track = new_df_track.applymap(lambda x: int(global_variables.VALUE_ENTITY_FAR) if pd.isna(x) else x)
        self.df_track = pd.concat([self.df_track, new_df_track], ignore_index=True)

    def get_df_track(self):
        return self.df_track

    def _step_agents_inside_train(self, current_states: np.ndarray, enemies_nearby_agents: np.ndarray,
                                  goals_nearby_agents: np.ndarray) -> list:

        actions = []
        for agent_n in range(self.n_agents):
            if global_variables.LABEL_CAUSAL in self.kind_of_alg:
                if global_variables.LABEL_DQN in self.kind_of_alg:
                    general_state = np.zeros(self.rows * self.cols)
                    general_state[current_states[agent_n][0] * self.rows + current_states[agent_n][1]] = 1

                    action = self.algorithm.select_action(general_state,
                                                          enemies_nearby_agents[agent_n],
                                                          goals_nearby_agents[agent_n],
                                                          self.CAUSAL_TABLE_ONLINE if global_variables.LABEL_CAUSAL_ONLINE in self.kind_of_alg else OFFLINE_CAUSAL_TABLE)
                else:
                    action = self.algorithm.select_action(current_states[agent_n],
                                                          enemies_nearby_agents[agent_n],
                                                          goals_nearby_agents[agent_n],
                                                          self.CAUSAL_TABLE_ONLINE if global_variables.LABEL_CAUSAL_ONLINE in self.kind_of_alg else OFFLINE_CAUSAL_TABLE)

            else:
                if global_variables.LABEL_DQN in self.kind_of_alg:
                    general_state = np.zeros(self.rows * self.cols)
                    general_state[current_states[agent_n][0] * self.rows + current_states[agent_n][1]] = 1
                    action = self.algorithm.select_action(general_state)
                else:
                    action = self.algorithm.select_action(current_states[agent_n])

            actions.append(action)

        return actions


if __name__ == '__main__':
    if_maze = False
    rows = 5
    cols = 5
    n_enemies = 2
    for x in range(1):
        seed_value = global_variables.seed_values[x]
        dict_env_params = {'rows': rows, 'cols': cols, 'n_agents': 1, 'n_enemies': n_enemies, 'n_goals': 1,
                           'n_actions': global_variables.N_ACTIONS_PAPER,
                           'if_maze': False,
                           'value_reward_alive': global_variables.VALUE_REWARD_ALIVE_PAPER,
                           'value_reward_winner': global_variables.VALUE_REWARD_WINNER_PAPER,
                           'value_reward_loser': global_variables.VALUE_REWARD_LOSER_PAPER,
                           'seed_value': seed_value, 'enemies_actions': 'random', 'env_type': 'numpy',
                           'predefined_env': None}
        dict_learning_params = global_variables.DICT_LEARNING_PARAMETERS_PAPER
        dict_other_params = global_variables.DICT_OTHER_PARAMETERS_PAPER

        # Create an environment
        environment = CustomEnv(dict_env_params)

        for label_kind_of_alg in [global_variables.LABEL_Q_LEARNING, global_variables.LABEL_DQN]:

            for label_kind_of_alg2 in [global_variables.LABEL_CAUSAL_OFFLINE, global_variables.LABEL_VANILLA]:

                for label_exploration_strategy in [global_variables.LABEL_SOFTMAX_ANNEALING,
                                                   global_variables.LABEL_THOMPSON_SAMPLING,
                                                   global_variables.LABEL_BOLTZMANN_MACHINE,
                                                   global_variables.LABEL_EPSILON_GREEDY]:
                    class_train = Training(dict_env_params, dict_learning_params, dict_other_params,
                                           f'{label_kind_of_alg}_{label_kind_of_alg2}',
                                           f'{label_exploration_strategy}')

                    add_name_dir_save = 'Maze' if if_maze else 'Grid' + f'{rows}x{cols}_{n_enemies}' + 'enemies' if n_enemies > 1 else 'enemy'

                    class_train.start_train(environment,
                                            dir_save_metrics=f'Comparison/{add_name_dir_save}',
                                            name_save_metrics=f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_game{x}',
                                            episodes_to_visualize=[0,
                                                                   int(global_variables.N_TRAINING_EPISODES / 3),
                                                                   int(global_variables.N_TRAINING_EPISODES * 0.66),
                                                                   global_variables.N_TRAINING_EPISODES - 1])
                    # dir_save_videos=f'Comparison/{add_name_dir_save}',
                    # name_save_videos=f'{label_kind_of_alg}_{label_kind_of_alg2}_{label_exploration_strategy}_game{x}')

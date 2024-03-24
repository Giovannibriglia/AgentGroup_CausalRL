import time

import numpy as np
from gymnasium.spaces import Discrete
import random
from tqdm.auto import tqdm
from scripts.algorithms.exploration_strategies import EpsilonGreedyQAgent
from scripts.environment import CustomEnv

KEY_METRIC_REWARDS_EPISODE = 'rewards_for_episodes'
KEY_METRICS_STEPS_EPISODE = 'steps_for_episode'
KEY_METRIC_TIME_EPISODE = 'time_for_episode'

path_images_for_render = 'C:\\Users\giova\Documents\Research\CausalRL\images_for_render'


class QLearning:
    def __init__(self, dict_env_parameters: dict, dict_learning_parameters: dict, dict_other_params: dict,
                 kind_of_alg: str,
                 exploration_strategy: str, episodes_to_visualize: list):

        self.key_metric_rewards_for_episodes = KEY_METRIC_REWARDS_EPISODE
        self.key_metric_steps_for_episodes = KEY_METRICS_STEPS_EPISODE
        self.key_metric_average_time_for_episodes = KEY_METRIC_TIME_EPISODE

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
        self.episodes_to_visualize = episodes_to_visualize

        """self.gamma = dict_learning_parameters['GAMMA']
        self.lr = dict_learning_parameters['LEARNING_RATE']
        self.start_exp_proba = dict_learning_parameters['START_EXPLORATION_PROBABILITY']
        self.min_exp_proba = dict_learning_parameters['MIN_EXPLORATION_PROBABILITY']
        self.exp_game_percent = dict_learning_parameters['EXPLORATION_GAME_PERCENT']
        
        self.batch_size = dict_learning_parameters['BATCH_SIZE']
        self.tau = dict_learning_parameters['TAU']
        self.hidden_layers = dict_learning_parameters['HIDDEN_LAYERS']
        self.replay_memory_capacity = dict_learning_parameters['REPLAY_MEMORY_CAPACITY']"""

        self.who_moves_first = dict_other_params['WHO_MOVES_FIRST']
        self.timeout_in_hours = dict_other_params['TIMEOUT_IN_HOURS']
        self.kind_th_CI = dict_other_params['KIND_TH_CHECKS_CAUSAL_INFERENCE']
        self.th_CI = dict_other_params['TH_CHECKS_CAUSAL_INFERENCE']
        self.n_episodes = dict_other_params['N_EPISODES']

        self.exploration_strategy = exploration_strategy
        if self.exploration_strategy == 'EG':
            self.agent = EpsilonGreedyQAgent(dict_env_parameters, dict_learning_parameters, self.n_episodes)

        self.kind_of_alg = kind_of_alg

    def train(self, env):
        dict_metrics = {f'{self.key_metric_rewards_for_episodes}': [],
                        f'{self.key_metric_steps_for_episodes}': [],
                        f'{self.key_metric_average_time_for_episodes}': []}

        first_visit = True

        pbar = tqdm(range(self.n_episodes))
        for episode in pbar:
            # TODO: multi-agents settings
            if episode == 0:
                current_state = env.reset(if_reset_n_time_loser=True)
                initial_time_game = time.time()
                first_visit = False
            else:
                current_state = env.reset(if_reset_n_time_loser=False)

            if episode in self.episodes_to_visualize:
                if_visualization = True
                env.init_gui(f'QL_{self.kind_of_alg}', self.exploration_strategy, self.n_episodes, path_images_for_render)
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
                        if if_visualization:
                            env.movement_gui(episode, step_for_episode)
                        action = self.agent.choose_action(current_state[agent_n])
                        actions.append(action)
                    else:
                        # TODO: agent moves first
                        pass

                    next_state = env.step_agents(actions)
                    if if_visualization:
                        env.movement_gui(episode, step_for_episode)

                    rewards, dones, if_loses = env.check_winner_gameover_agents()
                    if_lose = if_loses[agent_n]
                    done = dones[agent_n]
                    current_state_ag = current_state[agent_n]
                    action_ag = actions[agent_n]
                    reward_ag = rewards[agent_n]
                    next_state_ag = next_state[agent_n]
                    self.agent.update_Q_or_memory(current_state_ag, action_ag, reward_ag, next_state_ag)

                    total_episode_reward += reward_ag
                    step_for_episode += 1

                    if if_lose:
                        current_state = env.reset(if_reset_n_time_loser=False)
                    else:
                        current_state = next_state
                else:
                    pass
                    # TODO: implement here the timeout condition

            dict_metrics[f'{self.key_metric_rewards_for_episodes}'].append(total_episode_reward)
            dict_metrics[f'{self.key_metric_steps_for_episodes}'].append(step_for_episode)
            dict_metrics[f'{self.key_metric_average_time_for_episodes}'].append(
                round(time.time() - initial_time_episode, 3))

            mean = round(np.mean(dict_metrics[f'{self.key_metric_rewards_for_episodes}']), 2)
            pbar.set_postfix_str(f'Average reward: {mean}, Number of defeats: {env.n_times_loser}')


if __name__ == '__main__':
    dict_env_params = {'rows': 5, 'cols': 5, 'n_agents': 1, 'n_enemies': 1, 'n_goals': 1, 'n_actions': 5,
                       'if_maze': False,
                       'value_reward_alive': 0, 'value_reward_winner': 1, 'value_reward_loser': -1, 'seed_value': 4,
                       'enemies_actions': 'random', 'env_type': 'numpy', 'predefined_env': None}

    dict_learning_params = {'GAMMA': 0.99, 'LEARNING_RATE': 0.0001,
                            'START_EXPLORATION_PROBABILITY': 1, 'MIN_EXPLORATION_PROBABILITY': 0.01,
                            'EXPLORATION_GAME_PERCENT': 0.6,
                            'BATCH_SIZE': 64, 'TAU': 0.005, 'HIDDEN_LAYERS': 128, 'REPLAY_MEMORY_CAPACITY': 10000
                            }

    dict_other_params = {'WHO_MOVES_FIRST': 'enemy',
                         'TIMEOUT_IN_HOURS': 4,
                         'KIND_TH_CHECKS_CAUSAL_INFERENCE': 'consecutive',
                         'TH_CHECKS_CAUSAL_INFERENCE': 3,
                         'N_EPISODES': 3000}

    # Create an environment
    env = CustomEnv(dict_env_params, False)

    # Initialize Q-learning agent
    agent = QLearning(dict_env_params, dict_learning_params, dict_other_params, 'vanilla', 'EG', [0, 1])

    # Train the agent
    agent.train(env)

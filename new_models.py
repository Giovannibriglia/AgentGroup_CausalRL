import random
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import warnings
import time
from scipy.stats import beta
warnings.filterwarnings("ignore")

GAMMA = 0.99
LEARNING_RATE = 0.0001
EXPLORATION_PROBA = 1
MIN_EXPLORATION_PROBA = 0.01
EXPLORATION_GAME_PERCENT = 0.6
BATCH_SIZE = 64
TAU = 0.005
HIDDEN_LAYERS = 128

col_action = 'Action_Agent0'
col_deltaX = 'DeltaX_Agent0'
col_deltaY = 'DeltaY_Agent0'
col_reward = 'Reward_Agent0'
col_nearby_enemy = 'Enemy0_Nearby_Agent0'
col_nearby_goal = 'Goal0_Nearby_Agent0'
causal_table = pd.read_pickle('heuristic_table.pkl')


def get_possible_actions(n_act_agents, enemies_nearby_all_agents, goals_nearby_all_agents):
    nearbies_goals = goals_nearby_all_agents[0]
    nearbies_enemies = enemies_nearby_all_agents[0]
    possible_actions = list(np.arange(0, n_act_agents, 1))

    check_goal = False
    possible_actions_for_goal = []
    for nearby_goal in nearbies_goals:
        action_to_do = causal_table[
            (causal_table[col_reward] == 1) & (causal_table[col_nearby_goal] == nearby_goal)].reset_index(drop=True)
        if not action_to_do.empty:
            action_to_do = action_to_do.loc[0, col_action]
            if action_to_do in possible_actions:
                possible_actions_for_goal.append(action_to_do)
                check_goal = True

    if not check_goal:
        for nearby_enemy in nearbies_enemies:
            action_to_remove = causal_table[
                (causal_table[col_reward] == -1) & (causal_table[col_nearby_enemy] == nearby_enemy)].reset_index(
                drop=True)
            if not action_to_remove.empty:
                action_to_remove = action_to_remove.loc[0, col_action]
                if action_to_remove in possible_actions:
                    possible_actions.remove(action_to_remove)
        # print(f'Enemies nearby: {nearbies_enemies} -- Possible actions: {possible_actions}')
    else:
        possible_actions = possible_actions_for_goal
        # print(f'Goals nearby: {nearbies_goals} -- Possible actions: {possible_actions}')

    return possible_actions


class SoftmaxAnnealingQAgent:
    def __init__(self, rows, cols, action_space_size, n_episodes, initial_temperature=1.0):
        self.rows = rows
        self.cols = cols
        self.action_space_size = action_space_size
        self.temperature = initial_temperature
        self.n_episodes = n_episodes

        # Q-table initialization
        self.q_table = np.zeros((self.rows, self.cols, action_space_size))

        self.EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (
                EXPLORATION_GAME_PERCENT * self.n_episodes)

    def softmax(self, values):
        exp_values = np.exp(values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def choose_action(self, state, possible_actions=None):
        stateX = int(state[0])
        stateY = int(state[1])

        if possible_actions is not None and len(possible_actions) > 0:
            action_values = self.q_table[stateX, stateY, possible_actions]
            action_probabilities = self.softmax(action_values)
            chosen_action = np.random.choice(possible_actions, p=action_probabilities)
        else:
            action_values = self.q_table[stateX, stateY, :]
            action_probabilities = self.softmax(action_values)
            chosen_action = np.random.choice(self.action_space_size, p=action_probabilities)

        return chosen_action

    def update_Q(self, state, action, reward, next_state):
        stateX = int(state[0])
        stateY = int(state[1])
        next_stateX = int(next_state[0])
        next_stateY = int(next_state[1])

        current_q_value = self.q_table[stateX, stateY, action]
        max_next_q_value = np.max(self.q_table[next_stateX, next_stateY, :])

        new_q_value = current_q_value + LEARNING_RATE * (reward + GAMMA * max_next_q_value - current_q_value)
        self.q_table[state, action] = new_q_value

    def update_exp_fact(self, e):  # annealing temperature
        self.temperature = max(MIN_EXPLORATION_PROBA, np.exp(-self.EXPLORATION_DECREASING_DECAY * e))


class BoltzmannQAgent:
    def __init__(self, rows, cols, action_space_size, temperature=1.0):
        self.rows = rows
        self.cols = cols
        self.action_space_size = action_space_size
        self.temperature = temperature

        # Q-table initialization
        self.q_table = np.zeros((self.rows, self.cols, action_space_size))

    def softmax(self, values):
        exp_values = np.exp(values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def choose_action(self, state, possible_actions=None):
        stateX = int(state[0])
        stateY = int(state[1])

        if possible_actions is not None and len(possible_actions) > 0:
            action_values = self.q_table[stateX, stateY, possible_actions]
            action_probabilities = self.softmax(action_values)
            chosen_action = np.random.choice(possible_actions, p=action_probabilities)
        else:
            action_values = self.q_table[stateX, stateY, :]
            action_probabilities = self.softmax(action_values)
            chosen_action = np.random.choice(self.action_space_size, p=action_probabilities)

        return chosen_action

    def update_Q(self, state, action, reward, next_state):
        stateX = int(state[0])
        stateY = int(state[1])
        next_stateX = int(next_state[0])
        next_stateY = int(next_state[1])
        current_q_value = self.q_table[stateX, stateY, action]
        max_next_q_value = np.max(self.q_table[next_stateX, next_stateY, :])

        new_q_value = current_q_value + LEARNING_RATE * (reward + GAMMA * max_next_q_value - current_q_value)
        self.q_table[stateX, stateY, action] = new_q_value


class ThompsonSamplingQAgent:
    def __init__(self, rows, cols, action_space_size, alpha=1, beta=1):
        self.rows = rows
        self.cols = cols
        self.action_space_size = action_space_size

        # Initialize Beta distribution parameters for each action
        self.alpha = np.maximum(np.zeros((self.rows, self.cols, action_space_size)) * alpha, 1)
        self.beta = np.maximum(np.zeros((self.rows, self.cols, action_space_size)) * beta, 1)

    def choose_action(self, state, possible_actions=None):
        # Sample from the Beta distribution for each action
        stateX = int(state[0])
        stateY = int(state[1])
        sampled_values = np.random.beta(self.alpha[stateX, stateY, :], self.beta[stateX, stateY, :])

        if possible_actions is not None and len(possible_actions) > 0:
            all_actions = list(np.arange(0, self.action_space_size, 1))
            dict_all_actions = {}
            for act in all_actions:
                dict_all_actions[act] = sampled_values[act]

            dict_valid_actions = {act: dict_all_actions[act] for act in possible_actions}
            chosen_action, _ = max(dict_valid_actions.items(), key=lambda x: x[1])
        else:
            chosen_action = np.argmax(sampled_values)

        return chosen_action

    def update_Q(self, state, action, reward, next_state=None):
        stateX = int(state[0])
        stateY = int(state[1])
        if reward == 1:
            self.alpha[stateX, stateY, action] += 1
        elif reward == -1:
            self.beta[stateX, stateY, action] += 1


class EpsilonGreedyAgent:

    def __init__(self, rows, cols, n_agent_actions, n_episodes):
        self.rows = rows
        self.cols = cols
        self.n_agent_actions = n_agent_actions
        self.Q_table = np.zeros((self.rows, self.cols, self.n_agent_actions))
        self.n_episodes = n_episodes
        self.exp_proba = EXPLORATION_PROBA
        self.MIN_EXPLORATION_PROBA = MIN_EXPLORATION_PROBA
        self.EXPLORATION_DECREASING_DECAY = -np.log(self.MIN_EXPLORATION_PROBA) / (
                EXPLORATION_GAME_PERCENT * self.n_episodes)

    def choose_action(self, state, possible_actions=None):
        if np.random.uniform(0, 1) < self.exp_proba:  # exploration
            if possible_actions is not None and len(possible_actions) > 0:
                action = random.sample(possible_actions, 1)[0]
            else:
                action = np.random.randint(0, self.n_agent_actions, size=1)
        else:  # exploitation
            stateX = int(state[0])
            stateY = int(state[1])
            if possible_actions is not None and len(possible_actions) > 0:
                all_actions = list(np.arange(0, self.n_agent_actions, 1))
                dict_all_actions = {}
                for act in all_actions:
                    dict_all_actions[act] = self.Q_table[stateX, stateY, act]

                dict_valid_actions = {act: dict_all_actions[act] for act in possible_actions}
                chosen_action, _ = max(dict_valid_actions.items(), key=lambda x: x[1])
            else:
                action = np.argmax(self.Q_table[stateX, stateY, :])

        return action

    def update_Q(self, state, action, reward, next_state):
        stateX = int(state[0])
        stateY = int(state[1])
        next_stateX = int(next_state[0])
        next_stateY = int(next_state[1])
        current_q_value = self.Q_table[stateX, stateY, action]
        max_next_q_value = np.max(self.Q_table[next_stateX, next_stateY, :])

        new_q_value = current_q_value + LEARNING_RATE * (reward + GAMMA * max_next_q_value - current_q_value)
        self.Q_table[stateX, stateY, action] = new_q_value

    def update_exp_fact(self, episode):  # update exploration probability
        self.exp_proba = max(self.MIN_EXPLORATION_PROBA, np.exp(-self.EXPLORATION_DECREASING_DECAY * episode))


def QL_variations(env, n_act_agents, n_episodes, alg, who_moves_first):
    rows = env.rows
    cols = env.cols
    action_space_size = n_act_agents

    if 'SoftmaxAnnealing' in alg:
        agent_alg = SoftmaxAnnealingQAgent(rows, cols, action_space_size, n_episodes)
    elif 'EpsilonGreedy' in alg:
        agent_alg = EpsilonGreedyAgent(rows, cols, action_space_size, n_episodes)
    elif 'BoltzmannMachine' in alg:
        agent_alg = BoltzmannQAgent(rows, cols, action_space_size, n_episodes)
    elif 'ThompsonSampling' in alg:
        agent_alg = ThompsonSamplingQAgent(rows, cols, action_space_size, n_episodes)

    average_episodes_rewards = []
    steps_for_episode = []

    pbar = tqdm(range(n_episodes))
    time.sleep(1)
    for e in pbar:
        agent = 0
        if e == 0:
            env.reset(reset_n_times_loser=True)
        else:
            env.reset(reset_n_times_loser=False)

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        while not done:

            current_state = env.pos_agents[-1][agent]

            if 'Causal' in alg:
                enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                        goals_nearby_all_agents)
            else:
                possible_actions = None

            if who_moves_first == 'Enemy':
                env.step_enemies()
                _, _, if_lose = env.check_winner_gameover_agent(current_state[0], current_state[1])
                if_lose = False
                if not if_lose:
                    action = agent_alg.choose_action(current_state, possible_actions)
                    next_state = env.step_agent(action)[0]
                else:
                    next_state = current_state
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
            elif who_moves_first == 'Agent':
                action = agent_alg.choose_action(current_state, possible_actions)
                next_state = env.step_agent(action)[0]
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
                _, dones, _ = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
                if not dones[agent]:
                    env.step_enemies()

            rewards, dones, if_lose = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
            reward = int(rewards[agent])
            done = dones[agent]  # If agent wins, end loop and restart
            if_lose = if_lose

            """if possible_actions is not None and len(possible_actions) > 0 and if_lose:
                print(possible_actions, action, enemies_nearby_all_agents)
                print(f'lose: wrong causal gameover model in {alg}')
                print(f'agents: {env.pos_agents[-1]}')
                print(f'enemies: {env.pos_enemies[-1]}')"""

            # Update the Q-table/values
            agent_alg.update_Q(current_state, action, reward, next_state)

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)

            step_for_episode += 1

        if alg == 'QL_SoftmaxAnnealing' or alg == 'EpsilonGreedyAgent':
            agent_alg.update_exp_fact(e)

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode

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


def causal_model_movement(causal_table, action, oldX, oldY, rows, cols):
    row_action = causal_table[causal_table[col_action] == action].reset_index(drop=True)

    deltaX = row_action.loc[0, col_deltaX]
    deltaY = row_action.loc[0, col_deltaY]

    newX = oldX
    newY = oldY

    if 0 <= oldX + deltaX <= cols - 1:
        newX += deltaX

    if 0 <= oldY + deltaY <= rows - 1:
        newY += deltaY

    # print(f'\nOldX: {oldX} , action {action} --> NewX {newX}')
    # print(f'OldY: {oldY} , action {action} --> NewX {newY}')
    return newX, newY


def causal_model_winner_gameover(causal_table, possible_actions_in, nearbies_enemies, nearbies_goals):
    possible_actions = possible_actions_in

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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args, Transition):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, HIDDEN_LAYERS)
        self.layer2 = nn.Linear(HIDDEN_LAYERS, HIDDEN_LAYERS)
        self.final_layer = nn.Linear(HIDDEN_LAYERS, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.final_layer(x)


class BoltzmannQAgent:
    def __init__(self, rows, cols, action_space_size, temperature=1.0):
        self.rows = rows
        self.cols = cols
        self.action_space_size = action_space_size
        self.temperature = temperature

        # Q-table initialization
        self.q_table = np.zeros((rows, cols, action_space_size))

    def softmax(self, values):
        exp_values = np.exp(values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def choose_action(self, state):
        stateX = int(state[0])
        stateY = int(state[1])
        action_values = self.q_table[stateX, stateY, :]
        action_probabilities = self.softmax(action_values)

        # Choose action based on probabilities
        chosen_action = np.random.choice(self.action_space_size, p=action_probabilities)
        return chosen_action

    def update_q_table(self, state, action, reward, next_state):
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
        self.alpha = np.maximum(np.zeros((rows, cols, action_space_size)) * alpha, 1)
        self.beta = np.maximum(np.zeros((rows, cols, action_space_size)) * beta, 1)

    def choose_action(self, state):
        # Sample from the Beta distribution for each action
        stateX = int(state[0])
        stateY = int(state[1])
        sampled_values = np.random.beta(self.alpha[stateX, stateY, :], self.beta[stateX, stateY, :])

        # Choose the action with the highest sampled value
        chosen_action = np.argmax(sampled_values)
        return chosen_action

    def update_q_values(self, state, action, reward):

        stateX = int(state[0])
        stateY = int(state[1])
        if reward == 1:
            self.alpha[stateX, stateY, action] += 1
        elif reward == -1:
            self.beta[stateX, stateY, action] += 1


def QL_EpsGreedy(env, n_act_agents, n_episodes):
    global EXPLORATION_PROBA
    rows = env.rows
    cols = env.cols
    Q_table = np.zeros((rows, cols, n_act_agents))

    EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (EXPLORATION_GAME_PERCENT * n_episodes)

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

            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            # print('current acquired', [current_stateX, current_stateY])
            step_for_episode += 1
            _, _, _ = env.step_enemies()

            " epsilon-greedy "
            if np.random.uniform(0, 1) < EXPLORATION_PROBA:  # exploration
                action = env.action_space.sample()
            else:  # exploitation
                action = np.argmax(Q_table[current_stateX, current_stateY, :])

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][1])
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                Q_table[next_stateX, next_stateY, :]))

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_stateX = current_state[agent][0]
                current_stateY = current_state[agent][1]

            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        EXPLORATION_PROBA = max(MIN_EXPLORATION_PROBA, np.exp(-EXPLORATION_DECREASING_DECAY * e))

        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def QL_BoltzmannMachine(env, n_act_agents, n_episodes):
    rows = env.rows
    cols = env.cols
    action_space_size = n_act_agents

    # Initialize the Boltzmann Q-learning agent
    boltzmann_q_agent = BoltzmannQAgent(rows, cols, action_space_size)

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
            step_for_episode += 1

            current_state = env.pos_agents[-1][agent]

            _, _, _ = env.step_enemies()

            # Choose action using Boltzmann exploration
            action = boltzmann_q_agent.choose_action(current_state)

            result = env.step_agent(action)
            # print('result:', result)
            next_state = result[0][agent]
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            # Update the Q-table
            boltzmann_q_agent.update_q_table(current_state, action, reward, next_state)

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def QL_ThompsonSampling(env, n_act_agents, n_episodes):
    rows = env.rows
    cols = env.cols
    action_space_size = n_act_agents

    thompson_q_agent = ThompsonSamplingQAgent(rows, cols, action_space_size)

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
            step_for_episode += 1

            current_state = env.pos_agents[-1][agent]

            _, _, _ = env.step_enemies()

            # Choose action using Boltzmann exploration
            action = thompson_q_agent.choose_action(current_state)

            result = env.step_agent(action)
            # print('result:', result)
            next_state = result[0][agent]
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            # Update the Q-table
            thompson_q_agent.update_q_values(current_state, action, reward)

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def CQL3(env, n_act_agents, n_episodes, causal_table):
    global EXPLORATION_PROBA
    EXPLORATION_ACTIONS_TH = 10

    rows = env.rows
    cols = env.cols
    # initialize the Q-Table
    Q_table = np.zeros((rows, cols, n_act_agents))  # x, y, actions
    # initialize table to keep track of the explored states
    Q_table_track = np.zeros((rows, cols))  # x, y, actions

    EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (EXPLORATION_GAME_PERCENT * n_episodes)

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
            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            # print('current acquired', [current_stateX, current_stateY])
            step_for_episode += 1
            enemies_nearby_all_agents, goals_nearby_all_agents, new_en_coord = env.step_enemies()
            " epsilon-greedy "
            if np.random.uniform(0, 1) < EXPLORATION_PROBA:  # exploration
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                goals_nearby_agent = goals_nearby_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_winner_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                                goals_nearby_agent)

                new = False
                check_tries = 0
                if len(possible_actions) > 0:
                    while not new:
                        action = random.sample(possible_actions, 1)[0]
                        next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY,
                                                                         current_stateX, rows, cols)

                        if Q_table_track[next_stateX, next_stateY] == 0:
                            new = True
                            reward = 0  # reward to have found a new state
                            Q_table_track[next_stateX, next_stateY] = 1
                        else:
                            check_tries += 1
                            if check_tries == EXPLORATION_ACTIONS_TH:
                                new = True
                                reward = 0  # reward to have not found a new state
                                action = random.sample(possible_actions, 1)[0]
                else:
                    action = env.action_space.sample()

                next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX,
                                                                 rows, cols)

                # additional Q-Table update
                Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                    current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                    Q_table[next_stateX, next_stateY, :]))

            else:  # exploitation
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                goals_nearby_agent = goals_nearby_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_winner_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                                goals_nearby_agent)

                if len(possible_actions) > 0:
                    max_value = max(Q_table[current_stateX, current_stateY, possible_actions])

                    actions_wrong = []
                    for act_test in [s for s in range(n_act_agents)]:
                        if act_test not in possible_actions:
                            actions_wrong.append(act_test)
                    actions_wrong = list(set(actions_wrong))
                    q_table_values = list(
                        np.array(np.where(Q_table[current_stateX, current_stateY, :] == max_value)[0]))

                    possibilities = []
                    for act_test in q_table_values:
                        if act_test not in actions_wrong:
                            possibilities.append(act_test)
                    possibilities = list(set(possibilities))
                    action = random.sample(possibilities, k=1)[0]

            if action not in possible_actions and len(possible_actions) > 0:
                print(f'PROBLEM) Action: {action} - Nearbies: {enemies_nearby_agent} - Pos Act: {possible_actions}')
            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][1])
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                Q_table[next_stateX, next_stateY, :]))
            total_episode_reward = total_episode_reward + reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_stateX = current_state[agent][0]
                current_stateY = current_state[agent][1]

            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        EXPLORATION_PROBA = max(MIN_EXPLORATION_PROBA, np.exp(-EXPLORATION_DECREASING_DECAY * e))

        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def CQL4(env, n_act_agents, n_episodes, causal_table):
    global EXPLORATION_PROBA
    EXPLORATION_ACTIONS_TH = 10

    rows = env.rows
    cols = env.cols
    # initialize the Q-Table
    Q_table = np.zeros((rows, cols, n_act_agents))  # x, y, actions
    # initialize table to keep track of the explored states
    Q_table_track = np.zeros((rows, cols))  # x, y, actions

    EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (EXPLORATION_GAME_PERCENT * n_episodes)

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
            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            # print('current acquired', [current_stateX, current_stateY])
            step_for_episode += 1
            enemies_nearby_all_agents, goals_nearby_all_agents, new_en_coord = env.step_enemies()
            " epsilon-greedy "
            if np.random.uniform(0, 1) < EXPLORATION_PROBA:  # exploration
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                goals_nearby_agent = goals_nearby_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_winner_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                                goals_nearby_agent)

                if len(possible_actions) > 0:
                    action = random.sample(possible_actions, 1)[0]
                else:
                    action = env.action_space.sample()

                next_stateX, next_stateY = causal_model_movement(causal_table, action, current_stateY, current_stateX,
                                                                 rows, cols)

                # additional Q-Table update
                reward = 0
                Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                    current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                    Q_table[next_stateX, next_stateY, :]))

            else:  # exploitation
                enemies_nearby_agent = enemies_nearby_all_agents[agent]
                goals_nearby_agent = goals_nearby_all_agents[agent]
                possible_actions = [s for s in range(n_act_agents)]

                possible_actions = causal_model_winner_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                                goals_nearby_agent)

                if len(possible_actions) > 0:
                    max_value = max(Q_table[current_stateX, current_stateY, possible_actions])

                    actions_wrong = []
                    for act_test in [s for s in range(n_act_agents)]:
                        if act_test not in possible_actions:
                            actions_wrong.append(act_test)
                    actions_wrong = list(set(actions_wrong))
                    q_table_values = list(
                        np.array(np.where(Q_table[current_stateX, current_stateY, :] == max_value)[0]))

                    possibilities = []
                    for act_test in q_table_values:
                        if act_test not in actions_wrong:
                            possibilities.append(act_test)
                    possibilities = list(set(possibilities))
                    action = random.sample(possibilities, k=1)[0]

            if action not in possible_actions and len(possible_actions) > 0:
                print(f'PROBLEM) Action: {action} - Nearbies: {enemies_nearby_agent} - Pos Act: {possible_actions}')
            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][1])
            reward = int(result[1][agent])
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            Q_table[current_stateX, current_stateY, action] = (1 - LEARNING_RATE) * Q_table[
                current_stateX, current_stateY, action] + LEARNING_RATE * (reward + GAMMA * max(
                Q_table[next_stateX, next_stateY, :]))
            total_episode_reward = total_episode_reward + reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_stateX = current_state[agent][0]
                current_stateY = current_state[agent][1]

            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        EXPLORATION_PROBA = max(MIN_EXPLORATION_PROBA, np.exp(-EXPLORATION_DECREASING_DECAY * e))

        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def DeepQNetwork(env, n_act_agents, n_episodes):
    global EXPLORATION_PROBA
    EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (EXPLORATION_GAME_PERCENT * n_episodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # print('Device: ', torch.cuda.get_device_name(0))
        pass

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    policy_net: DQN = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(10000)

    def select_action(state, exp_proba):

        if np.random.uniform(0, 1) < exp_proba:  # exploration
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                actions_values = policy_net(state)
                return actions_values.max(1).indices.view(1, 1)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
        # selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in
        # case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

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
            step_for_episode += 1
            _, _, new_en_coord = env.step_enemies()

            general_state = np.zeros(env.rows * env.cols)

            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            general_state[current_stateY * env.cols + current_stateX] = 1
            general_state = torch.tensor(general_state, dtype=torch.float32, device=device).unsqueeze(0)

            action = select_action(general_state, EXPLORATION_PROBA)

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][1])
            reward = int(result[1][agent])
            total_episode_reward += reward
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            reward = torch.tensor([reward], device=device)
            next_state = np.zeros(env.rows * env.cols)
            next_state[next_stateY * env.cols + next_stateX] = 1
            next_state = torch.tensor(next_state, dtype=torch.float32,
                                      device=device).unsqueeze(0)

            memory.push(general_state, action, next_state, reward, Transition=Transition)

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if if_lose:
                general_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and env.n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        EXPLORATION_PROBA = max(MIN_EXPLORATION_PROBA, np.exp(-EXPLORATION_DECREASING_DECAY * e))

        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def CausalDeepQNetwork(env, n_act_agents, n_episodes, causal_table):
    global EXPLORATION_PROBA

    def select_action(state, exp_proba, possible_actions):

        if np.random.uniform(0, 1) < exp_proba:  # exploration
            if len(possible_actions) > 0:
                return torch.tensor([[possible_actions[torch.randint(0, len(possible_actions), (1,)).item()]]],
                                    device=device, dtype=torch.long)
            else:
                return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:  # exploitation

            actions_to_avoid = [s for s in range(n_act_agents) if s not in possible_actions]

            actions_values = policy_net(state)
            for act_to_avoid in actions_to_avoid:
                actions_values[:, act_to_avoid] = - 10000

            return actions_values.max(1).indices.view(1, 1)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
        # selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in
        # case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (EXPLORATION_GAME_PERCENT * n_episodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # print('Device: ', torch.cuda.get_device_name(0))
        pass

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    policy_net: DQN = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(10000)

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
            step_for_episode += 1
            enemies_nearby_all_agents, goals_nearby_all_agents, new_en_coord = env.step_enemies()

            goals_nearby_agent = goals_nearby_all_agents[agent]
            enemies_nearby_agent = enemies_nearby_all_agents[0]  # only one agent

            possible_actions = [s for s in range(n_act_agents)]
            possible_actions = causal_model_winner_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                            goals_nearby_agent)
            possible_actions = torch.tensor(possible_actions, device=device)

            general_state = np.zeros(env.rows * env.cols)

            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            general_state[current_stateY * env.cols + current_stateX] = 1
            general_state = torch.tensor(general_state, dtype=torch.float32,
                                         device=device).unsqueeze(0)

            action = select_action(general_state, EXPLORATION_PROBA, possible_actions)

            if action not in possible_actions and len(possible_actions) > 0:
                print(f'PROBLEM) Action: {action} - Nearbies: {enemies_nearby_agent} - Pos Act: {possible_actions}')

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][1])
            reward = int(result[1][agent])
            total_episode_reward += reward
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            reward = torch.tensor([reward], device=device)
            next_state = np.zeros(env.rows * env.cols)
            next_state[next_stateY * env.cols + next_stateX] = 1
            next_state = torch.tensor(next_state, dtype=torch.float32,
                                      device=device).unsqueeze(0)

            memory.push(general_state, action, next_state, reward, Transition=Transition)

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if if_lose:
                general_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and env.n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        EXPLORATION_PROBA = max(MIN_EXPLORATION_PROBA, np.exp(-EXPLORATION_DECREASING_DECAY * e))

        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def DeepQNetwork_Mod(env, n_act_agents, n_episodes):
    global EXPLORATION_PROBA
    EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (EXPLORATION_GAME_PERCENT * n_episodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # print('Device: ', torch.cuda.get_device_name(0))
        pass

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    policy_net: DQN = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(10000)

    def select_action(state, exp_proba):

        if np.random.uniform(0, 1) < exp_proba:  # exploration
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                actions_values = policy_net(state)
                return actions_values.max(1).indices.view(1, 1)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
        # selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in
        # case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

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
            step_for_episode += 1
            _, _, new_en_coord = env.step_enemies()

            general_state = np.zeros(env.rows * env.cols)

            for coord_en in new_en_coord:
                general_state[coord_en[1] * env.cols + coord_en[0]] = -1

            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            general_state[current_stateY * env.cols + current_stateX] = 1
            general_state = torch.tensor(general_state, dtype=torch.float32,
                                         device=device).unsqueeze(0)

            action = select_action(general_state, EXPLORATION_PROBA)

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][1])
            reward = int(result[1][agent])
            total_episode_reward += reward
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            reward = torch.tensor([reward], device=device)
            next_state = np.zeros(env.rows * env.cols)
            next_state[next_stateY * env.cols + next_stateX] = 1
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(general_state, action, next_state, reward, Transition=Transition)

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if if_lose:
                general_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and env.n_act_agents < 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        EXPLORATION_PROBA = max(MIN_EXPLORATION_PROBA, np.exp(-EXPLORATION_DECREASING_DECAY * e))

        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


def CausalDeepQNetwork_Mod(env, n_act_agents, n_episodes, causal_table):
    global EXPLORATION_PROBA

    def select_action(state, exp_proba, possible_actions):

        if np.random.uniform(0, 1) < exp_proba:  # exploration
            if len(possible_actions) > 0:
                return torch.tensor([[possible_actions[torch.randint(0, len(possible_actions), (1,)).item()]]],
                                    device=device, dtype=torch.long)
            else:
                return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        else:  # exploitation

            actions_to_avoid = [s for s in range(n_act_agents) if s not in possible_actions]

            actions_values = policy_net(state)
            for act_to_avoid in actions_to_avoid:
                actions_values[:, act_to_avoid] = - 10000

            return actions_values.max(1).indices.view(1, 1)

            """with torch.no_grad():
                copy_weights = policy_net.final_layer.weight[:, not_possible_actions].clone()
                copy_biases = policy_net.final_layer.bias[not_possible_actions].clone()
                policy_net.final_layer.weight[:, not_possible_actions] = 0
                policy_net.final_layer.bias[not_possible_actions] = -1000

                actions_values = policy_net(state)

                policy_net.final_layer.weight[:, not_possible_actions] = copy_weights
                policy_net.final_layer.bias[not_possible_actions] = copy_biases

                return actions_values.max(1).indices.view(1, 1)"""

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
        # selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in
        # case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    EXPLORATION_DECREASING_DECAY = -np.log(MIN_EXPLORATION_PROBA) / (EXPLORATION_GAME_PERCENT * n_episodes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        # print('Device: ', torch.cuda.get_device_name(0))
        pass

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    policy_net: DQN = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net = DQN(env.cols * env.rows, n_act_agents).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
    memory = ReplayMemory(10000)

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
            step_for_episode += 1
            enemies_nearby_all_agents, goals_nearby_all_agents, new_en_coord = env.step_enemies()

            goals_nearby_agent = goals_nearby_all_agents[agent]
            enemies_nearby_agent = enemies_nearby_all_agents[agent]  # only one agent

            possible_actions = [s for s in range(n_act_agents)]
            possible_actions = causal_model_winner_gameover(causal_table, possible_actions, enemies_nearby_agent,
                                                            goals_nearby_agent)
            possible_actions = torch.tensor(possible_actions, device=device)

            general_state = np.zeros(env.rows * env.cols)

            for coord_en in new_en_coord:
                general_state[coord_en[1] * env.cols + coord_en[0]] = -1

            current_stateX = env.pos_agents[-1][agent][0]
            current_stateY = env.pos_agents[-1][agent][1]

            general_state[current_stateY * env.cols + current_stateX] = 1
            general_state = torch.tensor(general_state, dtype=torch.float32,
                                         device=device).unsqueeze(0)

            action = select_action(general_state, EXPLORATION_PROBA, possible_actions)

            if action not in possible_actions and len(possible_actions) > 0:
                print(f'PROBLEM) Action: {action} - Nearbies: {enemies_nearby_agent} - Pos Act: {possible_actions}')

            result = env.step_agent(action)
            # print('result:', result)
            next_stateX = int(result[0][agent][0])
            next_stateY = int(result[0][agent][1])
            reward = int(result[1][agent])
            total_episode_reward += reward
            done = result[2][agent]  # If agent wins, end loop and restart
            if_lose = result[3]

            reward = torch.tensor([reward], device=device)
            next_state = np.zeros(env.rows * env.cols)
            next_state[next_stateY * env.cols + next_stateX] = 1
            next_state = torch.tensor(next_state, dtype=torch.float32,
                                      device=device).unsqueeze(0)

            memory.push(general_state, action, next_state, reward, Transition=Transition)

            optimize_model()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_net.load_state_dict(target_net_state_dict)

            if if_lose:
                general_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
            if (abs(current_stateX - next_stateX) + abs(current_stateY - next_stateY)) > 1 and env.n_act_agents <= 5:
                print('movement control problem:', [current_stateX, current_stateY], [next_stateX, next_stateY])

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        # updating the exploration proba using exponential decay formula
        EXPLORATION_PROBA = max(MIN_EXPLORATION_PROBA, np.exp(-EXPLORATION_DECREASING_DECAY * e))

        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode

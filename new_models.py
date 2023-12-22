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

        self.lr = LEARNING_RATE
        self.gamma = GAMMA

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

        new_q_value = current_q_value + self.lr * (reward + self.gamma * max_next_q_value - current_q_value)
        self.q_table[stateX, stateY, action] = new_q_value

    def update_exp_fact(self, e):  # annealing temperature
        self.temperature = max(MIN_EXPLORATION_PROBA, np.exp(-self.EXPLORATION_DECREASING_DECAY * e))


class BoltzmannQAgent:
    def __init__(self, rows, cols, action_space_size, temperature=1.0):
        self.rows = rows
        self.cols = cols
        self.action_space_size = action_space_size
        self.temperature = temperature

        self.lr = LEARNING_RATE
        self.gamma = GAMMA

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

        new_q_value = current_q_value + self.lr * (reward + self.gamma * max_next_q_value - current_q_value)
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


class EpsilonGreedyQAgent:

    def __init__(self, rows, cols, n_agent_actions, n_episodes):
        self.rows = rows
        self.cols = cols
        self.n_agent_actions = n_agent_actions
        self.n_episodes = n_episodes

        self.Q_table = np.zeros((self.rows, self.cols, self.n_agent_actions))

        self.lr = LEARNING_RATE
        self.gamma = GAMMA

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
                action, _ = max(dict_valid_actions.items(), key=lambda x: x[1])
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

        new_q_value = current_q_value + self.lr * (reward + self.gamma * max_next_q_value - current_q_value)
        self.Q_table[stateX, stateY, action] = new_q_value

    def update_exp_fact(self, episode):  # update exploration probability
        self.exp_proba = max(self.MIN_EXPLORATION_PROBA, np.exp(-self.EXPLORATION_DECREASING_DECAY * episode))


def QL_variations(env, n_act_agents, n_episodes, alg, who_moves_first):
    rows = env.rows
    cols = env.cols
    action_space_size = n_act_agents

    if 'SA' in alg:
        agent = SoftmaxAnnealingQAgent(rows, cols, action_space_size, n_episodes)
    elif 'EG' in alg:
        agent = EpsilonGreedyQAgent(rows, cols, action_space_size, n_episodes)
    elif 'BM' in alg:
        agent = BoltzmannQAgent(rows, cols, action_space_size, n_episodes)
    elif 'TS' in alg:
        agent = ThompsonSamplingQAgent(rows, cols, action_space_size, n_episodes)

    average_episodes_rewards = []
    steps_for_episode = []

    pbar = tqdm(range(n_episodes))
    time.sleep(1)
    for e in pbar:
        agent_n = 0
        if e == 0:
            current_state, _, _, _, _ = env.reset(reset_n_times_loser=True)
        else:
            current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
        current_state = current_state[agent_n]

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        while not done:
            possible_actions = None
            if who_moves_first == 'Enemy':
                env.step_enemies()
                """_, _, if_lose = env.check_winner_gameover_agent(current_state[0], current_state[1])
                if not if_lose:"""
                if 'causal' in alg:
                    enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                    possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                            goals_nearby_all_agents)
                action = agent.choose_action(current_state, possible_actions)
                next_state = env.step_agent(action)[0]
                """else:
                    next_state = current_state"""
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]

            else:  # who_moves_first == 'Agent':
                if 'Causal' in alg:
                    enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                    possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                            goals_nearby_all_agents)

                action = agent.choose_action(current_state, possible_actions)
                next_state = env.step_agent(action)[0]
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
                _, dones, _ = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
                if not dones[agent_n]:
                    env.step_enemies()

            rewards, dones, if_lose = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
            reward = int(rewards[agent_n])
            done = dones[agent_n]  # If agent wins, end loop and restart

            if possible_actions is not None and if_lose and [current_state] != [next_state]:
                print(f'\nLose: wrong causal gameover model in {alg}')
                print(f'New agents pos: {env.pos_agents[-1]}')
                print(f'Enemies pos: {env.pos_enemies[-1]} - enemies nearby: {enemies_nearby_all_agents}')
                print(f'Possible actions: {possible_actions} - chosen action: {action}')

            # Update the Q-table/values
            agent.update_Q(current_state, action, reward, next_state)

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_state = current_state[agent_n]
            else:
                current_state = next_state
            step_for_episode += 1

        if alg == 'QL_SoftmaxAnnealing' or alg == 'EpsilonGreedyAgent':
            agent.update_exp_fact(e)

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode


"""
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args, Transition):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ClassDQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(ClassDQN, self).__init__()

        self.hidden_layers = HIDDEN_LAYERS
        self.n_observations = n_observations
        self.n_actions = n_actions

        self.layer1 = nn.Linear(self.n_observations, self.hidden_layers)
        self.layer2 = nn.Linear(self.hidden_layers, self.hidden_layers)
        self.final_layer = nn.Linear(self.hidden_layers, self.n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.final_layer(x)


class DeepQNetwork:
    def __init__(self, rows, cols, n_agent_actions, n_episodes):
        self.rows = rows
        self.cols = cols
        self.n_agent_actions = n_agent_actions
        self.n_episodes = n_episodes

        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.lr = LEARNING_RATE

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.exp_proba = EXPLORATION_PROBA
        self.MIN_EXPLORATION_PROBA = MIN_EXPLORATION_PROBA
        self.EXPLORATION_DECREASING_DECAY = -np.log(self.MIN_EXPLORATION_PROBA) / (
                EXPLORATION_GAME_PERCENT * self.n_episodes)

        self.policy_net: ClassDQN = ClassDQN(self.cols * self.rows, self.n_agent_actions).to(self.device)
        self.target_net = ClassDQN(self.cols * self.rows, self.n_agent_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def choose_action(self, state, possible_actions=None):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        if possible_actions is not None and len(possible_actions) > 0:
            if np.random.uniform(0, 1) < self.exp_proba:  # exploration
                if len(possible_actions) > 0:
                    return torch.tensor([[possible_actions[torch.randint(0, len(possible_actions), (1,)).item()]]],
                                        device=self.device, dtype=torch.long)
                else:
                    return torch.tensor([[np.random.randint(0, self.n_agent_actions, 1)]], device=self.device, dtype=torch.long)
            else:  # exploitation
                actions_to_avoid = [s for s in range(self.n_agent_actions) if s not in possible_actions]

                actions_values = self.policy_net(state)
                for act_to_avoid in actions_to_avoid:
                    actions_values[:, act_to_avoid] = - 10000

                return actions_values.max(1).indices.view(1, 1)
        else:
            if np.random.uniform(0, 1) < self.exp_proba:  # exploration
                return torch.tensor([[np.random.randint(0, self.n_agent_actions, 1)]], device=self.device, dtype=torch.long)
            else:
                with torch.no_grad():
                    actions_values = self.policy_net(state)
                    return actions_values.max(1).indices.view(1, 1)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = self.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
        # These are the actions which would've been taken for each batch state according to policy_net
        action_batch = action_batch.view(-1)
        action_batch = action_batch.unsqueeze(1)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based on the "older" target_net;
        # selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected state value or 0 in
        # case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_memory(self, state, action, next_state, reward):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        action = torch.tensor([action], device=self.device)
        self.memory.push(state, action, next_state, reward, Transition=self.Transition)

    def update_exp_factor(self, episode):
        self.exp_proba = max(self.MIN_EXPLORATION_PROBA, np.exp(-self.EXPLORATION_DECREASING_DECAY * episode))

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)
        
def DQN_variations(env, n_act_agents, n_episodes, alg, who_moves_first):

    rows = env.rows
    cols = env.cols
    action_space_size = n_act_agents

    if 'DQN' in alg:
        agent = DeepQNetwork(rows, cols, action_space_size, n_episodes)
    elif 'DeepQNetwork_Multi' in alg:
        agent = DeepQNetwork(rows, cols, action_space_size, n_episodes)

    average_episodes_rewards = []
    steps_for_episode = []

    pbar = tqdm(range(n_episodes))
    time.sleep(1)
    for e in pbar:
        agent_n = 0
        if e == 0:
            current_state, _, _, _, _ = env.reset(reset_n_times_loser=True)
        else:
            current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
        current_state = current_state[agent_n]

        total_episode_reward = 0
        step_for_episode = 0
        done = False

        while not done:

            if who_moves_first == 'Enemy':
                env.step_enemies()
                _, _, if_lose = env.check_winner_gameover_agent(current_state[0], current_state[1])
                if_lose = False
                if not if_lose:
                    if 'causal' in alg:
                        enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                        possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                                goals_nearby_all_agents)
                    else:
                        possible_actions = None
                    general_state = np.zeros(rows * cols)
                    current_stateX = current_state[0]
                    current_stateY = current_state[1]
                    general_state[current_stateX * rows + current_stateY] = 1

                    action = agent.choose_action(general_state, possible_actions)
                    next_state = env.step_agent(action)[agent_n]
                else:
                    next_state = current_state
                    print('perso subito')
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]

            elif who_moves_first == 'Agent':
                if 'Causal' in alg:
                    enemies_nearby_all_agents, goals_nearby_all_agents = env.get_nearbies_agent()
                    possible_actions = get_possible_actions(n_act_agents, enemies_nearby_all_agents,
                                                            goals_nearby_all_agents)
                else:
                    possible_actions = None

                general_state = np.zeros(rows * cols)
                current_stateX = current_state[0]
                current_stateY = current_state[1]

                general_state[current_stateY * env.cols + current_stateX] = 1

                action = agent.choose_action(general_state, possible_actions)
                next_state = env.step_agent(action)[agent_n]
                new_stateX_ag = next_state[0]
                new_stateY_ag = next_state[1]
                _, dones, _ = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
                if not dones[agent_n]:
                    env.step_enemies()

            rewards, dones, if_lose = env.check_winner_gameover_agent(new_stateX_ag, new_stateY_ag)
            reward = int(rewards[agent_n])
            done = dones[agent_n]
            if_lose = if_lose

            if possible_actions is not None and len(possible_actions) > 0 and if_lose and [current_state] != [next_state]:
                print(f'\nLose: wrong causal gameover model in {alg}')
                print(f'New agents pos: {env.pos_agents[-1]}')
                print(f'Enemies pos: {env.pos_enemies[-1]} - enemies nearby: {enemies_nearby_all_agents}')
                print(f'Possible actions: {possible_actions} - chosen action: {action}')

            next_state = np.zeros(env.rows * env.cols)
            next_state[new_stateX_ag * rows + new_stateY_ag] = 1

            agent.update_memory(general_state, action, next_state, reward)

            agent.optimize_model()

            agent.update_target_net()

            total_episode_reward += reward

            if if_lose:
                current_state, _, _, _, _ = env.reset(reset_n_times_loser=False)
                current_state = current_state[agent_n]
            else:
                current_state = next_state

            step_for_episode += 1

        agent.update_exp_factor(e)

        average_episodes_rewards.append(total_episode_reward)
        steps_for_episode.append(step_for_episode)
        pbar.set_postfix_str(
            f"Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}")

    print(f'Average reward: {np.mean(average_episodes_rewards)}, Number of defeats: {env.n_times_loser}')
    return average_episodes_rewards, steps_for_episode
"""
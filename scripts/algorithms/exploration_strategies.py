import random
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete

from scripts.algorithms.class_dqn_and_buffer import ClassDQN, ReplayMemory


class EpsilonGreedyQAgent:

    def __init__(self, dict_env_parameters, dict_learning_parameters, n_episodes, if_deep=False, predefined_q_table=None):
        self.rows = dict_env_parameters['rows']
        self.cols = dict_env_parameters['cols']
        self.n_actions = int(dict_env_parameters['n_actions'])
        self.action_space = Discrete(self.n_actions, start=0)
        self.seed_value = dict_env_parameters['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

        self.n_episodes = n_episodes

        self.gamma = dict_learning_parameters['GAMMA']
        self.lr = dict_learning_parameters['LEARNING_RATE']
        self.start_exp_proba = dict_learning_parameters['START_EXPLORATION_PROBABILITY']
        self.min_exp_proba = dict_learning_parameters['MIN_EXPLORATION_PROBABILITY']
        self.exp_game_percent = dict_learning_parameters['EXPLORATION_GAME_PERCENT']

        self.EXPLORATION_DECREASING_DECAY = -np.log(self.min_exp_proba) / (self.exp_game_percent * self.n_episodes)

        self.if_deep = if_deep

        if self.if_deep:
            self.batch_size = dict_learning_parameters['BATCH_SIZE']
            self.tau = dict_learning_parameters['TAU']
            self.hidden_layers = dict_learning_parameters['HIDDEN_LAYERS']
            self.replay_memory_capacity = dict_learning_parameters['REPLAY_MEMORY_CAPACITY']
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net: ClassDQN = ClassDQN(self.cols * self.rows, self.action_space, self.hidden_layers).to(
                self.device)
            self.target_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
            self.memory = ReplayMemory(self.replay_memory_capacity)
        else:
            if predefined_q_table is not None:
                self.q_table = predefined_q_table
            else:
                self.q_table = np.zeros((self.rows, self.cols, self.n_actions))

    def choose_action(self, state: np.ndarray, possible_actions=None):
        if self.if_deep:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                if possible_actions is not None and len(possible_actions) > 0:
                    if np.random.uniform(0, 1) < self.start_exp_proba:  # exploration
                        if len(possible_actions) > 0:
                            return torch.tensor(
                                [[possible_actions[torch.randint(0, len(possible_actions), (1,)).item()]]],
                                device=self.device, dtype=torch.long)
                        else:
                            return torch.tensor([[np.random.randint(0, self.n_actions, 1)]], device=self.device,
                                                dtype=torch.long)
                    else:  # exploitation
                        actions_to_avoid = [s for s in range(self.n_actions) if s not in possible_actions]

                        actions_values = self.policy_net(state)
                        for act_to_avoid in actions_to_avoid:
                            actions_values[:, act_to_avoid] = - 10000

                        return actions_values.max(1).indices.view(1, 1)
                else:
                    if np.random.uniform(0, 1) < self.start_exp_proba:  # exploration
                        return torch.tensor([[np.random.randint(0, self.n_actions, 1)]], device=self.device,
                                            dtype=torch.long)
                    else:
                        with torch.no_grad():
                            actions_values = self.policy_net(state)
                            return actions_values.max(1).indices.view(1, 1)
        else:
            if np.random.uniform(0, 1) < self.start_exp_proba:  # exploration
                if possible_actions is not None and len(possible_actions) > 0:
                    action = random.sample(possible_actions, 1)[0]
                else:
                    action = np.random.randint(0, self.n_actions, size=1)[0]
            else:  # exploitation
                stateX = int(state[0])
                stateY = int(state[1])
                if possible_actions is not None and len(possible_actions) > 0:
                    all_actions = list(np.arange(0, self.n_actions, 1))
                    dict_all_actions = {}
                    for act in all_actions:
                        dict_all_actions[act] = self.q_table[stateX, stateY, act]

                    dict_valid_actions = {act: dict_all_actions[act] for act in possible_actions}
                    action, _ = max(dict_valid_actions.items(), key=lambda x: x[1])
                else:
                    action = np.argmax(self.q_table[stateX, stateY, :])

            return action

    def update_Q_or_memory(self, state, action, reward, next_state):
        if self.if_deep:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward = torch.tensor([reward], device=self.device)
            action = torch.tensor([action], device=self.device)
            self.memory.push(state, action, next_state, reward, Transition=self.Transition)
        else:
            stateX = int(state[0])
            stateY = int(state[1])
            next_stateX = int(next_state[0])
            next_stateY = int(next_state[1])
            current_q_value = self.q_table[stateX, stateY, action]
            max_next_q_value = np.max(self.q_table[next_stateX, next_stateY, :])

            new_q_value = current_q_value + self.lr * (reward + self.gamma * max_next_q_value - current_q_value)
            self.q_table[stateX, stateY, action] = new_q_value

    def update_exp_fact(self, episode):  # update exploration probability
        self.start_exp_proba = max(self.min_exp_proba, np.exp(-self.EXPLORATION_DECREASING_DECAY * episode))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
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
        action_batch = action_batch.unsqueeze(1).to(torch.int64)
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

    def update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
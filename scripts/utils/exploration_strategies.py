import random
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Discrete
from scripts.utils.dqn_class_and_memory import ClassDQN, ReplayMemory

"""class EpsilonGreedyQAgent:

    def __init__(self, dict_env_parameters, dict_learning_parameters, n_episodes, if_deep=False):
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
        self.exp_proba = dict_learning_parameters['START_EXPLORATION_PROBABILITY']
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
            self.policy_net: ClassDQN = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(
                self.device)
            self.target_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())

            self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

            self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
            self.memory = ReplayMemory(self.replay_memory_capacity)
        else:
            predefined_q_table = dict_learning_parameters['KNOWLEDGE_TRANSFERRED']
            if predefined_q_table is not None:
                self.q_table = predefined_q_table
            else:
                self.q_table = np.zeros((self.rows, self.cols, self.n_actions))

    def choose_action(self, state, possible_actions=None):

        if self.if_deep:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

                if possible_actions is not None and len(possible_actions) > 0:
                    if np.random.uniform(0, 1) < self.exp_proba:  # exploration
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
                    if np.random.uniform(0, 1) < self.exp_proba:  # exploration
                        return torch.tensor([[np.random.randint(0, self.n_actions, 1)]], device=self.device,
                                            dtype=torch.long)
                    else:
                        with torch.no_grad():
                            actions_values = self.policy_net(state)
                            return actions_values.max(1).indices.view(1, 1)
        else:
            if np.random.uniform(0, 1) < self.exp_proba:  # exploration
                if possible_actions is not None and len(possible_actions) > 0:  # causal filter
                    if len(possible_actions) == 1:
                        possible_actions = [int(s) for s in possible_actions]
                        action = random.sample(possible_actions, 1)[0]
                    else:
                        action = possible_actions[0]
                else:
                    action = np.random.randint(0, self.n_actions, size=1)[0]  # classic
            else:  # exploitation
                stateX = int(state[0])
                stateY = int(state[1])
                if possible_actions is not None and len(possible_actions) > 0:  # causal filter
                    all_actions = list(np.arange(0, self.n_actions, 1))
                    dict_all_actions = {}
                    for act in all_actions:
                        dict_all_actions[act] = self.q_table[stateX, stateY, act]

                    dict_valid_actions = {act: dict_all_actions[act] for act in possible_actions}
                    action, _ = max(dict_valid_actions.items(), key=lambda x: x[1])
                else:  # classic
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
        self.exp_proba = max(self.min_exp_proba, np.exp(-self.EXPLORATION_DECREASING_DECAY * episode))

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
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)"""


class EpsilonGreedyQAgent:
    def __init__(self, dict_env_parameters, dict_learning_parameters, n_episodes, if_deep=False):
        self.n_episodes = n_episodes
        self.if_deep = if_deep
        self._init_env_parameters(dict_env_parameters)
        self._init_learning_parameters(dict_learning_parameters)

        if self.if_deep:
            self._init_deep_learning_components(dict_learning_parameters)
        else:
            self._init_q_table(dict_learning_parameters)

    def _init_env_parameters(self, dict_env_parameters):
        self.rows = dict_env_parameters['rows']
        self.cols = dict_env_parameters['cols']
        self.n_actions = int(dict_env_parameters['n_actions'])
        self.action_space = Discrete(self.n_actions, start=0)
        self.seed_value = dict_env_parameters['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

    def _init_learning_parameters(self, dict_learning_parameters):
        self.gamma = dict_learning_parameters['GAMMA']
        self.lr = dict_learning_parameters['LEARNING_RATE']
        self.exp_proba = dict_learning_parameters['START_EXPLORATION_PROBABILITY']
        self.min_exp_proba = dict_learning_parameters['MIN_EXPLORATION_PROBABILITY']
        self.exp_game_percent = dict_learning_parameters['EXPLORATION_GAME_PERCENT']
        self.EXPLORATION_DECREASING_DECAY = -np.log(self.min_exp_proba) / (self.exp_game_percent * self.n_episodes)

    def _init_deep_learning_components(self, dict_learning_parameters):
        self.batch_size = dict_learning_parameters['BATCH_SIZE']
        self.tau = dict_learning_parameters['TAU']
        self.hidden_layers = dict_learning_parameters['HIDDEN_LAYERS']
        self.replay_memory_capacity = dict_learning_parameters['REPLAY_MEMORY_CAPACITY']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(self.replay_memory_capacity)

    def _init_q_table(self, dict_learning_parameters):
        predefined_q_table = dict_learning_parameters.get('KNOWLEDGE_TRANSFERRED')
        self.q_table = predefined_q_table if predefined_q_table is not None else np.zeros(
            (self.rows, self.cols, self.n_actions))

    def choose_action(self, state, possible_actions=None):
        if self.if_deep:
            return self._choose_action_deep(state, possible_actions)
        else:
            return self._choose_action_q_table(state, possible_actions)

    def _choose_action_deep(self, state, possible_actions):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if np.random.uniform(0, 1) < self.exp_proba and possible_actions is not None:
                action = random.choice(possible_actions)
            else:
                q_values = self.policy_net(state)
                if possible_actions is not None:
                    q_values[:, [i for i in range(self.n_actions) if i not in possible_actions]] = -float('inf')

                action = q_values.max(1)[1].item()

        return action

    def _choose_action_q_table(self, state, possible_actions):
        if np.random.uniform(0, 1) < self.exp_proba:
            action = np.random.choice(possible_actions) if possible_actions is not None and len(possible_actions) > 0 else np.random.randint(self.n_actions)

        else:
            q_values = self.q_table[state[0], state[1], :]
            if possible_actions is not None:
                if len(possible_actions) > 0:
                    q_values = np.array([q_values[a] if a in possible_actions else -np.inf for a in range(self.n_actions)])
            action = np.argmax(q_values)
        return action

    def update_Q_or_memory(self, state, action, reward, next_state):
        if self.if_deep:
            self._update_memory(state, action, reward, next_state)
        else:
            self._update_q_table(state, action, reward, next_state)

    def _update_memory(self, state, action, reward, next_state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        action = torch.tensor([action], device=self.device)
        self.memory.push(state, action, next_state, reward, Transition=self.Transition)

    def _update_q_table(self, state, action, reward, next_state):
        stateX = int(state[0])
        stateY = int(state[1])
        next_stateX = int(next_state[0])
        next_stateY = int(next_state[1])
        current_q_value = self.q_table[stateX, stateY, action]
        max_next_q_value = np.max(self.q_table[next_stateX, next_stateY, :])

        new_q_value = current_q_value + self.lr * (reward + self.gamma * max_next_q_value - current_q_value)
        self.q_table[stateX, stateY, action] = new_q_value

    def update_exp_fact(self, episode):  # update exploration probability
        self.exp_proba = max(self.min_exp_proba, np.exp(-self.EXPLORATION_DECREASING_DECAY * episode))

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
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


class BoltzmannQAgent:
    def __init__(self, dict_env_parameters, dict_learning_parameters, n_episodes, if_deep=False,
                 predefined_q_table=None):
        self.init_environment(dict_env_parameters)
        self.init_learning_parameters(dict_learning_parameters, n_episodes)
        self.if_deep = if_deep
        self.init_model_or_table(predefined_q_table, dict_learning_parameters)

    def init_environment(self, params):
        self.rows, self.cols = params['rows'], params['cols']
        self.n_actions = int(params['n_actions'])
        self.seed_value = params['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

    def init_learning_parameters(self, params, n_episodes):
        self.temperature = params['START_EXPLORATION_PROBABILITY']
        self.min_exp_proba = params['MIN_EXPLORATION_PROBABILITY']
        self.gamma, self.lr = params['GAMMA'], params['LEARNING_RATE']
        exp_decay = -np.log(self.min_exp_proba) / (params['EXPLORATION_GAME_PERCENT'] * n_episodes)
        self.EXPLORATION_DECREASING_DECAY = exp_decay

    def init_model_or_table(self, predefined_q_table, learning_params):
        if self.if_deep:
            self.init_deep_learning_components(learning_params)
        else:
            self.q_table = predefined_q_table if predefined_q_table is not None else np.zeros(
                (self.rows, self.cols, self.n_actions))

    def init_deep_learning_components(self, learning_params):
        params = learning_params  # Assuming dict_learning_parameters is accessible or passed
        self.batch_size, self.tau = params['BATCH_SIZE'], params['TAU']
        self.hidden_layers = params['HIDDEN_LAYERS']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(params['REPLAY_MEMORY_CAPACITY'])

    def softmax(self, values):
        max_val = np.max(values)  # Ensure numerical stability
        exp_values = np.exp((values - max_val) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def choose_action(self, state: np.ndarray, possible_actions: list = None):
        if self.if_deep:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            values = self.policy_net(state).squeeze().detach().cpu().numpy()
        else:
            stateX, stateY = int(state[0]), int(state[1])
            values = self.q_table[stateX, stateY, :]
        if possible_actions is not None:
            if len(possible_actions) > 0:
                values = values[possible_actions]

        probabilities = self.softmax(values)
        return np.random.choice(possible_actions if possible_actions is not None else np.arange(self.n_actions),
                                p=probabilities)

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

    def update_exp_fact(self, episode):
        pass

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
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


class SoftmaxAnnealingQAgent:
    def __init__(self, dict_env_parameters, dict_learning_parameters, n_episodes, if_deep=False,
                 predefined_q_table=None):
        self.init_environment(dict_env_parameters)
        self.init_learning_parameters(dict_learning_parameters, n_episodes)
        self.if_deep = if_deep
        self.init_model_or_table(predefined_q_table, dict_learning_parameters)

    def init_environment(self, params):
        self.rows, self.cols = params['rows'], params['cols']
        self.n_actions = int(params['n_actions'])
        self.seed_value = params['seed_value']
        np.random.seed(self.seed_value)
        random.seed(self.seed_value)

    def init_learning_parameters(self, params, n_episodes):
        self.temperature = params['START_EXPLORATION_PROBABILITY']
        self.min_exp_proba = params['MIN_EXPLORATION_PROBABILITY']
        self.gamma, self.lr = params['GAMMA'], params['LEARNING_RATE']
        exp_decay = -np.log(self.min_exp_proba) / (params['EXPLORATION_GAME_PERCENT'] * n_episodes)
        self.EXPLORATION_DECREASING_DECAY = exp_decay

    def init_model_or_table(self, predefined_q_table, learning_params):
        if self.if_deep:
            self.init_deep_learning_components(learning_params)
        else:
            self.q_table = predefined_q_table if predefined_q_table is not None else np.zeros(
                (self.rows, self.cols, self.n_actions))

    def init_deep_learning_components(self, learning_params):
        params = learning_params  # Assuming dict_learning_parameters is accessible or passed
        self.batch_size, self.tau = params['BATCH_SIZE'], params['TAU']
        self.hidden_layers = params['HIDDEN_LAYERS']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(params['REPLAY_MEMORY_CAPACITY'])

    def softmax(self, values):
        max_val = np.max(values)  # Ensure numerical stability
        exp_values = np.exp((values - max_val) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        return probabilities

    def choose_action(self, state, possible_actions=None):
        if self.if_deep:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            values = self.policy_net(state).squeeze().detach().cpu().numpy()
        else:
            stateX, stateY = int(state[0]), int(state[1])
            values = self.q_table[stateX, stateY, :]

        if possible_actions is not None:
            if len(possible_actions) > 0:
                values = values[possible_actions]
        probabilities = self.softmax(values)

        return np.random.choice(possible_actions if possible_actions is not None else np.arange(self.n_actions),
                                p=probabilities)

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

    def update_exp_fact(self, episode):
        self.temperature = max(self.min_exp_proba, np.exp(-self.EXPLORATION_DECREASING_DECAY * episode))

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
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


class ThompsonSamplingQAgent:
    def __init__(self, dict_env_parameters, dict_learning_parameters, n_episodes, if_deep=False,
                 predefined_q_table=None):
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
        self.if_deep = if_deep

        if self.if_deep:
            self.initialize_deep_learning_components(dict_learning_parameters)
        else:
            self.initialize_non_deep_components(predefined_q_table)

    def initialize_deep_learning_components(self, dict_learning_parameters):
        self.batch_size = dict_learning_parameters['BATCH_SIZE']
        self.tau = dict_learning_parameters['TAU']
        self.hidden_layers = dict_learning_parameters['HIDDEN_LAYERS']
        self.replay_memory_capacity = dict_learning_parameters['REPLAY_MEMORY_CAPACITY']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        alpha = beta = 1.0  # Initialized here to avoid redundancy
        self.alpha = torch.full((self.rows, self.cols, self.n_actions), alpha, dtype=torch.float, device=self.device)
        self.beta = torch.full((self.rows, self.cols, self.n_actions), beta, dtype=torch.float, device=self.device)

        self.policy_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net = ClassDQN(self.cols * self.rows, self.n_actions, self.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(self.replay_memory_capacity)

    def initialize_non_deep_components(self, predefined_q_table):
        if predefined_q_table is None:
            alpha = beta = 1.0  # Initialized here to avoid redundancy
            self.alpha = np.full((self.rows, self.cols, self.n_actions), alpha)
            self.beta = np.full((self.rows, self.cols, self.n_actions), beta)
        else:
            self.alpha, self.beta = predefined_q_table

    def choose_action(self, state, possible_actions=None):
        sampled_values = self.sample_beta_distribution(state)

        if possible_actions is not None and len(possible_actions) > 0:
            chosen_action = max(possible_actions, key=lambda x: sampled_values[x])
        else:
            chosen_action = np.argmax(sampled_values).item()  # Works for both numpy arrays and torch tensors

        return chosen_action

    def sample_beta_distribution(self, state):
        if self.if_deep:
            # Assuming state is an array-like or tensor with two elements representing coordinates
            # Convert state to a tensor if it's not already, ensuring it's on the correct device
            state_tensor = torch.as_tensor(state, dtype=torch.long, device=self.device)

            # Direct indexing into alpha and beta using state coordinates
            # This assumes state_tensor is something like [x, y] for grid coordinates
            alpha_tensor = self.alpha[state_tensor[0], state_tensor[1], :]
            beta_tensor = self.beta[state_tensor[0], state_tensor[1], :]

            # Sampling from the beta distribution
            beta_dist = torch.distributions.Beta(alpha_tensor, beta_tensor)
            sample = beta_dist.sample()
            # Ensure to return a numpy array if needed, converting only at the end
            return sample.cpu().numpy() if self.device.type == 'cuda' else sample.numpy()
        else:
            # Non-deep case with numpy
            stateX, stateY = int(state[0]), int(state[1])  # Assuming state is indexed directly
            return np.random.beta(self.alpha[stateX, stateY], self.beta[stateX, stateY])

    def update_Q_or_memory(self, state, action, reward, next_state):
        stateX, stateY = int(state[0]), int(state[1])
        update_value = 1 if reward == 1 else -1
        getattr(self, 'alpha' if update_value > 0 else 'beta')[stateX, stateY, action] += abs(update_value)

    def optimize_model(self):
        if self.if_deep and len(self.memory) >= self.batch_size:
            self.perform_deep_optimization()

    def perform_deep_optimization(self):
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
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                    1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

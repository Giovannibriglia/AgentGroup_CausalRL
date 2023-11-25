import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque
import gym

# Define the DQN model
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the experience replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the DQNAgent
class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=32, gamma=0.95, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayBuffer(2000)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            q_values = self.model(state)
            return np.argmax(q_values.cpu().numpy())

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool, device=self.device)
        non_final_next_states = torch.stack([torch.tensor(s, dtype=torch.float32, device=self.device) for s in batch.next_state if s is not None])

        state_batch = torch.stack([torch.tensor(s, dtype=torch.float32, device=self.device) for s in batch.state])
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)

        state_action_values = self.model(state_batch).gather(1, action_batch.unsqueeze(1))

        next_state_values = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        next_state_values[non_final_mask] = self.model(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Training the DQN
env = gym.make('FrozenLake-v1')
state_size = env.observation_space.n
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)

num_episodes = 1000

for episode in range(num_episodes):
    state = env.reset()
    state = np.identity(state_size)[state[0]]
    total_reward = 0

    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info, _ = env.step(action)
        next_state = np.identity(state_size)[next_state]  # one-hot encoding
        total_reward += reward

        agent.memory.push(state, action, next_state, reward, done)
        state = next_state

        agent.train()

        if done:
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

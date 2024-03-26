import random
from collections import deque
from collections import deque
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from gymnasium.spaces import Discrete
import global_variables
from scripts import exploration_strategies


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

    def __init__(self, n_observations, n_actions, HIDDEN_LAYERS):
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
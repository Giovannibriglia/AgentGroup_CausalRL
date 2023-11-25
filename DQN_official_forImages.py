import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(self._calculate_fc_input_size(input_channels), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _calculate_fc_input_size(self, input_channels):
        # Dummy input to calculate the size of the flattened output
        x = torch.zeros((1, input_channels, 84, 84), dtype=torch.float32).cuda()
        x = self._convolutional_layers(x)
        return x.view(1, -1).size(1)

    def _convolutional_layers(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

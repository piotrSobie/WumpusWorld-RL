import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.fc1 = nn.Linear(in_features=input_dims, out_features=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=output_dims)

    def forward(self, state):
        t = F.relu(self.fc1(state))
        t = F.relu(self.fc2(t))
        actions = self.out(t)
        return actions

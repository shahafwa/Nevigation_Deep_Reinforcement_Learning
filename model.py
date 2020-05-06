import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_size1 = 128
        hidden_size2 = 64
        hidden_size3 = 32
        self.fc1 = nn.Linear(state_size,hidden)
        self.fc2 = nn.Linear(hidden, int(hidden/2))      
        self.fc3 = nn.Linear(int(hidden/2), int(hidden/4))      
        self.fc4 = nn.Linear(int(hidden/4), action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

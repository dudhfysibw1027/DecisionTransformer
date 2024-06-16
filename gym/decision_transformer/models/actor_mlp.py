import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action=1):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.mean_layer = nn.Linear(300, action_dim)
        self.log_std_layer = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        mean = torch.tanh(self.mean_layer(x)) * self.max_action
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)  # Clamping log_std to keep std in reasonable bounds
        return mean, log_std

    def get_action(self, state, actions=None, rewards=None, target_returns=None, timesteps=None):
        mean, log_std = self.forward(state[-1].float())
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        action = normal.sample()
        action = torch.clamp(action, -1, 1)  # Ensure action bounds
        return action
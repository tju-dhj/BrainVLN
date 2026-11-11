"""Action head shared by the MLANet checkpoint."""

import torch.nn as nn
from torch.distributions import Categorical


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return Categorical(logits=self.linear(x).float())


class ILPolicy(nn.Module):
    def __init__(self, net, dim_actions: int) -> None:
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.action_distribution = CategoricalNet(self.net.output_size, dim_actions)

    def act(self, observations, rnn_states, prev_actions, masks, deterministic=True):
        features, rnn_states = self.net(observations, rnn_states, prev_actions, masks)
        distribution = self.action_distribution(features)
        if deterministic:
            action = distribution.probs.argmax(dim=-1, keepdim=True)
        else:
            action = distribution.sample().unsqueeze(-1)
        return action, rnn_states

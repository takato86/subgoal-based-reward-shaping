
import torch
import torch.nn as nn


def init_weights_zeros(m):
    if isinstance(m, nn.Linear):
        nn.init.zeros_(m.weight)
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, d_input, d_output, act_limit):
        super(Actor, self).__init__()
        self.actor_network = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_output),
            nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        action = self.actor_network(obs)
        return action * self.act_limit


class Critic(nn.Module):
    def __init__(self, d_input, d_output):
        super(Critic, self).__init__()
        self.critic_network = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_output),
            nn.Identity()
        )

    def forward(self, obs, action):
        q_value = self.critic_network(torch.cat([obs, action], dim=-1))
        return torch.squeeze(q_value, -1)


class ActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(ActorCritic, self).__init__()
        input_dim = observation_space.shape[1]
        self.actor = Actor(input_dim, action_space.shape[0], action_space.high[0])
        self.critic = Critic(input_dim + action_space.shape[0], 1)

    def act(self, obs):
        with torch.no_grad():
            return self.actor(obs).cpu().numpy()

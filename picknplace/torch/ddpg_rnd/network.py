from torch import nn


class TargetRandomNetwork(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_output)
        )

    def forward(self, obs):
        return self.network(obs)


class Predictor(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_input, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, d_output)
        )

    def forward(self, obs):
        return self.network(obs)

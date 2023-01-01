import torch
from torch.optim import Adam
from picknplace.utils.constants import device
from picknplace.torch.ddpg_rnd.network import TargetRandomNetwork
from picknplace.torch.ddpg_rnd.network import Predictor
from picknplace.utils.mpi_torch import mpi_avg_grad


class RNDAgent:
    """Random Network Distillation.
    """
    def __init__(self, observation_space, feature_size, lr):
        input_dim = observation_space.shape[1]
        self.target = TargetRandomNetwork(input_dim, feature_size).to((device()))
        self.predictor = Predictor(input_dim, feature_size).to(device())
        self.predictor_optimizer = Adam(self.predictor.parameters(), lr=lr)

    def update(self, datum):
        self.predictor_optimizer.zero_grad()
        target_feats = self.target(datum)
        predictor_feats = self.predictor(datum)
        mse_loss = torch.mean((target_feats - predictor_feats)**2)
        mse_loss.backward()
        mpi_avg_grad(self.predictor)
        self.predictor_optimizer.step()
        return mse_loss.detach().cpu().numpy()

    def act(self, obs):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device())
        target_feature = self.target(obs_tensor)
        predictor_feature = self.predictor(obs_tensor)
        intrinsic_reward = torch.mean((target_feature - predictor_feature)**2)
        return intrinsic_reward.detach().cpu().numpy()

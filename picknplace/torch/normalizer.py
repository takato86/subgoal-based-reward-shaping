import numpy as np
from pdrl.utils.mpi import mpi_sum, num_procs
from pdrl.utils.constants import device
import torch


class Normalizer:
    def __call__(self, obs):
        """正規化を実施する。"""
        raise NotImplementedError


class Zscorer(Normalizer):
    def __init__(self, norm_clip, norm_eps):
        self.norm_clip = norm_clip
        self.norm_eps = norm_eps
        self.sum_obs = 0
        self.sum_obs_torch = torch.as_tensor(0, device=device())
        self.sum_sq_obs = 0
        self.sum_sq_obs_torch = torch.as_tensor(0, device=device())
        self.count_obs = 0
        self.count_obs_torch = torch.as_tensor(0, device=device())
        self.device = device

    def __call__(self, obs):
        input_type = type(obs)

        if input_type == torch.Tensor:
            obs = obs.cpu().numpy()

        z = (obs - self._mean()) / self._std()
        clipped_z = np.clip(z, a_min=-self.norm_clip, a_max=self.norm_clip)

        if input_type == np.ndarray:
            return clipped_z
        else:
            # torch.Double型に変換されることを防ぐ
            return torch.as_tensor(clipped_z, dtype=torch.float32, device=device())

    def batch(self, obs_batch: torch.Tensor):
        # torchのまま計算
        with torch.no_grad():
            z = (obs_batch - self._mean_torch()) / self._std_torch()
            clipped_z = torch.clip(z, -self.norm_clip, self.norm_clip)

        return clipped_z

    def _mean(self):
        return self.sum_obs / self.count_obs

    def _mean_torch(self):
        return self.sum_obs_torch / self.count_obs_torch

    def _std(self):
        var = self.sum_sq_obs / self.count_obs - (self.sum_obs / self.count_obs)**2
        var[var < 0] = 0
        std = (var)**0.5
        # 計算元となる経験が溜まっていない場合は誤差により負の値になる。
        std = np.nan_to_num(std, nan=0.0)
        norm_eps_arr = np.full_like(std, self.norm_eps**2)
        return np.max([std, norm_eps_arr], axis=0)

    def _std_torch(self):
        var = self.sum_sq_obs_torch / self.count_obs_torch - (self.sum_obs_torch / self.count_obs_torch)**2
        var[var < 0] = 0
        std = torch.sqrt(var)
        std = torch.nan_to_num(std, nan=0.0)
        norm_eps_arr = torch.full_like(std, self.norm_eps ** 2)
        return torch.max(std, norm_eps_arr)

    def sync(self, obs):
        """各プロセスから収集した観測を同期する。"""
        input_type = type(obs)

        if input_type == torch.Tensor:
            obs = obs.cpu().numpy()

        self.sum_obs += np.sum(mpi_sum(obs), axis=0)
        self.sum_sq_obs += np.sum(mpi_sum(obs**2), axis=0)
        self.count_obs += num_procs() * obs.shape[0]
        self.sum_obs_torch = torch.as_tensor(self.sum_obs, dtype=torch.float32, device=device())
        self.sum_sq_obs_torch = torch.as_tensor(self.sum_sq_obs, dtype=torch.float32, device=device())
        self.count_obs_torch = torch.as_tensor(self.count_obs, dtype=torch.float32, device=device())

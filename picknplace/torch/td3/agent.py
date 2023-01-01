"""Twin Delayed DDPGの実装."""
import numpy as np
import torch
from torch.optim import Adam
import copy
from picknplace.torch.agent import Agent
from picknplace.torch.ddpg.agent import DDPGAgent
from picknplace.utils.mpi_torch import mpi_avg_grad


class TD3Agent(Agent):
    def __init__(self, observation_space, action_space, gamma, actor_lr, critic_lr, polyak, l2_action, clip_return,
                 is_pos_return, logger):
        self.ddpg = DDPGAgent(
            observation_space, action_space, gamma, actor_lr, critic_lr, polyak, l2_action, clip_return,
            is_pos_return, logger
        )
        self.max_act = torch.Tensor(action_space.high)
        self.second_q_net = copy.deepcopy(self.ddpg.actor_critic.critic)
        self.second_target_q_net = copy.deepcopy(self.second_q_net)
        self.second_critic_optimizer = Adam(self.second_q_net.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.return_clipping_params = {
            "min": -clip_return,
            "max": np.inf if is_pos_return else 0
        }

    def act(self, observation, noise_scale, epsilon):
        return self.ddpg.act(observation, noise_scale, epsilon)

    def compute_Q_loss(self, datum):
        o, a, r, o2, d = datum['obs'], datum['act'], datum['rew'], datum['obs2'], datum['done']
        q_value = self.ddpg.actor_critic.critic(o, a / self.max_act)
        second_q_value = self.second_q_net(o, a / self.max_act)

        with torch.no_grad():
            # ターゲットネットワークの更新をしないようにする。
            target_act = self.ddpg.target_ac.actor(o2)
            q_pi_target = self.ddpg.target_ac.critic(o2, target_act / self.max_act)
            second_q_target = self.second_target_q_net(o2, target_act / self.max_act)
            min_q = torch.min(q_pi_target, second_q_target)
            backup = r + self.gamma * (1 - d) * min_q
            backup = torch.clip(backup, **self.return_clipping_params)

        loss_q = torch.mean((q_value - backup)**2)
        second_loss_q = torch.mean((second_q_value - backup)**2)
        # "detach" Returns a new Tensor, detached from the current graph.
        loss_info = dict(QVals=q_value.detach().numpy())
        second_loss_info = dict(QVals=second_q_value.detach().numpy())
        return loss_q, loss_info, second_loss_q, second_loss_info

    def update(self, datum):
        self.ddpg.critic_optimizer.zero_grad()
        loss_q, loss_q_info, second_loss_q, _ = self.compute_Q_loss(datum)
        qs = loss_q_info["QVals"]
        loss_q.backward()
        second_loss_q.backward()
        mpi_avg_grad(self.ddpg.actor_critic.critic)
        self.ddpg.critic_optimizer.step()
        mpi_avg_grad(self.second_q_net)
        self.second_critic_optimizer.step()
        loss_pi = self.ddpg.update_actor(datum)
        loss_q_numpy, loss_pi_numpy = loss_q.detach().numpy(), loss_pi.detach().numpy()
        return loss_q_numpy, loss_pi_numpy, np.mean(qs, 0)

    def sync_target(self):
        self.ddpg.sync_target()
        with torch.no_grad():
            for p, p_target in zip(self.second_q_net.parameters(), self.second_target_q_net.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

    def save(self, path):
        torch.save(
            {
                'second_q_net': self.second_q_net.state_dict(),
                'second_target_q_net': self.second_target_q_net.state_dict(),
                'target_ac': self.ddpg.target_ac.state_dict(),
                'ac': self.ddpg.actor_critic.state_dict()
            },
            path
        )
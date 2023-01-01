"""DDPGの実装"""
import torch
from torch.optim import Adam
import copy
import numpy as np
from picknplace.utils.constants import device
from picknplace.torch.agent import Agent
from picknplace.torch.ddpg.network import ActorCritic
from picknplace.utils.mpi_torch import mpi_avg_grad, sync_params


class DDPGAgent(Agent):
    def __init__(self, observation_space, action_space, gamma, actor_lr, critic_lr, polyak, l2_action, clip_return,
                 is_pos_return, logger):
        self.gamma = gamma
        self.actor_critic = ActorCritic(observation_space, action_space).to(device())
        # MPIプロセス間で重みを共通化
        sync_params(self.actor_critic.actor)
        sync_params(self.actor_critic.critic)
        self.target_ac = copy.deepcopy(self.actor_critic)
        self.actor_optimizer = Adam(self.actor_critic.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.actor_critic.critic.parameters(), lr=critic_lr)
        self.polyak = polyak
        self.act_dim = action_space.shape[0]
        self.max_act_tensor = torch.Tensor(action_space.high).to(device())
        self.max_act = action_space.high
        self.logger = logger
        self.l2_action = l2_action
        self.random_act = action_space.sample
        self.return_clipping_params = {
            "min": -clip_return,
            "max": np.inf if is_pos_return else 0
        }

    def act(self, observation, noise_scale, epsilon):
        action = self.actor_critic.act(
            torch.as_tensor(observation, dtype=torch.float32, device=device())
        )
        # gaussian noise
        # TODO add clipping noise
        action += noise_scale * self.max_act * np.random.randn(self.act_dim)
        # binomialは1をepsilonの確率で返す。1になった時はrandom_actが実行される。これを次元毎に。
        # epsilon-greedy
        action = np.clip(action, -self.max_act, self.max_act)
        action += np.random.binomial(1, epsilon, self.act_dim) * (self.random_act() - action)
        action = action.reshape(-1)
        return action

    def compute_Q_loss(self, datum):
        o, a, r, o2, d = datum['obs'], datum['act'], datum['rew'], datum['obs2'], datum['done']
        q_value = self.actor_critic.critic(o, a / self.max_act_tensor)

        with torch.no_grad():
            # ターゲットネットワークの更新をしないようにする。
            q_pi_target = self.target_ac.critic(o2, self.target_ac.actor(o2) / self.max_act_tensor)
            backup = r + self.gamma * (1 - d) * q_pi_target
            backup = torch.clip(backup, **self.return_clipping_params)

        loss_q = torch.mean((q_value - backup)**2)
        # "detach" Returns a new Tensor, detached from the current graph.
        loss_info = dict(QVals=q_value.detach().cpu().numpy())
        return loss_q, loss_info

    def compute_pi_loss(self, datum):
        o = datum['obs']
        normalized_a_pi = self.actor_critic.actor(o) / self.max_act_tensor
        q_pi = self.actor_critic.critic(o, normalized_a_pi)
        loss_pi = -torch.mean(q_pi)
        # 正則化項
        loss_pi += self.l2_action * torch.mean(torch.square(normalized_a_pi))
        return loss_pi

    def update(self, datum):
        """方策と価値関数の更新"""
        # Critic Networkの更新処理
        self.critic_optimizer.zero_grad()
        loss_q, loss_q_info = self.compute_Q_loss(datum)
        qs = loss_q_info["QVals"]
        loss_q.backward()
        # ここで平均化
        mpi_avg_grad(self.actor_critic.critic)
        self.critic_optimizer.step()

        loss_pi = self.update_actor(datum)

        loss_q_numpy, loss_pi_numpy = loss_q.detach().cpu().numpy(), loss_pi.detach().cpu().numpy()
        return loss_q_numpy, loss_pi_numpy, np.mean(qs, 0)

    def update_actor(self, datum):
        # Freeze Critic Network
        for p in self.actor_critic.critic.parameters():
            p.requires_grad = False

        self.actor_optimizer.zero_grad()
        loss_pi = self.compute_pi_loss(datum)
        loss_pi.backward()
        # ここで勾配を平均化
        mpi_avg_grad(self.actor_critic.actor)
        self.actor_optimizer.step()

        for p in self.actor_critic.critic.parameters():
            p.requires_grad = True

        return loss_pi

    def sync_target(self):
        """Target Networkとの同期"""
        with torch.no_grad():
            for p, p_target in zip(self.actor_critic.parameters(), self.target_ac.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

    def save(self, path):
        torch.save(
            {
                'target_ac': self.target_ac.state_dict(),
                'ac': self.actor_critic.state_dict()
            },
            path
        )

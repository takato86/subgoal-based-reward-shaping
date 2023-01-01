import numpy as np
import os
import logging
from scipy.special import logsumexp

from pinball.src.fourier import FourierBasis

logger = logging.getLogger(__name__)


class Actor(object):
    def __init__(self, rng, n_actions, n_features, lr_theta, temperature=1.0):
        self.rng = rng
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.zeros((n_actions, self.n_features))
        self.lr_theta = lr_theta
        self.temperature = temperature

    def update(self, feat, action, q_value):
        lr = self.lr_theta/np.linalg.norm(feat)
        self.theta += lr * q_value * self.grad(feat, action)

    def act(self, feat):
        return int(self.rng.choice(self.n_actions, 1, p=self.pmf(feat)))

    def pmf(self, feat):
        """
        Returns: ndarray(#actions)
        """
        # TODO check
        energy = self.value(feat)  # / self.temperature # >> (1, #actions)
        return np.exp(energy - logsumexp(energy))

    def value(self, feat, action=None):
        energy = np.dot(self.theta, feat)
        if action is None:
            return energy
        return energy[action]

    def grad(self, feat, action):
        action_dist = self.pmf(feat)
        action_dist = action_dist.reshape(1, len(action_dist))
        grad = - np.multiply(action_dist.T, feat)
        grad[action] += feat
        return grad


class Critic(object):
    def __init__(self, n_actions, n_features, gamma, lr_q):
        self.w_q = np.zeros((n_actions, n_features))
        self.gamma = gamma
        self.lr_q = lr_q

    def update(self, feat, action, reward, next_feat, next_action, done):
        # Q値の算出はfeatと重みの内積で算出する．
        lr = self.lr_q/np.linalg.norm(feat)
        update_target = reward
        if not done:
            update_target += self.gamma * self.value(next_feat, next_action)
        td_error = update_target - self.value(feat, action)
        self.w_q[:] += lr * td_error * self.grad(feat, action)
        return td_error

    def value(self, feat, action=None):
        if action is None:
            return np.dot(self.w_q[:], feat)
        return np.dot(self.w_q[action], feat)

    def grad(self, feat, action):
        # TODO Q(s,a) = w_a * φ_a(s)
        # 本来なら．該当actionだけfeatでそれ以外は0
        grad = np.zeros(self.w_q.shape)
        grad[action] = feat
        return grad


class ActorCriticAgent(object):
    def __init__(self, seed, action_space, observation_space, basis_order,
                 epsilon, gamma, lr_theta, lr_q):
        self.action_space = action_space
        self.basis_order = basis_order
        self.shape_state = observation_space.shape  # case Pinball Box(4,0)
        self.fourier_basis = FourierBasis(
            self.shape_state[0], observation_space, order=self.basis_order
        )
        self.n_features = self.fourier_basis.getNumBasisFunctions()
        self.rng = np.random.RandomState(seed=seed)

        # parameters
        # Hyper parameters
        self.epsilon = epsilon
        self.actor = Actor(self.rng, action_space.n, self.n_features, lr_theta)
        self.critic = Critic(action_space.n, self.n_features, gamma, lr_q)

        # variables for analysis
        self.td_error_list = []
        self.td_error_list_meta = []
        self.vis_action_dist = np.zeros(action_space.n)
        self.max_error = 0

    def act(self, observation):
        feature = self.fourier_basis(observation)
        self.vis_action_dist = self.actor.pmf(feature)
        return self.actor.act(feature)

    def update(self, pre_obs, pre_a, r, obs, a, done, info):
        """
        1. Update critic(pi_Omega: Intra Q Learning,
                         pi_u: IntraAction Q learning)
        2. Improve actors
        """
        pre_feat = self.fourier_basis(pre_obs)
        feat = self.fourier_basis(obs)
        q_value = self.critic.value(pre_feat, pre_a)
        self.actor.update(pre_feat, pre_a, q_value)
        error = self.critic.update(pre_feat, pre_a, r, feat, a, done)
        # TODO this is baseline version.
        # q_u_list -= q_omega
        self.max_error = abs(error) if abs(error) > self.max_error\
            else self.max_error
        return 0

    def get_max_q(self, obs):
        feat = self.fourier_basis(obs)
        q_value = self.critic.value(feat)
        return max(q_value)

    def get_max_td_error(self):
        ret_error = self.max_error
        self.max_error = 0
        return ret_error

    def save_model(self, dir_path, episode_count):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_path = os.path.join(dir_path, f'ac_model_{episode_count}.npz')
        np.savez(file_path, w_q=self.critic.w_q, theta=self.actor.theta)

    def load_model(self, file_path):
        oc_model = np.load(file_path)
        if self._check_model(oc_model):
            self.critic.w_q = oc_model['w_q']
            self.actor.theta = oc_model['theta']
        else:
            raise Exception('Not suitable model data.')

    def _check_model(self, model):
        if model['w_q'].shape != self.critic.w_q.shape:
            return False
        if model['theta'].shape != self.actor.theta.shape:
            return False
        return True

    def summary(self):
        return {
            "td_errors": self.td_error_list
        }

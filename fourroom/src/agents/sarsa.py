import numpy as np
import logging
from fourroom.src.policies import SoftmaxPolicy
from fourroom.src.tabulars import Tabular


logger = logging.getLogger(__name__)


class SarsaAgent:
    def __init__(self, config, nfeatures, nactions, rng, q_value={}):
        logger.debug("SarsaAgent is going to perform!")
        self.discount = float(config["AGENT"]["discount"])
        self.epsilon = float(config["AGENT"]["epsilon"])
        self.lr = float(config["AGENT"]["lr"])
        self.nfeatures = nfeatures
        self.nactions = nactions
        self.temperature = float(config["AGENT"]["temperature"])
        self.q_value = q_value
        self.policy = SoftmaxPolicy(rng, nfeatures, nactions, self.temperature)
        self.critic = Sarsa(self.discount, self.lr, self.policy.weights)
        self.features = Tabular(nfeatures)
        self.total_shaped_reward = 0

        for state, value in q_value.items():
            phi = self.features(state)
            self.critic.initialize(phi, value)
        
        self.rng = np.random.RandomState(np.random.randint(0, 100))

    def seed(self, seed):
        self.rng = np.random.RandomState(seed)

    def act(self, state):
        return self.policy.sample(self.features(state))

    def update(self, state, action, next_state, reward, done, info):
        phi = self.features(state)
        next_phi = self.features(next_state)
        next_action = self.act(next_state)
        _ = self.critic.update(
            phi, action, next_phi, reward, done, next_action
        )

    def reset(self):
        self.policy = SoftmaxPolicy(self.rng, self.nfeatures, self.nactions,
                                    self.temperature)
        self.critic = Sarsa(self.discount, self.lr, self.policy.weights)
        self.features = Tabular(self.nfeatures)
        self.total_shaped_reward = 0
        for state, value in self.q_value.items():
            phi = self.features(state)
            self.critic.initialize(phi, value)

    def get_max_value(self):
        return np.amax(self.critic.weights)

    def get_min_value(self):
        return np.amin(self.critic.weights)

    def get_value(self, state):
        phi = self.features(state)
        return max(self.critic.value(phi))

    def info(self, state):
        return {
            "state": state,
            "v": self.get_value(state)
        }


class Sarsa:
    def __init__(self, discount, lr, weights):
        self.lr = lr
        self.discount = discount
        self.weights = weights

    def initialize(self, phi, q_value, action=None):
        if action is None:
            self.weights[phi, :] = q_value
        else:
            self.weights[phi, action] = q_value

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def advantage(self, phi, action=None):
        values = self.value(phi)
        advantages = values - np.max(values)
        if action is None:
            return advantages
        return advantages[action]

    def update(self, phi, action, next_phi, reward, done, next_action):
        # One-step update target
        update_target = reward
        if not done:
            next_values = self.value(next_phi)
            update_target += self.discount * next_values[next_action]
        # Dense gradient update step
        tderror = update_target - self.value(phi, action)
        self.weights[phi, action] += self.lr * tderror
        return update_target

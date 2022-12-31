import logging
import numpy as np
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


class BasePolicy:
    def __init__(self):
        # 分析用
        self.exploit_count = 0
        self.explore_count = 0

    def reset_count(self):
        self.exploit_count = 0
        self.explore_count = 0


class EgreedyPolicy(BasePolicy):
    def __init__(self, rng, nfeatures, nactions, epsilon):
        super().__init__()
        self.rng = rng
        self.epsilon = epsilon
        self.weights = np.zeros((nfeatures, nactions))

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def sample(self, phi):
        if self.rng.uniform() < self.epsilon:
            self.explore_count += 1
            return int(self.rng.randint(self.weights.shape[1]))
        self.exploit_count += 1
        return int(np.argmax(self.value(phi)))

    def get_values(self, env):
        state_values = np.zeros(env.occupancy.shape)
        for state in range(len(self.weights)):
            value = np.max(self.value([state]))
            cell = env.to_cell(state)
            state_values[cell[0]][cell[1]] = value
        return state_values

    def get_policy(self, env):
        policy = np.zeros(env.occupancy.shape)
        for state in range(len(self.weights)):
            action = np.argmax(self.value([state]))
            cell = env.to_cell(state)
            policy[cell[0]][cell[1]] = action
        return policy


class SoftmaxPolicy(BasePolicy):
    def __init__(self, rng, nfeatures, nactions, temp=1.):
        super().__init__()
        self.rng = rng
        self.weights = np.zeros((nfeatures, nactions))
        self.temp = temp

    def value(self, phi, action=None):
        if action is None:
            return np.sum(self.weights[phi, :], axis=0)
        return np.sum(self.weights[phi, action], axis=0)

    def pmf(self, phi):
        v = self.value(phi)/self.temp
        return np.exp(v - logsumexp(v))

    def sample(self, phi):
        max_action = np.argmax(self.weights.shape[1])
        if np.random.rand() < 0.001:
            logger.debug(self.pmf(phi))
        actual_action = int(
            self.rng.choice(self.weights.shape[1], p=self.pmf(phi))
        )
        if max_action == actual_action:
            self.exploit_count += 1
        else:
            self.explore_count += 1
        return actual_action

    def get_values(self, env):
        state_values = np.zeros(env.occupancy.shape)
        for state in range(len(self.weights)):
            values = self.value([state])
            percents = self.pmf([state])
            value = np.dot(values, percents)
            cell = env.to_cell(state)
            state_values[cell[0]][cell[1]] = value
        return state_values

    def get_policy(self, env):
        policy = np.zeros(env.occupancy.shape)
        for state in range(len(self.weights)):
            action = np.argmax(self.value([state]))
            cell = env.to_cell(state)
            policy[cell[0]][cell[1]] = action
        return policy


class SoftmaxStepPolicy(SoftmaxPolicy):
    def __init__(self, rng, nfeatures, nactions, action_policy, temp=1.):
        super(SoftmaxStepPolicy, self).__init__(rng, nfeatures, nactions, temp)

    def sample(self, phi):
        return np.argmax(-self.weights.shape[1])


class PolicyOverOptions:
    def __init__(self, policy, initiation_sets):
        self.policy = policy
        self.initiation_sets = initiation_sets

    def value(self, phi, action=None):
        return self.policy.value(phi, action)

    def pmf(self, phi):
        return self.policy.pmf(phi)

    def sample(self, phi):
        candidates = []
        for option, initiation_set in enumerate(self.initiation_sets):
            if phi in initiation_set:
                candidates.append(option)
        pmf = self.pmf(phi)
        cand_pmf = [pmf[option] for option in candidates]
        cand_pmf = cand_pmf/sum(cand_pmf)
        return int(self.policy.rng.choice(candidates, p=cand_pmf))

    def get_values(self):
        state_values = []
        for state in range(len(self.policy.weights)):
            values = self.value(state)
            percents = self.pmf(state)
            value = np.dot(values, percents)
            state_values.append(value)
        return state_values


class FixedActionPolicy:
    def __init__(self, action, nactions):
        self.action = action
        self.probs = np.eye(nactions)[action]

    def sample(self, phi):
        return self.action

    def pmf(self, phi):
        return self.probs

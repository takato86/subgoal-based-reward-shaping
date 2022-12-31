import numpy as np
import logging
from shaper.shaping.interface import AbstractShaping


logger = logging.getLogger(__name__)


class ShapedAgent:
    def __init__(self, raw_agent, env, subgoals, config):
        self.raw_agent = raw_agent
        self.config = config
        self.reward_shaping: AbstractShaping = self._generate_shaping(env, subgoals)
        self.current_shaping = 0
        self.env = env
        self.subgoals = subgoals

    def _generate_shaping(self, env, subgoals):
        raise NotImplementedError

    def seed(self, seed):
        self.raw_agent.seed(seed)

    def update(self, state, action, next_state, reward, done, info):
        F = self.reward_shaping.step(
            state, action, reward, next_state, done, info
        )
        if np.random.rand() < 0.001:
            logger.debug("shaping reward: {}".format(F))
        self.raw_agent.update(
            state, action, next_state, reward + F, done, info
        )
        self.current_shaping = F

    def act(self, state):
        return self.raw_agent.act(state)

    def reset(self):
        self.raw_agent.reset()
        self.reward_shaping = self._generate_shaping(self.env, self.subgoals)

    def get_max_value(self):
        return self.raw_agent.get_max_value()

    def get_min_value(self):
        return self.raw_agent.get_min_value()

    def info(self, state):
        raw_agent_info = self.raw_agent.info(state)
        info = {
            "v_z": self.reward_shaping.potential(
                        self.reward_shaping.current_state
                   ),
            "F": self.current_shaping
        }
        joined_info = {**raw_agent_info, **info}
        return joined_info


from abc import ABCMeta, abstractmethod


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def transform(self, pre_obs, pre_action, r, obs, d, info):
        for step in self.steps:
            pre_obs, pre_action, r, obs, d, info = step.transform(
                pre_obs, pre_action, r, obs, d, info
            )
        return pre_obs, pre_action, r, obs, d, info


class Step(metaclass=ABCMeta):
    @abstractmethod
    def transform(self, pre_obs, pre_action, r, obs, d, info):
        pass

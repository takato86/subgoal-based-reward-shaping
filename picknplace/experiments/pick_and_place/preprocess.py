import numpy as np
from picknplace.transform.pipeline import Step


def trans_obs(obs):
    g = obs["desired_goal"]
    observation = obs["observation"]
    obs = np.hstack([observation, g])
    obs = obs.reshape([1, -1])
    return obs


class RoboticsObservationTransformer(Step):
    def transform(self, pre_obs, pre_action, r, obs, d, info):
        if pre_obs is not None:
            pre_obs = trans_obs(pre_obs)
        if obs is not None:
            obs = trans_obs(obs)
        return pre_obs, pre_action, r, obs, d, info

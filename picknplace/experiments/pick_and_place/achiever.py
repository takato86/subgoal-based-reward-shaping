import logging
import numpy as np
from shaper.achiever import AbstractAchiever

from picknplace.transform.pipeline import Step


logger = logging.getLogger(__name__)


class FetchPickAndPlaceAchiever(AbstractAchiever):
    def __init__(self, _range: float, subgoals: np.ndarray):
        """initialize

        Args:
            _range (float): the range to judge whether achieving subgoals.
            subgs (np.ndarray): subgoal numpy list.
        """
        self._range = _range
        self._subgoals = subgoals

    @property
    def subgoals(self):
        return self._subgoals

    def eval(self, obs, subgoal_idx):
        if len(self._subgoals) <= subgoal_idx:
            # すべてのサブゴールを達成していたら常にFalse
            return False

        subgoal = np.array(self._subgoals[subgoal_idx])
        # idxs = np.argwhere(subgoal == subgoal) # np.nanでない要素を取り出し
        target_v = subgoal[6:11]
        target_obs = obs[:, 6:11]
        lower_bound = target_v - self._range
        upper_bound = target_v + self._range
        idxs = np.argwhere(lower_bound == lower_bound).flatten()
        b_lower = lower_bound[idxs] <= target_obs[:, idxs]
        b_higher = target_obs[:, idxs] <= upper_bound[idxs]
        res = np.all(b_lower & b_higher)
        if res:
            logger.debug("Achieve the subgoal{}".format(subgoal_idx))
        return res


class FetchPickAndPlaceAchieverStep(Step):
    def __init__(self, _range, subgoals):
        self.achiever = FetchPickAndPlaceAchiever(_range, subgoals)
        self.subgoal_idx = 0
        self.reset = True

    def transform(self, pre_obs, pre_action, r, obs, d, info):
        if self.reset:
            self.subgoal_idx = 0
            self.reset = False

        if obs is not None and info is not None:

            if self.achiever.eval(obs, self.subgoal_idx):
                self.subgoal_idx += 1

            if d:
                self.reset = True

            info["subgoal"] = self.subgoal_idx
        return (pre_obs, pre_action, r, obs, d, info)

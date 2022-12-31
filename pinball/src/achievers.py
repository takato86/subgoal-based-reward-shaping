from shaper.achiever import AbstractAchiever
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PinballAchiever(AbstractAchiever):
    def __init__(self, _range, subgoals):
        self._range = _range
        self.__subgoals = subgoals  # 2d-ndarray shape(#obs, #subgoals)

    @property
    def subgoals(self):
        return self.__subgoals

    def eval(self, obs, subgoal_idx):
        if len(self.subgoals) <= subgoal_idx:
            return False

        subgoal = np.array(self.subgoals[subgoal_idx])
        idxs = np.argwhere(subgoal == subgoal)  # np.nanでない要素を取り出し
        b_in = l2_norm_dist(
            subgoal[idxs].reshape(-1),
            obs[idxs].reshape(-1)
        ) <= self._range
        res = np.all(b_in)

        if res:
            logger.debug("Achieve the subgoal{}".format(subgoal_idx))
        return res


def l2_norm_dist(x_arr, y_arr):
    return np.linalg.norm(x_arr - y_arr, ord=2)

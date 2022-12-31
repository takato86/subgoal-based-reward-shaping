from typing import List, Tuple
import numpy as np
from shaper.achiever import AbstractAchiever
from shaper.transiter import AbstractTransiter


class RoomsAchiever(AbstractAchiever):
    def __init__(self, _range, n_obs, subgoals, **params):
        self.__subgoals = subgoals[0]

    @property
    def subgoals(self):
        return self.__subgoals

    def eval(self, obs, current_state):
        if len(self.__subgoals) <= current_state:
            return False
        subgoal = self.__subgoals[current_state]
        return obs == subgoal


class RoomsTransiter(AbstractTransiter[int, int]):
    def __init__(self, _range: float, subgoals: List[Tuple[int]]):
        self.range = _range
        # 抽象状態空間の作成
        subgs = []

        for prev_sg, next_sg in subgoals:
            subgs.append(prev_sg), subgs.append(next_sg)

        subg_set = list(dict.fromkeys(subgs))
        self.n_subs = len(subg_set)
        self.subg2abs = {
            subg: i+1 for i, subg in enumerate(subg_set)
        }
        # 抽象状態における次に達成すべきサブゴールリストの作成
        self.next_subgoals = {
            i: [] for i in range(self.n_subs + 1)
        }

        after_transitions = set()

        for prev_sg, next_sg in subgoals:
            self.next_subgoals[self.subg2abs[prev_sg]].append(next_sg)
            after_transitions.add(next_sg)
        
        # どこからも遷移されないサブゴールは抽象状態0の次の達成すべきサブゴールとして登録する。
        self.next_subgoals[0] = []
        for prev_sg, _ in subgoals:
            if prev_sg not in after_transitions:
                self.next_subgoals[0].append(prev_sg)

    def reset(self) -> int:
        return 0

    def transit(self, obs: int, subgoal_idx: int) -> int:
        next_subgoals = self.next_subgoals[subgoal_idx]
        
        if obs in next_subgoals:
            return self.subg2abs[obs]

        return subgoal_idx

    @property
    def n_states(self) -> Tuple[int]:
        return self.n_subs + 1

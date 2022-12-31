import shaper
from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation
from src.agents.dta import is_success
from src.agents.shaped import ShapedAgent
from src.achievers import PinballAchiever


class NaiveRSAgent(ShapedAgent):
    def _create_reward_shaping(self, env):
        achiever = PinballAchiever(
                self.config["shaping"]["range"],
                self.subgoals
            )
        aggregator = DynamicTrajectoryAggregation(achiever, is_success)
        return shaper.NaiveSRS(
            self.config["agent"]["gamma"],
            self.config["shaping"]["eta"],
            aggregator
        )


class LinearNaiveRSAgent(ShapedAgent):
    def _create_reward_shaping(self, env):
        achiever = PinballAchiever(
            self.config["shaping"]["range"],
            self.subgoals
        )
        aggregator = DynamicTrajectoryAggregation(achiever, is_success)
        return shaper.LinearNaiveSRS(
            self.config["agent"]["gamma"],
            self.config["shaping"]["eta"],
            aggregator
        )

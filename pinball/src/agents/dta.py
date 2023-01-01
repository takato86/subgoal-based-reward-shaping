import shaper
from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation
from pinball.src.achievers import PinballAchiever
from pinball.src.agents.shaped import ShapedAgent


class DTAAgent(ShapedAgent):
    def _create_reward_shaping(self, env):
        achiever = PinballAchiever(
            self.config["shaping"]["range"],
            self.subgoals
        )
        aggregator = DynamicTrajectoryAggregation(achiever, is_success)
        vfunc = aggregator.create_vfunc()
        return shaper.SarsaRS(
            self.config["agent"]["gamma"],
            self.config["shaping"]["lr"],
            aggregator, vfunc, is_success
        )


def is_success(done, info):
    return done

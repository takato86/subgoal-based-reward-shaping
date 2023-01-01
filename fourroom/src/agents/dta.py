import shaper
from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation, DynamicStateAggregation
from fourroom.src.agents.shaped import ShapedAgent
from fourroom.src.achievers import RoomsAchiever, RoomsTransiter


class DTAAgent(ShapedAgent):
    def __init__(self, raw_agent, env, subgoals, config):
        super().__init__(
            raw_agent, env, subgoals, config
        )

    def _generate_shaping(self, env, subgoals):
        nfeatures = env.observation_space.n
        aggregator = DynamicTrajectoryAggregation(
            RoomsAchiever(
                float(self.config["SHAPING"]["_range"]),
                nfeatures, subgoals
            ), is_success
        )
        vfunc = aggregator.create_vfunc()
        return shaper.SarsaRS(
            float(self.config["AGENT"]["discount"]),
            float(self.config["SHAPING"]["lr"]),
            aggregator, vfunc,
            is_success
        )

    def info(self, state):
        super_info = super().info(state)
        info = {
            "z": self.reward_shaping.current_state
        }
        joined_info = {**super_info, **info}
        return joined_info


class PartiallyDTAAgent(ShapedAgent):
    def __init__(self, raw_agent, env, subgoals, config):
        super().__init__(raw_agent, env, subgoals, config)

    def _generate_shaping(self, env, subgoals):
        transiter = RoomsTransiter(float(self.config["SHAPING"]["_range"]), subgoals)
        aggregator = DynamicStateAggregation(transiter, is_success)
        vfunc = aggregator.create_vfunc()
        return shaper.SarsaRS(
            float(self.config["AGENT"]["discount"]),
            float(self.config["SHAPING"]["lr"]),
            aggregator, vfunc,
            is_success
        )


def is_success(done, info):
    return done

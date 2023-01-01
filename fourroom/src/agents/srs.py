import logging
import shaper
from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation, DynamicStateAggregation
from fourroom.src.agents.shaped import ShapedAgent
from fourroom.src.achievers import RoomsAchiever, RoomsTransiter
from fourroom.src.agents.dta import is_success

logger = logging.getLogger(__name__)


class SRSAgent(ShapedAgent):
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
            ),
            is_success
        )
        return shaper.SubgoalRS(
            float(self.config["AGENT"]["discount"]),
            float(self.config["SHAPING"]["eta"]),
            aggregator
        )


class PartiallySRSAgent(ShapedAgent):
    def __init__(self, raw_agent, env, subgoals, config):
        super().__init__(
            raw_agent, env, subgoals, config
        )

    def _generate_shaping(self, env, subgoals):
        transiter = RoomsTransiter(float(self.config["SHAPING"]["_range"]), subgoals)
        aggregator = DynamicStateAggregation(transiter, is_success)
        return shaper.SubgoalRS(
            float(self.config["AGENT"]["discount"]),
            float(self.config["SHAPING"]["eta"]),
            aggregator
        )

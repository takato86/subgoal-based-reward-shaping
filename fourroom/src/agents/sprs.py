import logging
import shaper
from fourroom.src.agents.dta import is_success
from fourroom.src.agents.shaped import ShapedAgent
from fourroom.src.achievers import RoomsAchiever

logger = logging.getLogger(__name__)


class SPRSAgent(ShapedAgent):
    def __init__(self, raw_agent, env, subgoals, config):
        super().__init__(
            raw_agent, env, subgoals, config
        )

    def _generate_shaping(self, env, subgoals):
        nfeatures = env.observation_space.n
        return shaper.SubgoalPulseRS(
            float(self.config["AGENT"]["discount"]),
            RoomsAchiever(
                float(self.config["SHAPING"]["_range"]),
                nfeatures, subgoals
            ),
            is_success
        )

    def update(self, state, action, next_state, reward, done, info):
        next_state_value = self.raw_agent.get_value(next_state)
        # 価値関数のセット
        self.reward_shaping.set_value(next_state_value)
        super().update(state, action, next_state, reward, done, info)

    def info(self, state):
        # TODO potentialの取得をp_potentialの値に
        raw_agent_info = self.raw_agent.info(state)
        info = {
            "potential": self.reward_shaping.p_potential
        }
        joined_info = {**raw_agent_info, **info}
        return joined_info

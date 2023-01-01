import shaper
from pinball.src.agents.shaped import ShapedAgent
from pinball.src.achievers import PinballAchiever


class SPRSAgent(ShapedAgent):
    def _create_reward_shaping(self, env):
        return shaper.SubgoalPulseRS(
            self.config["agent"]["gamma"],
            self.raw_agent.get_max_q,
            PinballAchiever(
                self.config["shaping"]["range"],
                env.observation_space.shape[0],
                self.subgoals
            )
        )

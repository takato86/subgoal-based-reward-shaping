

class ShapedAgent:
    def __init__(self, raw_agent, env, config, subgoals):
        self.raw_agent = raw_agent
        self.subgoals = subgoals
        self.config = config
        self.reward_shaping = self._create_reward_shaping(env)

    def _create_reward_shaping(self, env, config):
        raise NotImplementedError

    def act(self, observation):
        return self.raw_agent.act(observation)

    def update(self, pre_obs, pre_a, r, obs, a, done, info):
        f = self.reward_shaping.step(pre_obs, pre_a, r, obs, done, info)
        if done:
            self.reward_shaping.reset()
        self.raw_agent.update(pre_obs, pre_a, r+f, obs, a, done, info)
        return r + f

    def get_max_q(self, obs):
        return self.raw_agent.get_max_q(obs)

    def get_max_td_error(self):
        return self.raw_agent.get_max_td_error()

    def save_model(self, dir_path, episode_count):
        self.raw_agent.save_model(dir_path, episode_count)

    def load_model(self, file_path):
        self.raw_agent(file_path)

    def summary(self):
        return self.raw_agent.summary()

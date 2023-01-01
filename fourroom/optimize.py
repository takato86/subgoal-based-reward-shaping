import gym
import gym_fourrooms
import hydra
from hydra.utils import get_original_cwd
import logging
import numpy as np
from omegaconf import DictConfig
import optuna
import os
from statistics import mean

from main import load_subgoals
from fourroom.src.agents.factory import create_agent
from fourroom.src.agents.hypara_opts import HYPERPARAMS_SAMPLER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


@hydra.main(config_path="in/configs", config_name="hypara_opts")
def main(cfg: DictConfig):
    study = optuna.create_study()

    def objective(trial):
        hypara_config = HYPERPARAMS_SAMPLER[cfg.agent.name](trial)
        subgoal_path = os.path.join(
            get_original_cwd(), cfg.shaping.subgoal_path
        )
        subgoals = load_subgoals(subgoal_path, task_id=1)
        cum_steps = 0
        cum_rewards = []

        rng = np.random.RandomState(1)
        env = gym.make(cfg.env.env_id)
        agent = create_agent(hypara_config, env, rng, subgoals[0])
        agent.reset()

        for episode in range(cfg.common.n_episodes):
            cum_reward = 0
            next_observation = env.reset()

            for _ in range(cfg.common.n_steps):
                cum_steps += 1
                observation = next_observation
                action = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                # Critic update
                agent.update(
                    observation, action, next_observation, reward, done, info
                )
                cum_reward += reward
                if done:
                    break

            cum_rewards.append(cum_reward)

        return mean(cum_rewards)

    study.optimize(objective, n_trials=100)
    best_params = study.best_params
    print(best_params)


if __name__ == "__main__":
    main()

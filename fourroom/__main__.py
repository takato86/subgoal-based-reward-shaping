from datetime import datetime
import gym
from gym import wrappers
import numpy as np
import pandas as pd
import gym_fourrooms
import logging
import os
import shutil
import json
import time
import argparse
from tqdm import tqdm
from src.agents.factory import create_agent
from src.config import config
from concurrent.futures import ProcessPoolExecutor
from pyvirtualdisplay import Display


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
gym_fourrooms


def export_steps(file_path, steps):
    with open(file_path, 'w') as f:
        f.write('\n'.join(list(map(str, steps))))


def export_runtimes(file_path, runtimes):
    runtimes_df = pd.DataFrame(runtimes, columns=["runtime"])
    runtimes_df.to_csv(file_path)


def export_details(file_path, details):
    detail_df = pd.DataFrame.from_dict(details, orient="index")
    detail_df.to_csv(file_path)


def prep_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def load_subgoals(file_path, task_id=None):
    logger.info(f"Loading {file_path}")
    subgoal_df = pd.read_csv(file_path)
    if task_id is not None:
        subgoal_df = subgoal_df[subgoal_df["task_id"] == task_id]
    subg_serieses_df = subgoal_df.groupby(["user_id", "task_id"]).agg(list)
    subg_serieses = []
    for subg_series in list(subg_serieses_df['state'].values):
        subg_serieses.append([subg_series])
    return subg_serieses


def load_partially_subgoals(file_path):
    logger.info(f"Loading partially subgoals: {file_path}")
    subgoal_df = pd.read_csv(file_path)
    subgs = subgoal_df.groupby("user_id").agg(list).values.tolist()
    all_subg_tuples = []

    for prev_subgs, next_subgs in subgs:
        subg_tuples = [(prev_sg, next_sg) for prev_sg, next_sg in zip(prev_subgs, next_subgs)]
        all_subg_tuples.append(subg_tuples)
    
    return all_subg_tuples


def learning_loop(env, subgoal, nruns, nepisodes, nsteps,
                  id, env_id, nprocess, out_dir):
    runtimes = []
    args = []
    for run in range(nruns):
        rng = np.random.RandomState((id + 1) * (run + 1))
        agent = create_agent(config, env, rng, subgoal)
        env.seed(run)
        args.append(
            [run, env, agent, nepisodes, nsteps, id, env_id, out_dir]
        )
    with ProcessPoolExecutor(max_workers=nprocess) as executor:
        ret = tqdm(executor.map(run_loop, args), total=nruns)
    runtimes = list(ret)

    # single process
    # runtimes = []
    # for arg in args:
    #     runtimes.append(
    #         run_loop(arg)
    #     )

    runtimes_dir = os.path.join(out_dir, "runtime")
    export_runtimes(
        os.path.join(
            runtimes_dir,
            f"{env_id}-{id}.csv"
        ),
        runtimes
    )


def run_loop(args):
    """The multiprocessed method."""
    run, env, agent, nepisodes, nsteps, id, env_id, out_dir = args
    steps_dir = os.path.join(out_dir, "steps")
    detail_dir = os.path.join(out_dir, "detail")
    start_time = time.time()
    agent.reset()
    steps = []
    runtimes = []
    details = {}
    for episode in tqdm(range(nepisodes), leave=False):
        next_observation = env.reset()
        logger.debug(f"start: {next_observation}, goal: {env.goal}")
        cumreward = 0
        for step in range(nsteps):
            observation = next_observation
            action = agent.act(observation)
            next_observation, reward, done, info = env.step(action)
            # Critic update
            agent.update(
                observation, action, next_observation, reward, done, info
            )
            detail = agent.info(next_observation)
            if detail:
                ind = episode * nepisodes + step
                detail["episode"] = episode
                detail["step"] = step
                detail["reward"] = reward
                details[ind] = detail
            cumreward += reward
            if done:
                break
        steps.append(step)
        logger.debug(
            'Run {} episode {} steps {}, cumreward {:.4f}, min_Q: {:.4f},'
            ' max_Q: {:.4f}'.format(
                run, episode, step, cumreward,
                agent.get_min_value(), agent.get_max_value()
            )
        )
    runtimes.append(time.time() - start_time)
    export_steps(
        os.path.join(
            steps_dir,
            f"{env_id}-{run}-{id}.csv"
        ),
        steps
    )
    export_details(
        os.path.join(
            detail_dir,
            f"{env_id}-{run}-{id}.csv"
        ),
        details
    )
    return runtimes


def main(out_dir):
    logger.info(
        "env: {}, alg: {}".format(
            config["ENV"]["env_id"], config["AGENT"]["name"]
        )
    )
    env_to_wrap = gym.make(config["ENV"]["env_id"])

    if config.getboolean("ENV", "video"):
        movie_folder = prep_dir(
            os.path.join(
                out_dir, 'movies'
            )
        )
        env = wrappers.Monitor(
            env_to_wrap, movie_folder, force=True,
            video_callable=(lambda ep: ep % 100 == 0 or (ep > 30 and ep < 35)))
    else:
        env = env_to_wrap

    subgoals = [[]]

    if "SHAPING" not in config.sections():
        pass
    elif config.get("SHAPING", "subgoal_path") is not None:

        if config["AGENT"]["name"] in ["PartiallyDTAAgent", "PartiallySRSAgent"]:
            subgoals = load_partially_subgoals(config["SHAPING"]["subgoal_path"])
        else:
            subgoals = load_subgoals(config["SHAPING"]["subgoal_path"], task_id=1)

        logger.info(f"subgoals: {subgoals}")
    elif config.get("SHAPING", "mapping_path") is not None:
        json_open = open(config["SHAPING"]["mapping_path"])
        aggr_set = [l for _, l in json.load(json_open).items()]
        logger.info(f"mappings: {aggr_set}")

    logger.info("Start learning")

    for learn_id, subgoal in enumerate(subgoals):
        logger.info(f"Subgoal {learn_id+1}/{len(subgoals)}: {subgoal}")
        # elif "sarsa-rs" in config["SHAPING"]["alg"]:
        #     logger.debug("SarsaRSAgent")
        #     agent = SarsaRSSarsaAgent(
        #         float(config["AGENT"]["discount"]),
        #         float(config["AGENT"]["epsilon"]),
        #         float(config["AGENT"]["lr"]), nfeatures, nactions,
        #         float(config["AGENT"]["temperature"]), rng, aggr_set)

        learning_loop(
            env, subgoal,
            int(config["AGENT"]["nruns"]),
            int(config["AGENT"]["nepisodes"]),
            int(config["AGENT"]["nsteps"]),
            learn_id,
            config["ENV"]["env_id"],
            int(config["ENV"]["nprocess"]),
            out_dir
        )
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config.read(args.config)
    config_fname = os.path.basename(args.config)
    now = datetime.now()
    out_dir = prep_dir(
        config["SETTING"]["out_dir"] + "_" + datetime.strftime(now,
                                                               "%Y%m%d_%H%M")
    )
    shutil.copy(
        args.config,
        os.path.join(out_dir, config_fname)
    )
    steps_dir = prep_dir(os.path.join(out_dir, "steps"))
    runtimes_dir = prep_dir(os.path.join(out_dir, "runtime"))
    detail_dir = prep_dir(os.path.join(out_dir, "detail"))

    with Display():
        main(out_dir)

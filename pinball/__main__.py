import argparse
import time
import gym
import os
import shutil
import json
import logging
from gym import wrappers
import numpy as np
import pandas as pd
import itertools
import gym_pinball
from tqdm import tqdm
from src.agents.factory import create_agent
from visualizer import Visualizer
from concurrent.futures import ProcessPoolExecutor
from pyvirtualdisplay import Display

gym_pinball
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_TYPES = [
    'total_reward',
    'td_error',
    'steps', 'runtime'
]
ALG_CHOICES = [
    "subgoal",
    "naive",
    "actor-critic",
    "sarsa-rs",
    "srs"
]


def export_csv(file_path, file_name, array):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    array = pd.DataFrame(array)
    saved_path = os.path.join(file_path, file_name)
    array.to_csv(saved_path)


def moved_average(data, window_size):
    b = np.ones(window_size)/window_size
    return np.convolve(data, b, mode='same')


def load_subgoals(file_path):
    logger.info(f"Loading {file_path}")
    subgoals_df = pd.read_csv(file_path)
    subgoals = subgoals_df.groupby(["user_id", "task_id"]).agg(list)
    xs = subgoals["x"].values.tolist()
    ys = subgoals["y"].values.tolist()
    rads = subgoals["rad"].values.tolist()
    subg_serieses = []
    for x, y, rad in zip(xs, ys, rads):
        subg_series = []
        for x_i, y_i, rad_i in zip(x, y, rad):
            subg_series.append({
                "pos_x": x_i,
                "pos_y": y_i,
                "rad": rad_i
                })
        subg_serieses.append([subg_series])
    return subg_serieses


def load_subgoals_new(file_path):
    logger.info(f"Loading {file_path}")
    subgoals_df = pd.read_csv(file_path)
    subgoals = subgoals_df.groupby(["user_id", "task_id"]).agg(list)
    xs = subgoals["x"].values.tolist()
    ys = subgoals["y"].values.tolist()
    rads = subgoals["rad"].values.tolist()
    subg_serieses = []
    for x, y, rad in zip(xs, ys, rads):
        subg_series = []
        for x_i, y_i, _ in zip(x, y, rad):
            subg_series.append(
                np.array([
                    x_i, y_i, np.nan, np.nan
                ])
            )
        subg_serieses.append(subg_series)
    return subg_serieses


def get_file_names(exe_id, l_id, run, eta=None, rho=None, k=None):
    fnames = {}
    for _type in OUTPUT_TYPES:
        fname = f"{exe_id}_{_type}_{l_id}_{run}"
        if eta is not None:
            fname += "_eta={}".format(eta)
        if rho is not None:
            fname += "_rho={}".format(rho)
        if k is not None:
            fname += "_k={}".format(k)
        fname += ".csv"
        fnames[_type] = fname
    return fnames


def learning_loop(args):
    run, config, subgoals, l_id = args
    logger.debug(f"start run {run}")
    subg_confs = list(itertools.chain.from_iterable(subgoals))
    config["setting"]["seed"] = run
    env = gym.make(config["env"]["id"], subg_confs=subg_confs)
    env.seed(config["setting"]["seed"])
    env = wrappers.Monitor(env, directory=d_kinds["mv"], force=True)
    agent = create_agent(config, env, subgoals)
    fnames = get_file_names(config["agent"]["name"], l_id, run)

    vis = Visualizer(["ACC_X", "ACC_Y", "DEC_X", "DEC_Y", "NONE"])

    if config["env"]["model"]:
        agent.load_model(config["env"]["model"])

    episode_count = config["setting"]["nepisodes"]
    visual = config["setting"]["visual"]
    reward = 0
    max_q = 0.0
    done = False
    total_reward_list, steps_list, max_q_list = [], [], []
    max_q_episode_list, runtimes = [], []
    start_time = time.time()

    for episode in range(episode_count):
        total_reward = 0
        total_shaped_reward = 0
        n_steps = 0
        ob = env.reset()
        action = agent.act(ob)
        pre_action = action
        is_render = False
        while True:
            if (episode+1) % 20 == 0 and visual:
                env.render()
                is_render = True
            pre_obs = ob
            ob, reward, done, info = env.step(action)
            # TODO
            reward = 0 if reward < 0 else reward
            n_steps += 1
            # rand_basis = np.random.uniform()
            pre_action = action
            action = agent.act(ob)
            shaped_reward = agent.update(
                pre_obs, pre_action, reward, ob, action, done, info
            )
            total_reward += reward
            total_shaped_reward += shaped_reward
            tmp_max_q = agent.get_max_q(ob)
            max_q_list.append(tmp_max_q)
            max_q = tmp_max_q if tmp_max_q > max_q else max_q

            if done:
                logger.debug(
                    "episode: {}, steps: {}, total_reward: {}, "
                    "total_shaped_reward: {}, max_q: {}, "
                    "max_td_error: {}".format(
                        episode, n_steps, total_reward,
                        int(total_shaped_reward), int(max_q), int(agent.get_max_td_error())
                    )
                )
                total_reward_list.append(total_reward)
                steps_list.append(n_steps)
                break

            if is_render:
                vis.set_action_dist(agent.vis_action_dist, action)
                vis.pause(.0001)
        max_q_episode_list.append(max_q)
        agent.save_model(d_kinds["mo"], episode)
    # export process
    runtimes.append(time.time() - start_time)
    export_csv(
        d_kinds['tr'],
        fnames['total_reward'],
        total_reward_list
    )
    summary = agent.summary()
    export_csv(
        d_kinds["td"],
        fnames['td_error'],
        summary["td_errors"]
    )
    total_reward_list = np.array(total_reward_list)
    steps_list = np.array(steps_list)
    max_q_list = np.array(max_q_list)
    logger.debug("Average return: {}".format(np.average(total_reward_list)))
    steps_file_path = os.path.join(d_kinds["st"], fnames['steps'])
    pd.DataFrame(steps_list).to_csv(steps_file_path)
    runtime_file_path = os.path.join(d_kinds["ru"], fnames['runtime'])
    pd.DataFrame(runtimes, columns=["runtime"]).to_csv(runtime_file_path)
    env.close()


def main():
    logger.info("ENV: {}".format(config["env"]["id"]))
    learning_time = time.time()
    if config["shaping"].get("subg_path") is not None:
        subg_serieses = load_subgoals_new(config["shaping"]["subg_path"])
    else:
        logger.debug("Nothing subgoal path.")
        subg_serieses = [[[{"pos_x": 0.512, "pos_y": 0.682, "rad": 0.04}, {"pos_x": 0.683, "pos_y": 0.296, "rad": 0.04}]]] # , {"pos_x":0.9 , "pos_y":0.2 ,"rad": 0.04}

    for l_id, subg_series in enumerate(subg_serieses):
        logger.debug(f"learning: {l_id+1}/{len(subg_serieses)}")
        logger.debug(f"subgoals: {subg_series}")
        arguments = [
            [
                run, config, subg_series, l_id
            ]
            for run in range(config["setting"]["nruns"])
        ]

        if config["setting"]["parallel"]:
            with ProcessPoolExecutor(max_workers=config["setting"]["nprocesses"]) as executor:
                tqdm(
                    executor.map(learning_loop, arguments),
                    total=config["setting"]["nruns"]
                )
        else:
            # Single Process
            for run in tqdm(range(config["setting"]["nruns"])):
                logger.debug(
                    "Run: {}/{}".format(
                        run+1, config["setting"]["nruns"]
                    )
                )
                learning_loop(arguments[run])
        #     # Close the env and write monitor result info to disk

    duration = time.time() - learning_time
    logger.info("Learning time: {}m {}s".format(
        int(duration//60), int(duration % 60)
    ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Actor-Critic Learning.')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    config_fname = os.path.basename(args.config)
    saved_dir = config["setting"]["out_dir"]
    d_kinds = {
        "tr": os.path.join(saved_dir, "total_reward"),
        "st": os.path.join(saved_dir, "steps"),
        "td": os.path.join(saved_dir, "td_error"),
        "mo": os.path.join(saved_dir, "model"),
        "ru": os.path.join(saved_dir, "runtime"),
        "mv": os.path.join(saved_dir, "movie")
    }

    for fpath in d_kinds.values():
        if not os.path.exists(fpath):
            os.makedirs(fpath)

    shutil.copy(args.config, os.path.join(saved_dir, config_fname))

    with Display() as disp:
        main()
    

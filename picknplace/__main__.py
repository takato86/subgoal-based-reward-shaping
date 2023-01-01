import argparse
import logging
import gym
import gym_m2s
import torch
import numpy as np
from picknplace.torch import TRAIN_FNS, OPTIMIZE_FNS
from picknplace.utils.config import load_config
import random


gym_m2s


def main(configs):
    # torch.backends.cudnn.benchmark = True
    env_name = configs["env_id"]
    env_params = configs["env_params"]
    # Fix seed
    seed = configs["seed"]
    alg = configs["alg"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    def create_env():
        env = gym.make(env_name, **env_params)
        env.seed(seed)
        return env

    env_fn = create_env

    if alg not in TRAIN_FNS or alg not in OPTIMIZE_FNS:
        logger.error(f"Not implement alg: {alg}")
        raise NotImplementedError

    logger.info("START")
    if args.optimize:
        OPTIMIZE_FNS[alg](env_fn, configs)
    else:
        TRAIN_FNS[alg](env_fn, configs)
    logger.info("END")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", "-o", action='store_true')
    parser.add_argument("--debug", "-d", action='store_true', default=False)
    parser.add_argument("--config", "-c", type=str, default="configs/config.json")
    args = parser.parse_args()
    configs = load_config(args.config)
    configs["debug"] = args.debug
    level = logging.DEBUG

    if not args.debug:
        # ignore FutureWarning not in debugging.
        import warnings
        warnings.simplefilter('ignore', FutureWarning)
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(filename)s %(funcName)s() L%(lineno)d: %(message)s",
        filename="out.log"
    )
    logger = logging.getLogger()

    main(configs)

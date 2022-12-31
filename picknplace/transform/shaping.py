import logging
import shaper
from shaper.aggregator.subgoal_based import DynamicTrajectoryAggregation
from pdrl.transform.pipeline import Step
from pdrl.experiments.pick_and_place.subgoal import subgoal_generator_factory
from pdrl.experiments.pick_and_place.is_success import is_success, is_success4sarsa_rs, is_success4dta
from pdrl.experiments.pick_and_place.achiever import FetchPickAndPlaceAchiever


logger = logging.getLogger()


def create_aggregator(configs):
    subgoals = subgoal_generator_factory[configs["subgoal_type"]]()
    achiever_params = configs["achiever_params"]
    achiever = FetchPickAndPlaceAchiever(
        subgoals=subgoals,
        **achiever_params
    )
    aggregator = DynamicTrajectoryAggregation(achiever, is_success4dta)
    return aggregator


def create_dta(configs):
    shaping_configs = configs["shaping_params"]
    gamma = shaping_configs["gamma"]
    lr = shaping_configs["lr"]
    aggregator = create_aggregator(configs)
    vfunc = aggregator.create_vfunc(shaping_configs.get("values"))
    return shaper.SarsaRS(gamma, lr, aggregator, vfunc, is_success4sarsa_rs)


def create_nrs(configs):
    shaping_configs = configs["shaping_params"]
    gamma = shaping_configs["gamma"]
    eta = shaping_configs["eta"]
    aggregator = create_aggregator(configs)
    return shaper.NaiveSRS(gamma, eta, aggregator)


def create_static(configs):
    shaping_configs = configs["shaping_params"]
    gamma = shaping_configs["gamma"]
    eta = shaping_configs["eta"]
    aggregator = create_aggregator(configs)
    return shaper.SubgoalRS(gamma, eta, aggregator)


def create_linrs(configs):
    shaping_configs = configs["shaping_params"]
    gamma = shaping_configs["gamma"]
    eta = shaping_configs["eta"]
    aggregator = create_aggregator(configs)
    return shaper.LinearNaiveSRS(gamma, eta, aggregator)


SHAPING_ALGS = {
    "dta": create_dta,
    "nrs": create_nrs,
    "static": create_static,
    "linrs": create_linrs
}


def create_shaper(configs, env_fn):
    shaping_method = configs.get("shaping_method")
    if shaping_method is not None:
        shaper = SHAPING_ALGS[shaping_method](configs)
        logger.info(f"{type(shaper)} is selected.")
        return shaper
    else:
        logger.info("shaping method is not selected.")
        return None


class ShapingStep(Step):
    """shaping class for usage in pipeline

    Args:
        Step (_type_): _description_
    """
    def __init__(self, shaper):
        self.shaper = shaper

    def transform(self, pre_obs, pre_action, r, obs, d, info):
        is_none = [
            x is None for x in [pre_obs, pre_action, r, obs, d]
        ]
        if not any(is_none):
            # Noneが一つでもある場合はスキップ。
            f = self.shaper.step(pre_obs, pre_action, r, obs, d, info)
            r += f

        return pre_obs, pre_action, r, obs, d, info

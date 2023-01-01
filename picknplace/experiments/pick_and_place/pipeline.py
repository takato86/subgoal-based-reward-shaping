from picknplace.experiments.pick_and_place.achiever import FetchPickAndPlaceAchieverStep
from picknplace.experiments.pick_and_place.subgoal import subgoal_generator_factory
from picknplace.transform.pipeline import Pipeline
from picknplace.experiments.pick_and_place.preprocess import RoboticsObservationTransformer


def create_test_pipeline(configs):
    subgoals = subgoal_generator_factory[configs["subgoal_type"]]()
    return Pipeline(
        [
            RoboticsObservationTransformer(),
            FetchPickAndPlaceAchieverStep(configs["achiever_params"]["_range"], subgoals)
        ]
    )


def create_pipeline(configs):
    subgoals = subgoal_generator_factory[configs["subgoal_type"]]()
    pipe = Pipeline(
        [
            RoboticsObservationTransformer(),
            FetchPickAndPlaceAchieverStep(configs["achiever_params"]["_range"], subgoals)
        ]
    )
    return pipe

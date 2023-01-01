import logging
import optuna
from picknplace.experiments.pick_and_place.pipeline import create_pipeline, create_test_pipeline
from picknplace.experiments.pick_and_place.sampler import sample_shaping_params
from picknplace.torch.td3.learn import learn
import mlflow
from datetime import datetime
from picknplace.torch.ddpg.replay_memory import create_replay_buffer_fn
from picknplace.transform.shaping import create_shaper


logger = logging.getLogger(__name__)


def optimize_hyparams(env_fn, configs):
    logger.info("OPTIMIZE HYPAPER PARAMETERS.")
    experiment_id = mlflow.create_experiment("DDPG Hyperparameter Tuning Experiment-{}".format(datetime.now()))
    n_trials = configs["hypara_optimization_params"]["n_trials"]
    max_ep_len = configs["training_params"]["max_ep_len"]
    shaping_method = configs.get("shaping_method")

    def objective(trial):
        mlflow.start_run(experiment_id=experiment_id)
        # max_gamma = 1. - 1. / max_ep_len
        # min_gamma = 1. - 5. / max_ep_len
        gamma = trial.suggest_categorical("gamma", [0.9, 0.99, 0.999, 0.9999, 0.95, 0.995, 0.98])
        logged_params = {
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "epsilon": trial.suggest_categorical("epsilon", [0.1, 0.2, 0.3, 0.4, 0.5]),
            "gamma": gamma,
            "actor_lr": trial.suggest_categorical("actor_lr", [0.001, 0.0001, 0.005, 0.01, 0.00001]),
            "critic_lr": trial.suggest_categorical("critic_lr", [0.001, 0.0001, 0.005, 0.01, 0.00001]),
            "replay_size": trial.suggest_categorical("replay_size", [int(1E6), int(1E5), int(1E4)]),
            "polyak": trial.suggest_categorical("polyak", [0.95, 0.98, 0.92]),
            "l2_action": trial.suggest_categorical("l2_action", [0.5, 0.8, 1.0, 1.2, 1.5]),
            "noise_scale": trial.suggest_categorical("noise_scale", [0.2, 0.3, 0.1, 0.4, 0.01]),
        }
        trial.set_user_attr("GAMMA", gamma)
        shaping_hyperparams = sample_shaping_params(trial, shaping_method)
        mlflow.log_params({**logged_params, **shaping_hyperparams})
        updated_configs = configs.copy()
        updated_configs["shaping_params"] = shaping_hyperparams
        pipeline = create_pipeline(updated_configs)
        test_pipeline = create_test_pipeline(updated_configs)
        shaper = create_shaper(updated_configs, env_fn)
        # Note that the learn method does not need the replay size argument.
        replay_buffer_fn = create_replay_buffer_fn(shaper, logged_params.pop("replay_size"))
        params = {
            "epochs": configs["training_params"]["epochs"],
            "steps_per_epoch": configs["training_params"]["steps_per_epoch"],
            "start_steps": configs["training_params"]["start_steps"],
            "update_after": configs["training_params"]["update_after"],
            "update_every": configs["training_params"]["update_every"],
            "num_test_episodes": configs["eval_params"]["num_test_episodes"],
            "max_ep_len": max_ep_len,
            "env_fn": env_fn,
            "pipeline": pipeline,
            "test_pipeline": test_pipeline,
            "replay_buffer_fn": replay_buffer_fn,
            "logdir": "picknplace/runs/optimize/" + str(datetime.now()),
            "norm_clip": configs["agent_params"]["norm_clip"],
            "norm_eps": configs["agent_params"]["norm_eps"],
            "clip_return": configs["agent_params"]["clip_return"],
            "is_pos_return": configs["agent_params"]["is_pos_return"],
            "video": False,
            **logged_params
        }
        perf = learn(**params)
        mlflow.log_metric("total_test_return", perf)
        mlflow.end_run()
        return perf

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    logger.info(best_params)
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.show()

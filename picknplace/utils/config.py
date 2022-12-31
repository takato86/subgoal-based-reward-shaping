import json

configs = None


def load_config(config_fpath):
    global configs

    with open(config_fpath, "r") as f:
        configs = json.load(f)
        # log_artifact("config.json")

    return configs


def export_config(config_fpath):
    global configs

    with open(config_fpath, "w") as f:
        json.dump(configs, f)

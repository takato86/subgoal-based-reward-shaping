

def create_sarsa_hyparams(trial):
    discount = trial.suggest_categorical(
        "discount", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    epsilon = trial.suggest_categorical(
        "epsilon", [1e-1, 5e-2, 1e-2, 1e-3, 1e-4]
    )
    lr = trial.suggest_categorical(
        "lr", [1e-1, 5e-2, 1e-2, 1e-3, 1e-4]
    )
    temperature = trial.suggest_categorical(
        "temperature", [1e-1, 5e-2, 1e-2, 1e-3, 1e-4]
    )
    return {
        "AGENT": {
            "name": "SarsaAgent",
            "discount": discount,
            "epsilon": epsilon,
            "lr": lr,
            "temperature": temperature
        }
    }


def create_dta_hyparams(trial):
    # TODO config.jsonをここで作る。
    dta_hyparams = create_sarsa_hyparams(trial)
    dta_hyparams["AGENT"]["raw_agent"] = dta_hyparams["AGENT"]["name"]
    dta_hyparams["AGENT"]["name"] = "DTAAgent"
    dta_hyparams["SHAPING"] = {}
    dta_hyparams["SHAPING"]["vid"] = "table"
    dta_hyparams["SHAPING"]["aggr_id"] = "dta"
    dta_hyparams["SHAPING"]["_range"] = 0
    return dta_hyparams


HYPERPARAMS_SAMPLER = {
    "DTAAgent": create_dta_hyparams
    # "OffsetDTAAgent": OffsetDTAAgent,
    # "SRSAgent": SRSAgent,
    # "NaiveRSAgent": NaiveRSAgent,
    # "SPRSAgent": SPRSAgent
}
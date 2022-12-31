from src.agents.sarsa import SarsaAgent
from src.agents.dta import DTAAgent, PartiallyDTAAgent
from src.agents.srs import SRSAgent, PartiallySRSAgent
from src.agents.naive import NaiveRSAgent
from src.agents.naive import LinearNaiveRSAgent
from src.agents.sprs import SPRSAgent


agent_dicts = {
    "SarsaAgent": SarsaAgent
}

shaped_agent_dicts = {
    "DTAAgent": DTAAgent,
    "SRSAgent": SRSAgent,
    "NaiveRSAgent": NaiveRSAgent,
    "LinearNaiveRSAgent": LinearNaiveRSAgent,
    "SPRSAgent": SPRSAgent,
    "PartiallyDTAAgent": PartiallyDTAAgent,
    "PartiallySRSAgent": PartiallySRSAgent
    }


def create_agent(config, env, rng, subgoals, q_value={}):
    agent_name = config["AGENT"]["name"]
    nfeatures, nactions = env.observation_space.n, env.action_space.n
    if agent_name in agent_dicts.keys():
        return agent_dicts[agent_name](
            config, nfeatures, nactions, rng, q_value
        )
    elif agent_name in shaped_agent_dicts.keys():
        raw_agent = agent_dicts[config["AGENT"]["raw_agent"]](
            config, nfeatures, nactions, rng, q_value
        )
        return shaped_agent_dicts[agent_name](
            raw_agent, env, subgoals, config
        )
    else:
        raise KeyError

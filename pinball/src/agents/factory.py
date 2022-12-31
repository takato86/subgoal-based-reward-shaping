from src.agents.actor_critic import ActorCriticAgent
from src.agents.srs import SRSAgent
from src.agents.dta import DTAAgent
from src.agents.naive import LinearNaiveRSAgent, NaiveRSAgent
from src.agents.sprs import SPRSAgent


agent_dicts = {
    "ActorCriticAgent": ActorCriticAgent
}

shaped_agent_dicts = {
    "SRSAgent": SRSAgent,
    "DTAAgent": DTAAgent,
    "NaiveRSAgent": NaiveRSAgent,
    "LinearNaiveRSAgent": LinearNaiveRSAgent,
    "SPRSAgent": SPRSAgent
}


def create_agent(config, env, subgoals):
    agent_name = config["agent"]["name"]
    if agent_name in agent_dicts.keys():
        return agent_dicts[agent_name](
            config["setting"]["seed"],
            env.action_space,
            env.observation_space,
            config["agent"]["basis_order"],
            config["agent"]["epsilon"],
            config["agent"]["gamma"],
            config["agent"]["lr_theta"],
            config["agent"]["lr_q"]
        )
    elif agent_name in shaped_agent_dicts.keys():
        raw_agent_name = config["agent"]["raw_agent"]
        raw_agent = agent_dicts[raw_agent_name](
            config["setting"]["seed"],
            env.action_space,
            env.observation_space,
            config["agent"]["basis_order"],
            config["agent"]["epsilon"],
            config["agent"]["gamma"],
            config["agent"]["lr_theta"],
            config["agent"]["lr_q"]
        )
        return shaped_agent_dicts[agent_name](
            raw_agent, env, config, subgoals
        )
    else:
        raise KeyError("{} is not in agent dicts: {}".format(
            agent_name,
            {**agent_dicts, **shaped_agent_dicts}.keys()
        ))

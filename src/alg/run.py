from src.alg.RoutingAgent import RoutingAgent
from src.mdp.NetworkMDP import RoutingEnv
from tqdm import trange

def run(            env: RoutingEnv,
                    agent: RoutingAgent,
                    steps: int,
                    train: bool = True,
                    render: bool = False,
                    show_progress: bool = True,
                    ):
    assert env is not None, " Env must exists. Got None instead of a gym.Env object"
    assert agent is not None, "Agent must exists. Got None instead of a Agent object"
    assert steps > 0, "Must run for some number positive number of steps. Got {} steps".format(steps)

    ret = []
    current_reward = 0

    # This iterable shows progress, or is a normal range depending
    r = range(steps) if not show_progress else trange(steps)
    obs, _ = env.reset()
    for i in r:
        connection_costs = env.get_costs()
        action = agent.act(obs, connection_costs)
        nobs, rewards, done, truncated, info = env.step(action)

        if train:
            agent.learn(obs, action, rewards, nobs, done, truncated, info)
        if render:
            env.render()

        # handle reset. Env may be a vector or a single env.
        # If single, done = bool. If vector, done = numpy array
        current_reward += sum(rewards)
        if done or truncated :
            ret.append(current_reward)
            current_reward = 0
            nobs, _ = env.reset()
        obs = nobs

    # wrap up
    env.close()
    return ret

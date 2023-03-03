import os

from matplotlib import pyplot as plt
from src.alg.run import run
from src.alg.RoutingAgent import RoutingAgent
from  src.mdp.NetworkMDP import *
import matplotlib.image
from moviepy.editor import *

# create env
env = RoutingEnv(render_mode="human")
print(env.observation_space)
print(env.action_space)



# show episode, save
for episode in range(10):
    done = False
    obs, _ = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        obs, rewards, dones, truncated, info = env.step(action)
        env.render()

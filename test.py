import os

from matplotlib import pyplot as plt
from src.alg.run import run
from src.alg.RoutingAgent import RoutingAgent
from  src.mdp.NetworkMDP import *
import matplotlib.image
from moviepy.editor import *
import torch
import numpy


# seed
seed = 2
numpy.random.seed(seed)
torch.manual_seed(seed)


# create env
env = RoutingEnv(render_mode=None)
print(env.observation_space)
print(env.action_space)

# create agent
agent = RoutingAgent(env.observation_space, env.action_space)


# train for some number of steps
accumulated_rewards, success_rates = run(env, agent, steps=100_000, train=True, show_progress= True)
accumulated_rewards = [np.mean(accumulated_rewards[max(i-5, 0):i+1]) for i in range(len(accumulated_rewards))]

# save learning graph
os.makedirs("results", exist_ok=True)
plt.plot(accumulated_rewards)
plt.xlabel("Episode")
plt.ylabel("Accumulated Reward")
plt.title("Learning Curve")
plt.savefig("results/learning_curve.png")
plt.clf()

# save success rates
plt.plot(success_rates)
plt.ylim(0.0, 1.0)
plt.xlabel("Episode")
plt.ylabel("Message Success Rate")
plt.title("Probability of Message Transmission")
plt.savefig("results/success_rate.png")
plt.clf()




# show episode, save
env = RoutingEnv(render_mode="rgb_array")
seconds_per_frame = 0.1
base_file_name = "experiment"
for episode in range(10):
    images = []
    done = False
    obs, _ = env.reset()
    for t in range(100):
        action = agent.act(obs, env.get_costs())
        obs, rewards, dones, truncated, info = env.step(action)
        frame = env.render()
        images.append(ImageClip(frame).set_duration(seconds_per_frame))
    video = concatenate(images, method="compose")
    video.write_videofile("results/episode_{}.mp4".format(episode), fps=24)

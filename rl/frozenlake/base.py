import gymnasium as gym
import torch
import math
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from tqdm import tqdm

# Observation Space
# 0 - 15
#
# Action Space
# left -> 0, down -> 1, right -> 2, up -> 3
# Reward -> +1 for reaching goal
# termination -> reach hole, frozen
# episode max_length = 100
    
def interact(policy, learner=None,
             episodes=50, return_env=False,
             size=4, run=1, max_runs=1,
             render_mode="rgb_array"):
    env = gym.make("FrozenLake-v1",
                   render_mode=render_mode,
                   desc=generate_random_map(size=size, p=0.9, seed=42))
    episodes = [[] for _ in range(episodes)]
    for i in tqdm(range(len(episodes)),
                  desc=f"Run ({run}/{max_runs})",
                  leave=False):
        observation, info = env.reset()
        terminated, truncated = False, False
        while not (terminated and truncated):
            old_obs = observation
            action = policy(env, observation)
            observation, reward, terminated, truncated, info = env.step(action)
            episodes[i].append((old_obs, action, reward, observation))
            if learner != None:
                learner(env, old_obs, action, reward, observation)

    if return_env:
        return episodes, env
    env.close()
    return episodes

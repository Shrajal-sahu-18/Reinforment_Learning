#Reinforcement Learning
import gymnasium as gym
import numpy as np
import random

env = gym.make('CliffWalking-v1')
print(env.observation_space.n)
print(env.action_space.n)

starting_state, _ = env.reset()

print(starting_state)

gamma = 0.99
alpha = 0.5
#Reinforcement Learning
import gymnasium as gym
import numpy as np
import random

env = gym.make('CliffWalking-v1')
print(env.observation_space.n)
print(env.action_space.n)
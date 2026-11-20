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
epsilon = 0.1
episodes = 500

Q = np.zeros((48,4))

def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        np.argmax(Q[state])
for episode in range(episodes):
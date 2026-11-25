import gymnasium as gym
import numpy as np
import random

gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500

Q = np.zeros((48,4))

def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

for episode in range(episodes):
    render = (episode % 50 == 0)
    if render:
        env = gym.make('CliffWalking-v1',render_mode = "human")
    else:
        env = gym.make('CliffWalking-v1')
    done = False
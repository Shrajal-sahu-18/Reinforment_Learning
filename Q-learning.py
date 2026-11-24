import gymnasium as gym
import numpy as np
import random

gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500

Q = np.zeros((48,4))
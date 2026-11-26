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
    state,_ = env.reset()
    total_reward = 0
    episode_len = 0

    while not done:
        action = epsilon_greedy(state)

        next_state,reward,terminated,truncated,_ = env.step(action)
        done = terminated or truncated

        Q[state,action]  += alpha * (reward + gamma * np.max(Q[ next_state]) - Q[state,action])
        state = next_state

        episode_len += 1
        total_reward += reward

    print(f"Episode ==> {episode+ 1} / {episodes} : Total_Reward ==> {total_reward} & Episode_len ==>{episode_len} ")
    env.close()
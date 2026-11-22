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
    render = (episode % 50 == 0)
    if render:
        env = gym.make('CliffWalking-v1',render_mode = "human")
        print(f"Rendering env for episode = {episode + 1} / {episodes}")
    else:
        env = gym.make('CliffWalking-v1')
    done = False
    state,_ = env.reset()
    action = epsilon_greedy(state)

    total_reward = 0
    episode_length = 0

    while not done:
        next_state,reward,terminated,truncated,_ = env.step(action)
        done = terminated or truncated
        next_action = epsilon_greedy(next_state)

        # SARSA update
        Q[state,action] += alpha * (reward + gamma *(Q[next_state,next_action] - Q[state,action]))

        state = next_state
        action = next_action

        total_reward += reward
        episode_length += 1
    print(f"Episode ==> {episode+1} / {episodes}: Total_reward = {total_reward} & Ep_length = {episode_length}")
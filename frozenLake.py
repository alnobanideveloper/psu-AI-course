import numpy as np
import random
from collections import deque

# Gym dependency replacement for FrozenLake
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=False ,render_mode="human") #   # Deterministic environment


# Q-learning parameters
alpha = 0.5  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.997  # Faster decay towards exploitation
min_epsilon = 0.01
episodes = 2000
max_timesteps = 100

# Initialize Q-table with zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Track rewards and success rates
episode_rewards = []
success_count = 0


def choose_action(state, epsilon):
    """Choose an action based on epsilon-greedy strategy."""
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration: choose random action
    else:
        return np.argmax(q_table[state])  # Exploitation: choose best action based on Q-table


def train_agent():
    """Train the agent using Q-learning."""
    global success_count

    epsilon = 1.0  # Reset epsilon for training
    for episode in range(episodes):
        # Reset the environment
        reset_result = env.reset()
        state = reset_result if isinstance(reset_result, int) else reset_result[0]

        total_reward = 0

        for t in range(max_timesteps):
            action = choose_action(state, epsilon)  # Choose an action based on current state
            next_state, reward, done, truncated, info = env.step(action)  # Take action and observe the result
            
            if done and reward == 0:  # If the agent fell into a hole
                reward = -1

            total_reward += reward

            # Update Q-value using the Q-learning formula
            q_table[state][action] = q_table[state][action] + alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state][action])

            state = next_state  # Move to the next state
            if done or truncated:  # End episode if the agent reaches a terminal state
                if reward > 0:  # If the agent reaches the goal
                    success_count += 1
                break

        # Track rewards
        episode_rewards.append(total_reward)

        # Decay epsilon (exploration rate) to slowly shift to exploitation
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])  # Average reward over the last 100 episodes
            success_rate = (success_count / (episode + 1)) * 100
            print(f"Episode {episode + 1}/{episodes} - Average reward (last 100 episodes): {avg_reward:.2f}")
            print(f"Episode {episode + 1}/{episodes} - Success rate: {success_rate:.2f}%\n")

def print_q_table():
    """Print the Q-table after training in a readable format."""
    print("Q-table after training:\n")
    # Iterate over all states and print Q-values for each action
    for state, q_values in enumerate(q_table):
        print(f"State {state}: {np.round(q_values, 2)}")



def render_policy():
    print("\nLearned policy:")
    symbols = ['←', '↓', '→', '↑']
    policy = []
    for i in range(env.observation_space.n):
        if np.all(q_table[i] == 0):  # Check if all Q-values are 0
            policy.append('?')  # Use '?' to indicate an unexplored state
        else:
            policy.append(symbols[np.argmax(q_table[i])])  # Choose the best action
    policy = np.array(policy).reshape(4, 4)  # Reshape to match the grid
    for row in policy:
        print(' '.join(row))

# Train the agent
train_agent()
#print q table
print_q_table()
# Print the learned policy grid
render_policy()


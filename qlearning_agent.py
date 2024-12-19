import numpy as np
import random

class QLearningAgent:
    def __init__(self, rows, cols, num_actions, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01):
        self.rows = rows
        self.cols = cols
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.Q = np.zeros((rows, cols, num_actions))
        self.episode_rewards = []

    def choose_action(self, state):
        r, c = state
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.Q[r, c, :])

    def update_q(self, old_state, action, reward, next_state, done):
        old_r, old_c = old_state
        new_r, new_c = next_state
        best_next = np.argmax(self.Q[new_r, new_c, :])
        td_target = reward + (0 if done else self.gamma * self.Q[new_r, new_c, best_next])
        td_error = td_target - self.Q[old_r, old_c, action]
        self.Q[old_r, old_c, action] += self.alpha * td_error

    def end_episode(self, episode_reward):
        self.episode_rewards.append(episode_reward)
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

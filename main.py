import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment import GridEnvironment
from qlearning_agent import QLearningAgent
from visualization import Visualizer

# Parameters
GRID_ROWS = 8
GRID_COLS = 8
START = (0, 0)
GOAL = (GRID_ROWS - 1, GRID_COLS - 1)
NUM_EPISODES = 50
MAX_STEPS_EPISODE = 200

env = GridEnvironment(GRID_ROWS, GRID_COLS, START, GOAL)
agent = QLearningAgent(GRID_ROWS, GRID_COLS, num_actions=4, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9, epsilon_min=0.01)
viz = Visualizer(GRID_ROWS, GRID_COLS, agent, env)

current_episode = 0
steps_in_episode = 0
state = START
done = False
training_finished = False
cumulative_reward = 0.0

def init():
    global current_episode, steps_in_episode, state, done, cumulative_reward, training_finished
    current_episode = 0
    steps_in_episode = 0
    state = START
    done = False
    training_finished = False
    cumulative_reward = 0.0

    viz.update_episode_text(current_episode, steps_in_episode)
    # Initialize Q-text
    for rr in range(GRID_ROWS):
        for cc in range(GRID_COLS):
            viz.update_cell_text(rr, cc)

    # Clear stats line
    viz.line_rewards.set_data([], [])
    return viz.agent_patch, viz.line_rewards

def update(frame):
    global current_episode, steps_in_episode, state, done, cumulative_reward, training_finished

    if training_finished:
        return viz.agent_patch, viz.line_rewards

    # Check if episode ended
    if done or steps_in_episode >= MAX_STEPS_EPISODE:
        agent.episode_rewards.append(cumulative_reward)
        current_episode += 1
        steps_in_episode = 0
        state = START
        done = False
        cumulative_reward = 0.0
        viz.update_agent_position(state)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        # Update stats line
        viz.update_rewards_plot()

        if current_episode >= NUM_EPISODES:
            training_finished = True
            viz.update_episode_text(current_episode, steps_in_episode, training_finished=True)
            return viz.agent_patch, viz.line_rewards
        else:
            viz.update_episode_text(current_episode, steps_in_episode)
        return viz.agent_patch, viz.line_rewards

    old_state = state
    action = agent.choose_action(state)
    next_state, reward, done_flag = env.step(state, action)

    # Update Q-values
    agent.update_q(old_state, action, reward, next_state, done_flag)

    state = next_state
    steps_in_episode += 1
    done = done_flag
    cumulative_reward += reward

    viz.update_agent_position(state)
    viz.update_episode_text(current_episode, steps_in_episode)
    viz.update_two_cells(old_state, next_state)

    return viz.agent_patch, viz.line_rewards

anim = FuncAnimation(viz.fig, update, frames=NUM_EPISODES * MAX_STEPS_EPISODE + 100,
                     init_func=init, interval=200)

plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Visualizer:
    def __init__(self, rows, cols, agent, environment):
        self.rows = rows
        self.cols = cols
        self.agent = agent
        self.env = environment

        self.fig = plt.figure(figsize=(10,5))
        # Left subplot for grid
        self.ax_board = self.fig.add_subplot(1, 2, 1)
        self.ax_board.set_xlim(0, cols)
        self.ax_board.set_ylim(0, rows)
        self.ax_board.set_aspect('equal')
        self.ax_board.set_title("Path finder via RL")

        # Draw chessboard
        for r in range(rows):
            for c in range(cols):
                color = 'darkgray' if (r+c)%2==0 else 'lightgray'
                square = Rectangle((c, r), 1, 1, facecolor=color, edgecolor='black')
                self.ax_board.add_patch(square)

        # Start & goal
        start_patch = Rectangle((self.env.start[1], self.env.start[0]), 1, 1, facecolor='green')
        self.ax_board.add_patch(start_patch)

        goal_patch = Rectangle((self.env.goal[1], self.env.goal[0]), 1, 1, facecolor='red')
        self.ax_board.add_patch(goal_patch)

        # Agent
        self.agent_patch = Rectangle((self.env.start[1], self.env.start[0]), 1, 1, facecolor='blue')
        self.ax_board.add_patch(self.agent_patch)

        # Q-values text
        self.cell_texts = [[None for _ in range(cols)] for _ in range(rows)]
        for rr in range(rows):
            for cc in range(cols):
                txt = self.ax_board.text(cc+0.5, rr+0.5, "0.00", ha='center', va='center', fontsize=6, color='white')
                self.cell_texts[rr][cc] = txt

        # Episode text
        self.episode_text = self.fig.text(0.15, 0.92, '', fontsize=10,
                                         color='black', backgroundcolor='white', ha='left')

        # Right subplot for stats
        self.ax_stats = self.fig.add_subplot(1, 2, 2)
        self.ax_stats.set_title("Episode Rewards")
        self.ax_stats.set_xlabel("Episode")
        self.ax_stats.set_ylabel("Total Reward")
        self.line_rewards, = self.ax_stats.plot([], [], color='blue', marker='o')
        self.ax_stats.set_xlim(0, 50)
        self.ax_stats.set_ylim(-200, 10)

    def update_cell_text(self, rr, cc):
        max_q = self.agent.Q[rr, cc, :].max()
        self.cell_texts[rr][cc].set_text(f"{max_q:.2f}")

    def update_two_cells(self, old_state, new_state):
        or_, oc_ = old_state
        nr_, nc_ = new_state
        self.update_cell_text(or_, oc_)
        self.update_cell_text(nr_, nc_)

    def update_episode_text(self, episode, steps, training_finished=False):
        if training_finished:
            self.episode_text.set_text(f"Episode: {episode}\nTraining Complete!")
        else:
            self.episode_text.set_text(f"Episode: {episode}\nSteps: {steps}")

    def update_agent_position(self, state):
        self.agent_patch.set_xy((state[1], state[0]))

    def update_rewards_plot(self):
        x_data = range(len(self.agent.episode_rewards))
        y_data = self.agent.episode_rewards
        self.line_rewards.set_data(x_data, y_data)
        if len(y_data) > 0:
            self.ax_stats.set_ylim(min(y_data)-10, max(y_data)+10)


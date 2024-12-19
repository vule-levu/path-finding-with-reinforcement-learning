import numpy as np

ACTIONS = [
    (-1, 0),  # up
    (0, +1),  # right
    (+1, 0),  # down
    (0, -1)   # left
]

class GridEnvironment:
    def __init__(self, rows, cols, start, goal):
        self.rows = rows
        self.cols = cols
        self.start = start
        self.goal = goal

    def step(self, state, action):
        r, c = state
        dr, dc = ACTIONS[action]
        nr = max(0, min(r + dr, self.rows - 1))
        nc = max(0, min(c + dc, self.cols - 1))
        next_state = (nr, nc)

        if next_state == self.goal:
            return next_state, 10.0, True
        else:
            return next_state, -1.0, False

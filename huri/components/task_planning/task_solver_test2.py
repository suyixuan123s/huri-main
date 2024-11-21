"""
Author: Yixuan Su
Date: 2024/11/12 21:30
File: task_solver_test2.py
"""

import numpy as np
from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver
import time

if __name__ == '__main__':


    goal_pattern_demo1 = np.array([
                             [1, 0, 0, 2, 3, 4, 0, 0, 5, 0, ],
                             [0, 1, 2, 2, 3, 4, 0, 0, 5, 0, ],
                             [0, 1, 1, 0, 3, 4, 0, 0, 5, 0, ],
                             [0, 0, 1, 2, 5, 4, 0, 0, 0, 0, ],
                             [0, 2, 0, 0, 3, 0, 4, 0, 5, 0, ]])

    goal_pattern_demo2 = np.array([
                             [1, 0, 0, 2, 3, 4, 0, 0, 5, 0, ],
                             [0, 1, 2, 2, 3, 4, 0, 0, 5, 0, ],
                             [0, 1, 1, 2, 3, 4, 0, 0, 5, 0, ],
                             [0, 0, 1, 2, 5, 4, 0, 0, 0, 0, ],
                             [0, 0, 0, 0, 3, 0, 4, 0, 5, 0, ]])

    goal_pattern = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]])
    solver = D3QNSolver(goal_pattern)
    st = time.time()
    r_path, path = solver.solve(init_state=goal_pattern_demo2, toggle_show=True)
    et = time.time()
    print(f"random solve time: {et-st}")
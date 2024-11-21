""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20240326osaka

"""
import time

if __name__ == '__main__':
    import numpy as np
    from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver

    goal_pattern = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                             [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]])
    solver = D3QNSolver(goal_pattern)
    st = time.time()
    r_path, path = solver.random_solve(toggle_show=True)
    et = time.time()
    print(f"random solve time: {et-st}")




"""
Author: Yixuan Su
Date: 2024/11/12 22:56
File: task_solver_test3.py
"""

import time

if __name__ == '__main__':
    import numpy as np
    from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver

    goal_pattern = np.array([
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
        [1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
    ])

    # init_pattern = np.array([
    #     [1, 0, 0, 2, 3, 4, 0, 0, 5, 0],
    #     [0, 1, 2, 2, 3, 4, 0, 0, 5, 0],
    #     [0, 1, 1, 0, 3, 4, 0, 0, 5, 0],
    #     [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
    #     [2, 0, 0, 0, 3, 0, 4, 0, 5, 0]
    # ])

    init_pattern = np.array([
        [1, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 2, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 0, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [2, 0, 0, 0, 3, 0, 4, 0, 5, 0]
    ])

    # 定义矩阵
    Combination_1 = np.array([
        [0, 0, 0, 0, 3, 4, 0, 0, 5, 0],
        [0, 1, 2, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_2 = np.array([
        [0, 0, 0, 0, 3, 4, 0, 0, 5, 0],
        [2, 1, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_7 = np.array([
        [0, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_7 = np.array([
        [0, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_7 = np.array([
        [0, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_11 = np.array([
        [1, 0, 0, 0, 3, 4, 0, 0, 5, 0],
        [0, 0, 2, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_22 = np.array([
        [0, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_23 = np.array([
        [1, 0, 0, 0, 3, 4, 0, 0, 5, 0],
        [2, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    # 定义矩阵
    Combination_28 = np.array([
        [1, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 0, 2, 3, 4, 0, 0, 5, 0],
        [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
        [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
        [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
    ])

    solver = D3QNSolver(goal_pattern)
    st = time.time()
    r_path, path = solver.solve(init_state=Combination_28, toggle_show=True)
    et = time.time()
    print(f"random solve time: {et - st}")

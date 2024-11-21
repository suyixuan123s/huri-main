"""
Author: Yixuan Su
Date: 2024/11/18 10:03
File: validate_moves_and_paths.py
"""
import os
import numpy as np
from itertools import product
from tube_puzzle_learning_solver import D3QNSolver
from huri.learning.env.rack_v3.env import RackArrangementEnv

# 定义目标状态
goal_pattern = np.array([
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
])

# 初始化初始状态
Combination_1 = np.array([
    [0, 0, 0, 0, 3, 4, 0, 0, 5, 0],
    [0, 1, 2, 2, 3, 4, 0, 0, 5, 0],
    [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
    [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
    [1, 0, 0, 2, 3, 0, 4, 0, 5, 0],
])

# 初始化环境和求解器
env = RackArrangementEnv(rack_size=(5, 10), num_classes=5)
env.reset_goal(goal_pattern)
solver = D3QNSolver(goal_pattern=goal_pattern)

# 确定试管位置和空余位置
nonzero_positions = np.argwhere(Combination_1 > 0)  # 非零试管位置
empty_positions = np.argwhere(Combination_1 == 0)  # 空余位置

print(f"非零试管的位置：\n{nonzero_positions}")
print(f"空余的位置：\n{empty_positions}")

# 保存成功或者失败生成路径的矩阵
successful_matrices = []
failed_matrices = []

# 定义文件路径
base_dir = r'E:\ABB-Project\Dr_Chen\huri-main\huri\components\task_planning\test2'
successful_matrices_path = os.path.join(base_dir, "successful_matrices_file.txt")
filed_matrices_path = os.path.join(base_dir, "failed_matrices_file.txt")
error_log_path = os.path.join(base_dir, "error_log.txt")
os.makedirs(base_dir, exist_ok=True)

# 清空文件内容
open(successful_matrices_path, "w").close()
open(filed_matrices_path, "w").close()
open(error_log_path, "w").close()

# 遍历所有可能的移动操作
for from_pos, to_pos in product(nonzero_positions, empty_positions):
    modified_pattern = Combination_1.copy()
    value_to_move = modified_pattern[tuple(from_pos)]

    # 执行移动
    modified_pattern[tuple(from_pos)] = 0
    modified_pattern[tuple(to_pos)] = value_to_move

    # 验证路径是否可生成
    env.reset_state(modified_pattern)
    try:
        action_sequence, sequence = solver.solve(modified_pattern, toggle_show=False)
        if sequence:  # 如果生成了路径
            print(f"初始状态为:\n {Combination_1}\n")
            print(f"成功生成路径: 从 {from_pos} 移动到 {to_pos}")
            successful_matrices.append(modified_pattern)
            with open(successful_matrices_path, "a") as success_file:
                success_file.write(f"初始状态为:\n {Combination_1}\n")
                success_file.write(f"成功生成路径: 从 {from_pos} 移动到 {to_pos}\n")
                success_file.write(f"Modified pattern:\n{np.array2string(modified_pattern)}\n\n")
        else:
            print(f"初始状态为:\n {Combination_1}\n")
            print(f"失败生成路径: 从 {from_pos} 移动到 {to_pos}")
            failed_matrices.append(modified_pattern)
            with open(filed_matrices_path, "a") as failed_file:
                failed_file.write(f"初始状态为:\n {Combination_1}\n")
                failed_file.write(f"失败生成路径: 从 {from_pos} 移动到 {to_pos}\n")
                failed_file.write(f"Modified pattern:\n{np.array2string(modified_pattern)}\n\n")
    except Exception as e:
        print(f"路径生成失败: 从 {from_pos} 移动到 {to_pos}, 错误: {e}")
        with open(error_log_path, "a") as error_log:
            error_log.write(f"路径生成失败: 从 {from_pos} 移动到 {to_pos}, 错误: {e}\n")

# 打印成功和失败矩阵
print("\n成功生成路径的矩阵:")
for idx, matrix in enumerate(successful_matrices, start=1):
    print(f"成功矩阵 {idx}:\n{np.array2string(matrix)}")

print("\n失败生成路径的矩阵:")
for idx, matrix in enumerate(failed_matrices, start=1):
    print(f"失败矩阵 {idx}:\n{np.array2string(matrix)}")

"""
Author: Yixuan Su
Date: 2024/11/17 11:19
File: state_to_action_verification2.py
"""

import os
import numpy as np
from datetime import datetime 
from itertools import combinations, product
from tube_puzzle_learning_solver import D3QNSolver
from huri.learning.env.rack_v3.env import RackArrangementEnv, from_action, RackState

# 定义目标状态
goal_pattern = np.array([
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0],
    [1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
])

# 定义初始状态
init_pattern = np.array([
    [1, 0, 0, 2, 3, 4, 0, 0, 5, 0],
    [2, 1, 2, 2, 3, 4, 0, 0, 5, 0],
    [0, 1, 1, 2, 3, 4, 0, 0, 5, 0],
    [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
    [1, 0, 0, 2, 3, 0, 4, 0, 5, 0]
])

# 初始化环境和求解器
env = RackArrangementEnv(rack_size=(5, 10), num_classes=5)
env.reset_goal(goal_pattern)
solver = D3QNSolver(goal_pattern=goal_pattern)

# 统计每种试管的数量
unique, counts = np.unique(init_pattern, return_counts=True)
tube_counts = dict(zip(unique, counts))
print(f"Initial tube counts: {tube_counts}")

# 收集所有需要删除的试管类型及其位置
to_delete = {}
for tube, count in tube_counts.items():
    if tube == 0 or count <= 5:  # 跳过空格（0）或数量正常的试管
        continue
    positions = np.argwhere(init_pattern == tube)
    excess = count - 5  # 需要删除的试管数量
    to_delete[tube] = (positions, excess)

print(f"Excess tubes to handle: {to_delete}")

# 生成多类试管的删除组合
tube_combinations = []
for tube, (positions, excess) in to_delete.items():
    tube_combinations.append(list(combinations(positions, excess)))

# 笛卡尔积生成所有可能的删除情况
all_combinations = product(*tube_combinations)

# 保存所有有效的状态矩阵
valid_patterns = []

# 定义输出文件路径
output_file_path = r"E:\ABB-Project\Dr_Chen\huri-main\huri\components\task_planning\deleted_combinations.txt"
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 添加时间和作者信息
author = "Yixuan Su"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 打开文件写入模式
with open(output_file_path, "w") as file:

    # 写入文件元信息
    file.write(f"File generated on: {timestamp}\n")
    file.write(f"Author: {author}\n\n")

    # 遍历所有组合，验证路径
    for idx, combo in enumerate(all_combinations, start=1):
        # 创建新的状态矩阵
        modified_pattern = init_pattern.copy()
        combined_positions = []
        for tube_combo in combo:
            combined_positions.extend(tube_combo)

        # 删除试管
        for pos in combined_positions:
            modified_pattern[tuple(pos)] = 0

        # 打印组合和修改后的矩阵
        print(f"\nCombination {idx}: Deleted positions {combined_positions}")
        print(modified_pattern)

        # 保存组合和修改后的矩阵到文件
        file.write(f"\nCombination {idx}: Deleted positions {combined_positions}\n")
        file.write(f"{modified_pattern}\n")


print(f"All combinations saved to: {output_file_path}")





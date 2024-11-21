"""
Author: Yixuan Su
Date: 2024/11/17 11:19
File: state_to_action_verification1.py
"""

import os
from datetime import datetime

from tube_puzzle_learning_solver import D3QNSolver
from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState, from_action
import numpy as np

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
    [0, 1, 2, 2, 3, 4, 0, 0, 5, 0],
    [0, 1, 1, 0, 3, 4, 0, 0, 5, 0],
    [0, 0, 1, 2, 5, 4, 0, 0, 0, 0],
    [2, 0, 0, 0, 3, 0, 4, 0, 5, 0]
])

# 初始化环境
env = RackArrangementEnv(rack_size=(5, 10), num_classes=5)
env.reset_goal(goal_pattern)
env.reset_state(init_pattern)

# 初始化D3QNSolver
solver = D3QNSolver(goal_pattern=goal_pattern)

# 定义输出文件路径
output_file_path = r"E:\ABB-Project\Dr_Chen\huri-main\huri\components\task_planning\Derived_action_sequence.txt"
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

# 添加时间和作者信息
author = "Yixuan Su"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 使用D3QNSolver计算路径
action_sequence, sequence = solver.solve(init_pattern, toggle_show=True)

try:
    # 打开文件写入模式
    with open(output_file_path, "w") as file:
        # 写入文件元信息
        file.write(f"File generated on: {timestamp}\n")
        file.write(f"Author: {author}\n\n")

        # 写入路径序列
        print("Action sequence to reach the goal:")
        file.write("Action sequence to reach the goal:\n")
        for idx, action in enumerate(sequence):
            print(f"Step {idx + 1}: Action {action}")
            file.write(f"Step {idx + 1}: Action {action}\n")

        for idx, state in enumerate(sequence):
            print(f"Step {idx + 1}: Type: {type(state)}, Shape: {state.shape}")
            file.write(f"Step {idx + 1}: Type: {type(state)}, Shape: {state.shape}\n")

        actions = []
        invalid_steps = []

        for i in range(1, len(sequence)):
            prev_state = sequence[i - 1]
            current_state = sequence[i]

            # 调试信息
            print(f"Step {i}: Previous state:\n{prev_state}")
            print(f"Step {i}: Current state:\n{current_state}")

            file.write(f"Step {i}: Previous state:\n{prev_state}\n")
            file.write(f"Step {i}: Current state:\n{current_state}\n")

            # 使用环境方法推导动作
            action = env.action_between_states(prev_state, current_state)

            # 检查动作是否有效
            if action is None or not isinstance(action, int):
                print(f"Step {i}: Derived action is invalid: {action}")
                file.write(f"Step {i}: Derived action is invalid: {action}\n")
                invalid_steps.append(i)
                continue

            # 打印当前推导的动作变换
            print(f"Step {i}: Derived action: {action}")
            file.write(f"Step {i}: Derived action: {action}\n")

            # 从状态获取 rack_size
            rack_size = prev_state.shape

            # 检查动作合法性
            feasible_actions = RackState(prev_state).feasible_action_set
            print(f"Step {i}: Feasible actions: {feasible_actions}")
            file.write(f"Step {i}: Feasible actions: {feasible_actions}\n")
            if action not in feasible_actions:
                print(f"Step {i}: Invalid action derived: {action}")
                file.write(f"Step {i}: Invalid action derived: {action}\n")
                invalid_steps.append(i)
                continue

            # 使用 from_action 函数解析动作 ID
            try:
                move_to_idx, move_from_idx = from_action(rack_size, action)
                if move_to_idx is None or move_from_idx is None:
                    print(f"Step {i}: Action {action} could not be decoded.")
                    file.write(f"Step {i}: Action {action} could not be decoded.\n")
                    invalid_steps.append(i)
                    continue
                print(f"Move to: {move_to_idx}, Move from: {move_from_idx}")
                file.write(f"Move to: {move_to_idx}, Move from: {move_from_idx}\n")
            except Exception as e:
                print(f"Step {i}: Error decoding action {action}: {e}")
                file.write(f"Step {i}: Error decoding action {action}: {e}\n")
                invalid_steps.append(i)
                continue

            # 保存合法动作
            actions.append(action)

        # 打印动作序列
        print("Derived action sequence:")
        file.write("Derived action sequence:\n\n")
        for idx, action in enumerate(actions):
            print(f"Action {idx + 1}: {action}")
            file.write(f"Action {idx + 1}: {action}\n")

        # 打印推导失败的步骤
        if invalid_steps:
            print(f"Steps with invalid actions: {invalid_steps}")
            file.write(f"Steps with invalid actions: {invalid_steps}\n")

    print(f"All Derived action sequence saved to: {output_file_path}")

except Exception as e:

    print(f"Error writing to file: {e}")

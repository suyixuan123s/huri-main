"""
Author: Yixuan Su
Date: 2024/11/17 11:19
File: state_to_action_verification3.py
"""

import os
import numpy as np
from datetime import datetime
from itertools import combinations, product
from tube_puzzle_learning_solver import D3QNSolver
from huri.learning.env.rack_v3.env import RackArrangementEnv, from_action, RackState, RackStatePlot

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
successful_combinations = []
failed_combinations = []

base_dir = r"E:\ABB-Project\Dr_Chen\huri-main\huri\components\task_planning\test3"
os.makedirs(base_dir, exist_ok=True)

# 添加时间和作者信息
author = "Yixuan Su"
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

    # 为每个组合创建一个独立的文件夹
    combination_folder = os.path.join(base_dir, f"Combination_{idx}")
    os.makedirs(combination_folder, exist_ok=True)

    # 文件名定义
    combination_file_path = os.path.join(combination_folder, f"Combination_{idx}.txt")
    successful_combinations_path = r"E:\ABB-Project\Dr_Chen\huri-main\huri\components\task_planning\test1\successful_combinations.txt"
    failed_combinations_path = r"E:\ABB-Project\Dr_Chen\huri-main\huri\components\task_planning\test1\failed_combinations.txt"

    with open(combination_file_path, "w") as file:
        # 写入文件元信息
        file.write(f"File generated on: {timestamp}\n")
        file.write(f"Author: {author}\n\n")

        file.write(f"Combination {idx}\n")
        file.write(f"Deleted positions: {combined_positions}\n")
        file.write(f"Modified pattern:\n{modified_pattern}\n\n")

        # 验证路径是否可生成
        env.reset_state(modified_pattern)
        try:
            action_sequence, sequence = solver.solve(modified_pattern, toggle_show=False)

            if sequence:  # 如果能生成路径，说明成功
                print(f"Combination {idx}: Successfully found a path.")
                file.write(f"Combination {idx}: Successfully found a path.\n")

                with open(successful_combinations_path, "a") as success_file:
                    success_file.write(f"Combination {idx}\n")
                    success_file.write(f"Deleted positions: {combined_positions}\n")
                    success_file.write(f"Modified pattern:\n{modified_pattern}\n\n")

                # # 保存图像：每次生成图像时保存
                # rsp = RackStatePlot(goal_pattern)
                # plot = rsp.plot_states(sequence, row=6, img_scale=10, toggle_fill=True)
                #
                # # 保存图像到文件夹
                # plot_file_path = os.path.join(combination_folder, f"Combination_{idx}_bg3.png")
                # plot.save_fig(plot_file_path, dpi=300)
                #
                # # 保存经过 A* 精炼后的路径图
                # plot = rsp.plot_states(sequence, row=8, img_scale=10, toggle_fill=True)
                # plot_file_path = os.path.join(combination_folder, f"Combination_{idx}_refined_path.jpg")
                # plot.save_fig(plot_file_path, dpi=300)

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

            else:
                print(f"Combination {idx}: No path found.")
                file.write(f"Combination {idx}: No path found.\n")
                with open(failed_combinations_path, "a") as false_file:
                    false_file.write(f"Combination {idx}\n")
                    false_file.write(f"Deleted positions: {combined_positions}\n")
                    false_file.write(f"Modified pattern:\n{modified_pattern}\n\n")

        except Exception as e:
            print(f"Combination {idx}: Error occurred during path finding: {e}")
            file.write(f"Combination {idx}: Error occurred during path finding: {e}\n")
            failed_combinations.append((combo, modified_pattern))  # 保存失败的组合

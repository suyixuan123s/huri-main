import numpy as np
from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStatePlot
import huri.core.file_sys as fs
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# initialize the environment

rack_size = (5, 10)
num_tube_classes = 3

goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         is_curriculum_lr=True,
                         is_goalpattern_fixed=True,
                         is_evl_mode=True,
                         difficulty=30)

env.goal_pattern = goal_pattern

init_state = env.reset()

print(init_state)
print(init_state[:])

solver = DQNSolver()
path = solver.solve(current_state=init_state,
                    goal_pattern=goal_pattern,
                    toggle_result=True)

# # visualize the result
if len(path) > 1:
    plotter = RackStatePlot(goal_pattern=goal_pattern)
    fig = plotter.plot_states(path)

    plt.show()

import numpy as np
from huri.learning.env.arrangement_planning_two_rack.env import TwoRackArrangementEnv, RackStatePlot
import huri.core.file_sys as fs
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolverTwoRack
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

# initialize the environment
num_tube_classes = 3
rack_size_1 = (5, 10)
rack_size_2 = (5, 10)
two_rack_size = (rack_size_1[0] + rack_size_2[0] + 1,
                 max(rack_size_1[1], rack_size_2[1]))
action_space_dim = np.prod(two_rack_size) ** 2
observation_space_dim = (1, *two_rack_size)
env = TwoRackArrangementEnv(from_rack_size=rack_size_1,
                            to_rack_size=rack_size_2,
                            num_classes=num_tube_classes,
                            observation_space_dim=observation_space_dim,
                            action_space_dim=action_space_dim,
                            is_curriculum_lr=True,
                            is_goalpattern_fixed=True,
                            difficulty=26, )

goal_pattern = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3], ])

env.goal_pattern = goal_pattern

init_state = env.reset()

print(init_state)
print(init_state[:])

solver = DQNSolverTwoRack()
path = solver.solve(current_state=init_state,
                    goal_pattern=goal_pattern,
                    toggle_result=True)
print(len(path))
# # visualize the result
if len(path) > 1:
    plotter = RackStatePlot(goal_pattern=goal_pattern)
    fig = plotter.plot_states(path, row=12)

    plt.show()

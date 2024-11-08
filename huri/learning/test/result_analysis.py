import numpy as np
import huri.core.file_sys as fs
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
from huri.learning.env.arrangement_planning_rack_gc_apex.env import RackStatePlot, RackArrangementEnv, RackState
from huri.learning.A_start_teacher.A_star_teacher import E, F
# initialize the environment
import cv2
GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])

drawer = RackStatePlot(GOAL_PATTERN)
# path = fs.load_pickle("demo")
path = fs.load_pickle("dqn")

if len(path) > 0:
    for i in range(1, len(path)):
        print(f"Step {i}")
        s1 = RackState(path[i - 1])
        s2 = RackState(path[i])

        e_s1 = E(s1.state, GOAL_PATTERN)
        e_s2 = E(s2.state, GOAL_PATTERN)

        f_s1 = F(s1.state)
        f_s2 = F(s2.state)

        print("Entropy s1: ", e_s1)
        print("Entropy s2: ", e_s2)
        print("Entropy Diff: ", np.sum(e_s2 - e_s1))

        print("Fillable f1: ", f_s1)
        print("Fillable f2: ", f_s2)
        print("Fillable Diff: ", np.sum(f_s2 - f_s1))
        print("Reward", np.sum(e_s2 - e_s1) + .5*np.sum(f_s2 - f_s1))
        print("-" * 20)

    p = drawer.plot_states(path, plot_movable=True, plot_fillable=True, row=12, )
    img = p.get_img()

    cv2.imwrite("as.jpg", img)
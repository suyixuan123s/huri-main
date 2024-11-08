import huri.core.file_sys as fs
from huri.learning.env.arrangement_planning_rack_gc_apex.env import RackStatePlot, RackArrangementEnv
import cv2
import numpy as np
from huri.learning.A_start_teacher.A_star_teacher import E, F
from huri.components.utils.matlibplot_utils import Plot

GOAL_PATTERN = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
rack_size = (5, 10)
num_classes = 3
observation_space_dim = (num_classes * 2, *rack_size)
action_space_dim = np.prod(rack_size) ** 2
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_goalpattern_fixed=False,
                         is_curriculum_lr=False)
env.goal_pattern = GOAL_PATTERN
drawer = RackStatePlot(GOAL_PATTERN)

while 1:
    s1 = env.reset()
    act = env.sample()
    s2, reward, is_done, _ = env.step(act)

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

    p = drawer.plot_states([s1.state, s2.state], plot_movable=True, plot_fillable=True)
    img = p.get_img()

    cv2.imshow("as", img)
    cv2.waitKey(0)

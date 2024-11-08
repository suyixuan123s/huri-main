import huri.core.file_sys as fs
from huri.learning.env.arrangement_planning_rack_gc_apex.env import RackStatePlot, RackState
import cv2
from constants import GOAL_PATTERN
import numpy as np
from huri.components.utils.matlibplot_utils import Plot

from huri.learning.A_start_teacher.A_star_teacher import rm_ras_actions, E, F

shortest_path_np, r_shortest_path_np, infeasible_dict = fs.load_pickle("debug_path")

drawer = RackStatePlot(GOAL_PATTERN)
p = drawer.plot_states(shortest_path_np, row=13)
img = p.get_img()
cv2.imshow("a", img)
cv2.waitKey(0)

r_r_shortest_path_np = rm_ras_actions(shortest_path_np, h=8, infeasible_dict=infeasible_dict,
                                      goal_pattern=GOAL_PATTERN)

drawer = RackStatePlot(GOAL_PATTERN)

E(r_shortest_path_np[0], goal_pattern=GOAL_PATTERN)

r_r_shortest_path_np = rm_ras_actions(r_shortest_path_np, h=8, infeasible_dict=infeasible_dict,
                                      goal_pattern=GOAL_PATTERN)

p = drawer.plot_states(r_shortest_path_np)
img = p.get_img()

p2 = drawer.plot_states(r_r_shortest_path_np, plot_movable=True, plot_fillable=True)
for i in range(1, len(r_r_shortest_path_np)):
    s1 = RackState(r_r_shortest_path_np[i - 1])
    s2 = RackState(r_r_shortest_path_np[i])

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
    print("-"*20)

img2 = p2.get_img()

cv2.imshow("as", img)
cv2.imshow("as2", img2)
cv2.waitKey(0)

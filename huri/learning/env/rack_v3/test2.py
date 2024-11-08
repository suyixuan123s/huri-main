import time

import cv2

from env import to_action, from_action, RackState, RackArrangementEnv, from_action_to_mat, RackStatePlot
import numpy as np
from ffmath import rotation_matrix
rack_size = (2, 2)

obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)
print(act_dim)
num_classes = 1
observation_space_dim = obs_dim
action_space_dim = act_dim

env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_goalpattern_fixed=False,
                         is_curriculum_lr=False,
                         seed=np.random.randint(999))

def multiply_matrices(matrices_list):
    result = matrices_list[0]
    for i in range(1, len(matrices_list)):
        result = np.dot(result, matrices_list[i])
    return result


mats = []
s_n = env.reset()
for _ in range(4):
    a_n = env.sample()
    s_np1, r_n, is_done, _ = env.step(a_n)

    # print(from_action(rack_size, a_n))
    # p = RackStatePlot(s_n)
    # img = p.plot_states([np.dot(from_action_to_mat(rack_size, a_n), s_n.state.ravel()).reshape(rack_size)]).get_img()
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    mat = from_action_to_mat(rack_size, a_n)
    print(mat)
    mats.append(mat)

print(multiply_matrices(mats))






print(rotation_matrix(s_n.state.ravel(), s_np1.state.ravel()))

print("Det is ",np.linalg.det(multiply_matrices(mats)))
p = RackStatePlot(s_n)
img = p.plot_states([np.dot(multiply_matrices(mats), s_n.state.ravel()).reshape(rack_size)]).get_img()
cv2.imshow("test", img)
cv2.waitKey(0)

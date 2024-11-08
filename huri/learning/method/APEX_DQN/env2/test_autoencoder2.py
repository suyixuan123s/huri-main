from ae2 import AE
import numpy as np
from huri.learning.env.rack_v2.env import RackArrangementEnv, RackStatePlot
from huri.learning.utils import select_device, LOGGER
import huri.core.file_sys as fs
import torch
import cv2

num_tube_classes = 1
rack_size = (5, 10)
action_space_dim = np.prod(rack_size) ** 2
observation_space_dim = (rack_size[0], rack_size[1])
observation_space_dim_nn = (1, *rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=False,
                         is_goalpattern_fixed=False,
                         seed=888)

device = select_device()
state_dim = np.prod(rack_size)
model = AE(input_shape=state_dim, output_shape=state_dim ** 2).to(device)
eval_net_path = fs.Path("test_encoder2.pt")
model.load_state_dict(torch.load(str(eval_net_path)))
model.eval()
for _ in range(1000):
    state = env.reset()
    goal = env.goal_pattern

    res = state.state.copy()
    res[(state.state - goal.state) <= 0] = 0
    res = res.astype(np.float32)

    s = torch.tensor(state.state, dtype=torch.float32, device=device)
    g = torch.tensor(goal.state, dtype=torch.float32, device=device)
    r, code = model(s.reshape(1, -1), g.reshape(1, -1))
    # r = torch.round(r).reshape(-1)
    r = r.reshape(-1)

    r_idx = torch.where(r > .9)
    r_idx = r_idx[0].cpu().detach().numpy()

    feasible_action = state.feasible_action_set

    common_v = np.intersect1d(r_idx, feasible_action)

    print(len(r_idx), len(feasible_action), len(common_v))

    input(" ")
    # p = RackStatePlot(goal_pattern=goal)
    # fig = p.plot_states([state.state, r]).get_img()
    # cv2.imshow("window", fig)
    # cv2.waitKey(0)

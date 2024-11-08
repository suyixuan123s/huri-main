import time
from env import to_action, from_action, RackState, RackArrangementEnv, from_action_mat
import numpy as np

a = time.time()
rack_size = (4, 4)
abs_id = 10
pick_id, place_id = from_action(rack_size, abs_id)
mat = from_action_mat(rack_size, abs_id)
print(mat)
print("pick, place", (pick_id, place_id))
# print("pick, place", to_action((2, 5), np.array([[0, 1]]), np.array([[1, 4]])))
# print("pick, place", to_abs_act((2, 2), 6))
b = time.time()
print((b - a) * 1000)
print("abs action id", to_action(rack_size, place_id, pick_id))

rack_size = (3, 3)

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
                         is_curriculum_lr=True,
                         seed=np.random.randint(999))

s = env.reset()
print([from_action(rack_size, i) for i in s.feasible_action_set])
print(s._cache[str(s.state)]["possible_actions"])
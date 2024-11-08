import time
from env import to_action, from_action, RackState, RackArrangementEnv
import numpy as np

a = time.time()
rack_size = (5, 10)
num_tube_classes = 2

observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(rack_size)
observation_space_dim_nn = (1, *rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=True,
                         is_goalpattern_fixed=True,
                         seed=888)

for i in range(100):
    stl, clsl = env.scheduler.update_training_level()
    print(f'state level: {stl}, class level: {clsl}')
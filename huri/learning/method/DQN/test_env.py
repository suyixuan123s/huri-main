if __name__ == "__main__":
    import numpy as np
    from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv

    # initialize the environment
    num_tube_classes = 3
    rack_size = (5, 10)
    action_space_dim = np.prod(rack_size) ** 2
    observation_space_dim = (1, *rack_size)
    env = RackArrangementEnv(rack_size=rack_size,
                             num_classes=num_tube_classes,
                             observation_space_dim=observation_space_dim,
                             action_space_dim=action_space_dim,
                             is_curriculum_lr=True,
                             is_goalpattern_fixed=True,
                             difficulty=12)
    env.goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                                 [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    s = env.reset()
    s = env.reset()
    s = env.reset()
    s = env.reset()
    print(f"number of tubes: {len(s[s > 0])}\n", s)

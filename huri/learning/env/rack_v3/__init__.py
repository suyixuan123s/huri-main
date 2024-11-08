def create_env(rack_sz,
               num_tube_class,
               seed=888,
               num_history=4,
               scheduler: 'GoalRackStateScheduler3' or 'GoalRackStateScheduler2' = None,
               toggle_curriculum=False,
               toggle_goal_fixed=False) -> 'RackArrangementEnv':
    from .env import RackArrangementEnv, RackState, RackStatePlot
    observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(tuple(rack_sz))
    return RackArrangementEnv(rack_size=tuple(rack_sz),
                              num_classes=num_tube_class,
                              num_history=num_history,
                              observation_space_dim=observation_space_dim,
                              action_space_dim=action_space_dim,
                              is_curriculum_lr=toggle_curriculum,
                              is_goalpattern_fixed=toggle_goal_fixed,
                              scheduler=scheduler,
                              seed=seed)


def create_env2(rack_sz,
                num_tube_class,
                seed=888,
                num_history=4,
                scheduler: 'GoalRackStateScheduler3' or 'GoalRackStateScheduler2' = None,
                toggle_curriculum=False,
                toggle_goal_fixed=False) -> 'RackArrangementEnv':
    from .env2 import RackArrangementEnv, RackState, RackStatePlot
    observation_space_dim, head_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(tuple(rack_sz))
    return RackArrangementEnv(rack_size=tuple(rack_sz),
                              num_classes=num_tube_class,
                              num_history=num_history,
                              observation_space_dim=observation_space_dim,
                              head_space_dim=head_space_dim,
                              action_space_dim=action_space_dim,
                              is_curriculum_lr=toggle_curriculum,
                              is_goalpattern_fixed=toggle_goal_fixed,
                              scheduler=scheduler,
                              seed=seed)

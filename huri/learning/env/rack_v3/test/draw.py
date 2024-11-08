from huri.learning.env.rack_v3.env import to_action, RackState, RackStatePlot, RackArrangementEnv
import numpy as np

import cv2

rack_size = (9, 9)
num_classes = 1
observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_goalpattern_fixed=False,
                         is_curriculum_lr=True,
                         seed=np.random.randint(999))
# goal = np.array([[0, 0, 3, 3, 0, 0, 2, 0, 3, 0],
#                  [1, 1, 0, 3, 2, 0, 2, 3, 2, 2],
#                  [0, 3, 3, 1, 1, 0, 0, 3, 0, 0],
#                  [0, 3, 1, 3, 0, 0, 0, 3, 0, 0],
#                  [0, 0, 3, 0, 0, 1, 2, 3, 0, 0]])
# state = np.array([[1, 3, 1, 0, 1, 0, 3, 0, 1, 0],
#                   [3, 0, 0, 0, 0, 3, 0, 0, 2, 0],
#                   [2, 2, 3, 0, 0, 2, 1, 0, 3, 0],
#                   [0, 2, 3, 0, 0, 2, 3, 3, 3, 3],
#                   [3, 0, 0, 0, 3, 0, 1, 0, 0, 0]])
# env.reset_state_goal(state, goal)
# for i in range(60):
#     nxt_state, reward, is_finished,_ = env.step(env.sample())
# print(repr(nxt_state.state))
#
# ss = np.array([[0, 3, 0, 0, 3, 0, 3, 3, 3, 3],
#        [1, 2, 0, 1, 0, 0, 1, 0, 2, 0],
#        [0, 2, 3, 1, 3, 0, 0, 0, 3, 0],
#        [2, 2, 3, 0, 0, 3, 2, 0, 0, 3],
#        [0, 0, 1, 0, 0, 0, 3, 0, 0, 1]])


# goal = np.array([[1, 1, 1, 1, 0, 0],
#                  [1, 1, 1, 0, 0, 0],
#                  [1, 1, 1, 0, 0, 0], ])
#
# state = np.array([[1, 1, 0, 1, 1, 1],
#               [0, 0, 1, 0, 1, 1],
#               [1, 0, 1, 0, 0, 1], ])
# env.scheduler.set_training_level(50)
# env.reset()
# goal = env.goal_pattern
# state = env.state

goal = np.array([[1, 1, 0, 0, 0, 1, 1, 1, 1],
                 [0, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 0, 0, 1, 1, 1, 0, 1],
                 [1, 0, 1, 0, 0, 1, 0, 1, 0],
                 [0, 1, 0, 0, 1, 0, 1, 1, 1],
                 [1, 0, 0, 1, 1, 0, 0, 1, 1],
                 [1, 1, 1, 0, 1, 0, 0, 0, 1],
                 [0, 1, 0, 1, 1, 1, 1, 0, 1],
                 [1, 0, 0, 1, 1, 0, 1, 1, 1]])
state = np.array([[0, 0, 0, 1, 1, 0, 1, 0, 1],
                  [0, 1, 0, 1, 1, 0, 0, 1, 0],
                  [1, 1, 1, 1, 0, 1, 0, 0, 1],
                  [1, 1, 0, 1, 1, 0, 1, 1, 1],
                  [0, 1, 0, 1, 0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 0, 1, 1],
                  [1, 1, 0, 1, 0, 1, 1, 0, 1],
                  [0, 1, 0, 1, 1, 1, 0, 1, 1],
                  [0, 1, 0, 1, 1, 1, 0, 0, 1]])


def get_chunk_slice(chunk_index, mini_window_size):
    row_start = chunk_index[0] * mini_window_size[0]
    row_end = row_start + mini_window_size[0]
    col_start = chunk_index[1] * mini_window_size[1]
    col_end = col_start + mini_window_size[1]
    return slice(row_start, row_end), slice(col_start, col_end)


fillable, movable = RackState(state)._cal_fillable_movable_slots()

goal = np.concatenate((goal[get_chunk_slice((0, 2), (3, 3))], goal[get_chunk_slice((2, 2), (3, 3))]), axis=1)
state = np.concatenate((state[get_chunk_slice((0, 2), (3, 3))], state[get_chunk_slice((2, 2), (3, 3))]), axis=1)
fconstraints = np.concatenate((fillable[get_chunk_slice((0, 2), (3, 3))], fillable[get_chunk_slice((2, 2), (3, 3))]), axis=1)
mconstraints = np.concatenate((movable[get_chunk_slice((0, 2), (3, 3))], movable[get_chunk_slice((2, 2), (3, 3))]), axis=1)

print(f'np.{repr(goal)}')
print(f'np.{repr(state)}')
# print(f'np.{repr(constraints)}')

# plot bg
rp = RackStatePlot(goal_pattern=goal)
img = rp.plot_states(rack_states=[state], row=1, img_scale=10,
                     fillable_mtx=[fconstraints],
                     movable_mtx=[mconstraints]).save_fig("bg.png", dpi=300)
# print(f'np.{repr(goal.state)}')
exit(0)
rp = RackStatePlot(goal_pattern=np.zeros((9, 9)))
img = rp.plot_states(rack_states=[state], row=1, img_scale=10).save_fig("state.png", dpi=300)

# print(f'np.{repr(state.state)}')

exit(0)

rp = RackStatePlot(goal_pattern=np.zeros((5, 10)))
img = rp.plot_states(rack_states=[state], row=1).save_fig("state.png", dpi=300)

rp = RackStatePlot(goal_pattern=np.zeros((5, 10)))
img = rp.plot_states(rack_states=[ss], row=1).save_fig("nstate.png", dpi=300)

rp = RackStatePlot(goal_pattern=ss)
img = rp.plot_states(rack_states=[np.zeros((5, 10))], row=1).save_fig("nbg.png", dpi=300)

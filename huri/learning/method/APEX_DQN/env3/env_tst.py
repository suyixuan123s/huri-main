from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState, RackStatePlot
import numpy as np
import cv2


def plot(states, goal, w_name="window"):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    cv2.imshow(w_name, fig)


num_tube_classes = 2
rack_size = (5, 10)
observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(rack_size)
observation_space_dim_nn = (1, *rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=True,
                         is_goalpattern_fixed=False,
                         seed=42)

env_n = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=False,
                         is_goalpattern_fixed=False,
                         seed=88)


if __name__ == "__main__":

    env1 = env.copy()
    print("env obs dim", env1.observation_space_dim)
    print("env act dim", env1.action_space_dim)
    env1.scheduler.state_level = 5
    env1.scheduler.class_level = 3
    while True:
        state = env1.reset()
        goal = env1.goal_pattern
        next_state, rew, _, _ = env1.step(env1.sample())
        # env1.action_between_states(state, goal)
        print("Reward is ", rew)
        plot([state, next_state], goal)
        # plot([np.flipud(state.state),np.flipud(next_state.state)],np.flipud(goal.state), w_name="2")
        # plot([np.fliplr(state.state), np.fliplr(next_state.state)], np.fliplr(goal.state), w_name="3")
        # plot([np.flipud(np.fliplr(state.state)), np.flipud(np.fliplr(next_state.state))], np.flipud(np.fliplr(goal.state)), w_name="4")
        cv2.waitKey(0)
        # plot([state, next_state], goal)
        print("?")

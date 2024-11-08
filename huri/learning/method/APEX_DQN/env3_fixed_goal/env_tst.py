from huri.learning.env.rack_v3.env import RackArrangementEnv, RackState, RackStatePlot
import numpy as np
import cv2
from dqn_model_cnn import DuelingDQNCNN, DuelingDQNCNN4, DuelingDQNCNN3, DuelingDQNCNN2
from hyperparameters import rack_size, num_tube_classes


# from huri.learning.method.APEX_DQN.env3.dqn_model_cnn_mini import DuelingDQNCNN, DuelingDQNCNN2, DuelingDQNCNN3


def plot(states, goal, w_name="window"):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states, row=15).get_img()
    cv2.imshow(w_name, fig)


observation_space_dim, action_space_dim = RackState.get_obs_act_dim_by_size(rack_size)
observation_space_dim_nn = (1, *rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=True,
                         is_goalpattern_fixed=True,
                         seed=888)
GOAL_PATTERN = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                         [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                         [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                         [1, 1, 0, 0, 0, 0, 0, 0, 3, 3],
                         [1, 1, 0, 0, 0, 0, 0, 0, 3, 3]])
env.set_goal_pattern(GOAL_PATTERN)


# env._get_reward = env._RackArrangementEnv__get_reward


def create_agent(obs_dim, act_dim, n_classes, device):
    return DuelingDQNCNN(obs_dim=obs_dim, action_dim=act_dim, num_classes=n_classes).to(device)


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
    env1.scheduler.state_level = 16
    env1.scheduler.class_level = 2
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

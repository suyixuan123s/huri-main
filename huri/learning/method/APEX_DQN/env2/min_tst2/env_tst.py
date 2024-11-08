from huri.learning.env.rack_v2.env import RackArrangementEnv, RackStatePlot
import numpy as np
import cv2


def plot(states, goal):
    drawer = RackStatePlot(goal)
    fig = drawer.plot_states(states).get_img()
    cv2.imshow("window", fig)
    cv2.waitKey(0)


num_tube_classes = 1
rack_size = (3, 3)
action_space_dim = np.prod(rack_size) ** 2
observation_space_dim = (rack_size[0], rack_size[1])
observation_space_dim_nn = (1, *rack_size)
env = RackArrangementEnv(rack_size=rack_size,
                         num_classes=num_tube_classes,
                         observation_space_dim=observation_space_dim,
                         action_space_dim=action_space_dim,
                         is_curriculum_lr=True,
                         is_goalpattern_fixed=False,
                         seed=42)
if __name__ == "__main__":

    env1 = env.copy()
    env1.scheduler.state_level = 16
    env1.scheduler.class_level = 3
    while True:
        state = env1.reset()
        goal = env1.goal_pattern
        plot([state], goal)
        print("?")

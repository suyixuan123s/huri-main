""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230627osaka

"""
import numpy as np


def compare_matrices(a, b):
    a = np.asarray(a.copy())
    b = np.asarray(b.copy())
    assert a.shape == b.shape
    if a.shape[0] != 1:
        a = a.flatten()
        b = b.flatten()
    return np.where(a != b, 1, 0)


if __name__ == '__main__':
    from huri.learning.env.rack_v3 import create_env, RackStatePlot
    import cv2

    env = create_env(rack_sz=(3, 3), num_tube_class=2, toggle_curriculum=False, )

    env.reset()
    env.reset()
    env.reset()
    env.reset()
    state = env.state
    goal = env.goal_pattern

    img = RackStatePlot(goal_pattern=goal).plot_states([state]).get_img()
    print("State: ")
    print(state)
    print("-" * 10)
    print("Goal: ")
    print(goal)
    print('-' * 10)
    print(compare_matrices(state, goal))
    cv2.imshow("img", img)
    cv2.waitKey(0)

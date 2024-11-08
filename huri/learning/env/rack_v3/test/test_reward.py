""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230726osaka

"""
from huri.learning.env.rack_v3.env import RackState, RackStatePlot, RackArrangementEnv
from huri.learning.env.rack_v3 import create_env
from huri.components.utils.matlibplot_utils import Plot
import cv2
import numpy as np
from PIL import Image
import io
import win32clipboard

def send_to_clipboard(image):
    output = io.BytesIO()
    image.convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]  # Remove the BMP header
    output.close()

    win32clipboard.OpenClipboard()  # Open the clipboard
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
    win32clipboard.CloseClipboard()

def on_mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:  # Right-click event
        # Convert the image from BGR to RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Save the image to a memory buffer
        send_to_clipboard(pil_image)

        # Notify the user
        print("Image copied to clipboard!")


if __name__ == '__main__':
    rack_size = (3, 6)
    num_classes = 2
    obs_dim, act_dim = RackState.get_obs_act_dim_by_size(rack_size)

    seed = 888
    env: RackArrangementEnv = create_env(rack_size, num_tube_class=num_classes, seed=seed, num_history=1,
                                         toggle_curriculum=False,
                                         toggle_goal_fixed=False,
                                         scheduler='GoalRackStateScheduler3', )
    env.scheduler.set_training_level(2)
    GOAL = np.array([[0, 1, 1, 0, 0],
                     [0, 0, 2, 1, 0],
                     [0, 0, 0, 0, 0], ])
    # env.set_goal_pattern(GOAL)

    a = env.action_between_states(np.array([[0, 1, 0, 0, 0],
                                            [0, 0, 0, 1, 0],
                                            [1, 2, 0, 0, 0], ]), np.array([[0, 1, 1, 0, 0],
                                                                           [0, 0, 0, 1, 0],
                                                                           [0, 2, 0, 0, 0], ]))

    a2 = env.action_between_states(np.array([[0, 1, 1, 0, 0],
                                             [0, 0, 0, 1, 0],
                                             [0, 2, 0, 0, 0], ]), np.array([[0, 1, 1, 0, 0],
                                                                            [0, 0, 0, 0, 0],
                                                                            [0, 2, 0, 1, 0], ]))

    # Create a named window
    cv2.namedWindow('image')

    # Bind the mouse callback function to the window
    cv2.setMouseCallback('image', on_mouse_click)

    for i in range(100000000000000000):
        print("\n" * 100)
        # state = env.reset_state_goal(np.array([[0, 1, 0, 0, 0],
        #                                        [0, 0, 0, 1, 0],
        #                                        [1, 2, 0, 0, 0], ]), GOAL)
        state = env.reset()
        # env.reset()
        goal_state = env.goal_pattern
        # state2, reward, done, _ = env.step(a)
        state2, reward, done, _ = env.step(env.sample(), True)
        # state3, reward2, done, _ = env.step(a2)
        state3, reward2, done, _ = env.step(env.sample(), True)

        # print(goal_state)
        # print(state)
        # print(state2)
        # print(state3)

        # print(f'Reward is {reward}, reward 2 is {reward2}')

        rsp = RackStatePlot(goal_state, plot=Plot(w=5, h=100, is_axis=False, dpi=300))
        img = rsp.plot_states([state, state2, state3], img_scale=4).get_img()
        # img = rsp.plot_states([np.zeros(rack_size)], img_scale=5).get_img()
        cv2.imshow(f"image", img)
        cv2.waitKey(0)

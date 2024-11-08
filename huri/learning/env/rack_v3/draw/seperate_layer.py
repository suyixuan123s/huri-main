""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231110osaka

"""
from huri.components.utils.img_utils import combine_images
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


img = None
# Create a named window
cv2.namedWindow('image')
# Bind the mouse callback function to the window
cv2.setMouseCallback('image', on_mouse_click)

if __name__ == '__main__':
    from huri.learning.env.rack_v3.env import to_action, RackState, RackStatePlot, RackArrangementEnv
    from huri.learning.method.APEX_DQN.distributed.utils import abs_state_np

    GOAL = np.array([[0, 1, 2, 0, 0, 0],
                     [0, 1, 2, 0, 0, 0],
                     [0, 2, 1, 0, 0, 0]])
    STATE = np.array([[0, 1, 0, 0, 0, 2],
                      [0, 2, 0, 2, 0, 0],
                      [0, 0, 1, 0, 0, 1]])

    nc = len(RackState(GOAL).num_classes)

    features = np.array(abs_state_np(state=STATE, goal=GOAL, env_classes=nc))

    rsp1 = RackStatePlot(goal_pattern=np.zeros_like(GOAL))
    rsp2 = RackStatePlot(goal_pattern=GOAL)
    st_1 = rsp2.plot_state(STATE, img_scale=5).get_img()
    st_2 = rsp1.plot_state(features[1][0], img_scale=5).get_img()
    g_1 = rsp1.plot_rack(features[1][1], img_scale=5).get_img()
    g_3 = rsp1.plot_state(features[0][2], img_scale=5).get_img()
    g_4 = rsp1.plot_state(features[0][3], img_scale=5).get_img()
    img = combine_images([st_1, st_2, g_1, g_3, g_4], toggle_sep_dash_line=False, columns=1)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    st_3 = rsp1.plot_state(features[1][0] * 2, img_scale=5).get_img()
    g_2 = rsp1.plot_rack(features[1][1]*2, img_scale=5).get_img()

    img = combine_images([st_1, st_2, g_1, st_3, g_2], toggle_sep_dash_line=False, columns=1)
    cv2.imshow('image', img)
    cv2.waitKey(0)

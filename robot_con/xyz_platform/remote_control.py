import time
import keyboard
import numpy as np

from robot_con.xyz_platform.xyz_platform import XYZPlatformController

xyz_con = XYZPlatformController(debug=False)
while True:
    pressed_keys = {'w': keyboard.is_pressed('w'),
                    'a': keyboard.is_pressed('a'),
                    's': keyboard.is_pressed('s'),
                    'd': keyboard.is_pressed('d'),
                    'r': keyboard.is_pressed('r'),  # x+ global
                    't': keyboard.is_pressed('t'),  # x- global
                    'f': keyboard.is_pressed('f'),  # y+ global
                    'g': keyboard.is_pressed('g'),  # y- global
                    'v': keyboard.is_pressed('v'),  # z+ global
                    'b': keyboard.is_pressed('b'),  # z- gglobal
                    'y': keyboard.is_pressed('y'),  # r+ global
                    'u': keyboard.is_pressed('u'),  # r- global
                    'h': keyboard.is_pressed('h'),  # p+ global
                    'j': keyboard.is_pressed('j'),  # p- global
                    'n': keyboard.is_pressed('n'),  # yaw+ global
                    'm': keyboard.is_pressed('m'),  # yaw- global
                    'o': keyboard.is_pressed('o'),  # gripper open
                    'p': keyboard.is_pressed('p')}  # gripper close
    values_list = list(pressed_keys.values())
    if pressed_keys["w"]:
        xyz_con.move_linear("y",.01,speed_n=1)
        time.sleep(.5)
    elif pressed_keys["s"] :
        xyz_con.move_linear("y",-.04,speed_n=1)
        time.sleep(.5)
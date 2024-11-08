""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230828osaka

"""
from typing import Literal, List
import numpy as np


class CalibrationSettings:
    work_arm: Literal['lft_arm', 'rgt_arm'] = 'lft_arm'
    led_power = 10 # it should a int value be among [0 to 4095]
    work_range_x: List[int] = [300 / 1000, 360 / 1000, 400 / 1000]
    work_range_y: List[int] = [-200 / 1000, 0, 200 / 1000]
    work_range_z: List[int] = [130 / 1000, 200 / 1000, 300 / 1000]


class RgtArmInfo:
    arm = 'rgt_arm'
    init_jnt_val = np.array([0.99291781, -1.52541777, -1.52925749, 0.3122394, -0.33946654,
                             1.15313904, 0.7162831267948966])
    action_center_rotmat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T
    action_center_pos = np.array([.30, -0.13, .23])


class LftArmInfo:
    arm = 'lft_arm'
    init_jnt_val = np.array([-0.99693207, -1.52541777, 1.52925749, 0.3122394, 0.33946654,
                             1.15313904, -0.7161085967948966])
    action_center_rotmat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]).T
    action_center_pos = np.array([.30, 0.13, .23])

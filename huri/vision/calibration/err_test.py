from time import strftime

import numpy as np

from huri.core.common_import import *
from huri.core.constants import SENSOR_INFO
import vision.depth_camera.sensor_marker_handler as dcsmh
from utils import CalibrationSettings, RgtArmInfo, LftArmInfo

if __name__ == "__main__":
    '''initilization'''
    from huri.components.yumi_control.yumi_con import YumiController

    # base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    robot_x = YumiController()  # robot control handler

    sensor_handler = dcsmh.SensorMarkerHandler2(SENSOR_INFO.IP_ADR_DEPTH_SENSOR, aruco_target_id_list=[0, 1],
                                                toggle_cv_show=True)
    # sensor_handler.sensor_client.changeledpower(CalibrationSettings.led_power)
    ArmInfo = RgtArmInfo if CalibrationSettings.work_arm == 'rgt_arm' else LftArmInfo
    robot_x.move_jnts(component_name=ArmInfo.arm,
                      jnt_vals=ArmInfo.init_jnt_val)

    pos, rot, conf = robot_x.get_pose(ArmInfo.arm, return_conf=True)
    zero_pose = np.average([sensor_handler.get_marker_center(), sensor_handler.get_marker_center(), sensor_handler.get_marker_center()], axis=0)
    print(zero_pose)

    offset_dis = .04
    x_pos = np.array([offset_dis, 0, 0]) + pos
    x_pose_list = []
    robot_x.move_p(component_name=ArmInfo.arm, pos=x_pos, rot=rot, conf=conf)
    x_pose = sensor_handler.get_marker_center() - zero_pose
    x_pose_list.append(np.linalg.norm(x_pose))
    x_pose = sensor_handler.get_marker_center() - zero_pose
    x_pose_list.append(np.linalg.norm(x_pose))
    x_pose = sensor_handler.get_marker_center() - zero_pose
    x_pose_list.append(np.linalg.norm(x_pose))
    print("Average Error in X axis", (np.average(x_pose_list) - offset_dis) * 10e3)
    print(x_pose_list)
    print("No average Error in X axis", [(x - offset_dis) * 10e3 for x in x_pose_list])

    y_pos = np.array([0, offset_dis, 0]) + pos
    y_pose_list = []
    robot_x.move_p(component_name=ArmInfo.arm, pos=y_pos, rot=rot, conf=conf)
    y_pose = sensor_handler.get_marker_center() - zero_pose
    y_pose_list.append(np.linalg.norm(y_pose))
    y_pose = sensor_handler.get_marker_center() - zero_pose
    y_pose_list.append(np.linalg.norm(y_pose))
    y_pose = sensor_handler.get_marker_center() - zero_pose
    y_pose_list.append(np.linalg.norm(y_pose))
    print("Average Error in Y axis", (np.average(y_pose_list) - offset_dis) * 10e3)
    print(y_pose_list)
    print("No average Error in Y axis", [(y - offset_dis) * 10e3 for y in y_pose_list])

    z_pos = np.array([0, 0, offset_dis]) + pos
    z_pose_list = []
    robot_x.move_p(component_name=ArmInfo.arm, pos=z_pos, rot=rot, conf=conf)
    z_pose = sensor_handler.get_marker_center() - zero_pose
    z_pose_list.append(np.linalg.norm(z_pose))
    z_pose = sensor_handler.get_marker_center() - zero_pose
    z_pose_list.append(np.linalg.norm(z_pose))
    z_pose = sensor_handler.get_marker_center() - zero_pose
    z_pose_list.append(np.linalg.norm(z_pose))
    print("Average Error in Z axis", (np.average(z_pose_list) - offset_dis) * 10e3)
    print(z_pose_list)
    print("No average Error in Z axis", [(z - offset_dis) * 10e3 for z in z_pose_list])

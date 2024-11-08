"""
Author: Hao Chen (chen960216@gmail.com 20221113)
The program to manually calibrate the camera
"""
__VERSION__ = '0.0.1'

import os
from pathlib import Path
import json

import numpy as np

from robot_sim.robots.robot_interface import RobotInterface


def py2json_data_formatter(data):
    """Format the python data to json format. Only support for np.ndarray, str, int, float ,dict, list"""
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, str) or isinstance(data, float) or isinstance(data, int) or isinstance(data, dict):
        return data
    elif isinstance(data, Path):
        return str(data)
    elif isinstance(data, list):
        return [py2json_data_formatter(d) for d in data]


def dump_json(data, path="", reminder=True):
    path = str(path)
    """Dump the data by json"""
    if reminder and os.path.exists(path):
        option = input(f"File {path} exists. Are you sure to write it, y/n: ")
        print(option)
        option_up = option.upper()
        if option_up == "Y" or option_up == "YES":
            pass
        else:
            return False
    with open(path, "w") as f:
        json.dump(py2json_data_formatter(data), f)
    return True


class PlotRT(object):
    def __init__(self, rbt_s: RobotInterface, rbt_x, sensor_hdl, init_calib_mat_1: np.ndarray = None,
                 init_calib_mat_2: np.ndarray = None,
                 component_name="arm", ):
        """
        Class to manually calibrate the point cloud data
        :param rbt_s: The simulation robot
        :param rbt_x: The real robot handler
        :param sensor_hdl: The sensor handler
        :param init_calib_mat: The initial calibration matrix. If it is None, the init calibration matrix will be identity matrix
        :param component_name: component name that mounted the camera
        :param move_resolution: the resolution for manual move adjustment
        :param rotation_resolution: the resolution for manual rotation adjustment
        """
        self._rbt_s = rbt_s
        self._rbt_x = rbt_x
        self._sensor_hdl = sensor_hdl
        self._init_calib_mat_1 = np.eye(4) if init_calib_mat_1 is None else init_calib_mat_1
        self._init_calib_mat_2 = np.eye(4) if init_calib_mat_2 is None else init_calib_mat_2
        self._component_name = component_name

        # variable stores robot plot and the point cloud plot
        self._plot_node_rbt = None
        self._plot_node_pcd = None
        self._pcd_1 = None
        self._pcd_2 = None

        # add task
        taskMgr.doMethodLater(.05, self.sync_rbt, "sync rbt", )
        taskMgr.doMethodLater(.5, self.sync_pcd, "sync pcd", )

    def get_pcd_1(self):
        pcd, pcd_color, _, _ = self._sensor_hdl.get_pcd_texture_depth(device_id=1)
        return np.hstack((pcd, pcd_color))

    def get_pcd_2(self):
        pcd, pcd_color, _, _ = self._sensor_hdl.get_pcd_texture_depth(device_id=2)
        return np.hstack((pcd, pcd_color))

    def get_rbt_jnt_val(self):
        return self._rbt_x.get_jnt_values()

    def sync_pcd(self, task):
        """
        Synchronize the real robot and the simulation robot
        :return: None
        """

        self._pcd_1 = self.get_pcd_1()
        self._pcd_2 = self.get_pcd_2()
        self.plot()
        return task.again

    def sync_rbt(self, task):
        rbt_jnt_val = self.get_rbt_jnt_val()
        self._rbt_s.fk(self._component_name, rbt_jnt_val)
        self.plot()
        return task.again

    def align_pcd_1(self, pcd):
        r2cam_mat = TCP2EYE_MAT_DEVICE_1
        rbt_pose = self._rbt_x.get_pose()
        w2r_mat = rm.homomat_from_posrot(*rbt_pose)
        w2c_mat = w2r_mat.dot(r2cam_mat)
        return rm.homomat_transform_points(w2c_mat, points=pcd)

    def align_pcd_2(self, pcd):
        r2cam_mat = TCP2EYE_MAT_DEVICE_2
        rbt_pose = self._rbt_x.get_pose()
        w2r_mat = rm.homomat_from_posrot(*rbt_pose)
        w2c_mat = w2r_mat.dot(r2cam_mat)
        return rm.homomat_transform_points(w2c_mat, points=pcd)

    def plot(self, task=None):
        """
        A task to plot the point cloud and the robot
        :param task:
        :return:
        """
        # clear previous plot
        if self._plot_node_rbt is not None:
            self._plot_node_rbt.detach()
        if self._plot_node_pcd is not None:
            self._plot_node_pcd.detach()
        self._plot_node_rbt = self._rbt_s.gen_meshmodel()
        self._plot_node_rbt.attach_to(base)
        pcd_1 = self._pcd_1
        pcd_2 = self._pcd_2
        if pcd_1 is not None and pcd_2 is not None:
            if pcd_1.shape[1] == 6:
                pcd_1, pcd_color_1 = pcd_1[:, :3], pcd_1[:, 3:6]
                pcd_color_rgba_1 = np.append(pcd_color_1, np.ones((len(pcd_color_1), 1)), axis=1)
            else:
                pcd_color_rgba_1 = np.array([1, 1, 1, 1])
            pcd_r_1 = self.align_pcd_1(pcd_1)
            self._plot_node_pcd = gm.gen_pointcloud(pcd_r_1, rgbas=pcd_color_rgba_1)
            self._plot_node_pcd.attach_to(base)

            if pcd_2.shape[1] == 6:
                pcd_2, pcd_color_2 = pcd_2[:, :3], pcd_2[:, 3:6]
                pcd_color_rgba_2 = np.append(pcd_color_2, np.ones((len(pcd_color_2), 1)), axis=1)
            else:
                pcd_color_rgba_2 = np.array([1, 1, 1, 1])
            pcd_r_2 = self.align_pcd_2(pcd_2)
            gm.gen_pointcloud(pcd_r_2, rgbas=pcd_color_rgba_2).attach_to(self._plot_node_pcd)

        if task is not None:
            return task.again


if __name__ == "__main__":
    import numpy as np

    array = np.array
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    from dual_d405_manager import RealSenseD405Dual
    import basis.robot_math as rm
    from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
    from robot_sim.robots.xarm_lite6_wrs import XArmLite6WRSGripper2
    from constants import TCP2EYE_MAT_DEVICE_1, TCP2EYE_MAT_DEVICE_2

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405Dual()
    # the first frame contains no data information
    rs_pipe.get_pcd_texture_depth(device_id=1)
    rs_pipe.get_pcd_texture_depth(device_id=1)
    rs_pipe.get_pcd_texture_depth(device_id=2)
    rs_pipe.get_pcd_texture_depth(device_id=2)
    rbtx = XArmLite6X(ip='192.168.1.190', has_gripper=False, tcp_offset=np.array([0, 0, 0.175, 0, 0, 0]))
    rbt = XArmLite6WRSGripper2()

    gm.gen_frame(pos=[.06, 0, 0]).attach_to(base)

    # xarm_mc = XArmLite6ManualCalib(rbt_s=rbt, rbt_x=rbtx, sensor_hdl=rs_pipe, init_calib_mat=mat,
    #                                rotation_resolution=np.radians(20), move_resolution=.001)
    xarm_mc = PlotRT(rbt_s=rbt, rbt_x=rbtx, sensor_hdl=rs_pipe, init_calib_mat_1=TCP2EYE_MAT_DEVICE_1,
                     init_calib_mat_2=TCP2EYE_MAT_DEVICE_2)
    base.run()

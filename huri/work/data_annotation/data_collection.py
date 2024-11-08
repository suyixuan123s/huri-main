__VERSION__ = '0.0.1'

from typing import List
import cv2
import numpy as np
from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from robot_sim.manipulators.xarm_lite6 import XArmLite6
from drivers.devices.realsense_d405.d405_driver import RealSenseD405
from huri.core.common_import import fs, wd, gm, rm


def combination(c_list: List[np.ndarray]) -> np.ndarray:
    c_list_len = len(c_list)
    comb = np.array(np.meshgrid(*c_list)).T.reshape(-1, c_list_len)
    return comb


class RobotMover:
    """
    Move the robot to indicated waypoint
    Author: Chen Hao
    Email: chen960216@gmail.com
    Date: 20220313
    """

    def __init__(self, rbt_s, rbt_con=None):
        """
        :param rbt_s: robot_sim.robots.yumi.Yumi
        :param rbt_con: huri.component.control.yumi_con.YumiController
        """
        # simulation robot, robot controller, rrt-connect planner
        self._rbt_s = rbt_s
        self._rbt_con = rbt_con

        # waypoints set for rgt and lft arm; waypoint counter
        self._wp = []
        self._wp_ctr = 0

    def add_wp_homomats(self, homomats: List[np.ndarray], ):
        """
        Add waypoint in form of homogeneous matrix. Waypoints will be restored in self._wp_rgt/self._wp_lft
        :param homomats: waypoints in form of homogenous matrix list (np.array with 4x4 shape)
        :param armname: indicate the arm for waypoints
        :param load: True: load previous generated waypoints, if the previous data does not exit, it will generate new data
                     False: generate new data
        """

        self._wp.extend(homomats)

    def _set_wp_ctr(self, val: int, ):
        """
        Set counter of waypoints
        """
        self._wp_ctr = val

    def _get_wp_ctr(self, ) -> int:
        """
        Get counter of waypoints
        """
        return self._wp_ctr

    def goto_next_wp(self, speed=100):
        """
        Move robot to the next waypoint
        :param armname: Arm name of the robot
        """
        if self._rbt_con is None:
            wp_list = self._wp
            wp_jnt_val = wp_list[self._get_wp_ctr()]
            self._set_wp_ctr(self._get_wp_ctr() + 1, )
            self._rbt_s.fk(wp_jnt_val)
            return

        wp_list = self._wp

        if self._get_wp_ctr() >= len(wp_list):
            print(len(wp_list))
            raise StopIteration

        wp_homomat = wp_list[self._get_wp_ctr()]
        # increase waypoint counter

        wp_pos = wp_homomat[:3, 3]
        wp_rot = wp_homomat[:3, :3]

        self._set_wp_ctr(self._get_wp_ctr() + 1, )
        self._rbt_con.move_p(pos=wp_pos, rot=wp_rot, speed=speed)


if __name__ == "__main__":
    import time

    base = wd.World(cam_pos=[0, 0, 1.5], lookat_pos=[0, 0, 0])
    rs_pipe = RealSenseD405()
    rbt = XArmLite6(enable_cc=True)
    rbtx = XArmLite6X()
    # goto home configuration
    rbtx.homeconf()
    gm.gen_frame().attach_to(base)
    rbt.fk(rbtx.get_jnt_values())
    rbt.gen_meshmodel().attach_to(base)

    rm_x = RobotMover(rbt_s=rbt, rbt_con=rbtx)
    rbtx.homeconf()

    origin_pos = np.array([0])
    origin_rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    wp_homomats = []

    for pos in combination([np.array([.28]),
                            np.array([0, ]),
                            np.array(np.random.randint(18, 20) / 100)]):
        rots_candidate = np.array(rm.gen_icorotmats(icolevel=3,
                                                    rotation_interval=np.radians(360),
                                                    crop_normal=np.array([0, 0, 1]),
                                                    crop_angle=np.radians(15),
                                                    toggleflat=True))
        for rot in rots_candidate:
            wp_homomats.append(
                rm.homomat_from_posrot(pos, np.dot(rm.rotmat_from_axangle(np.array([0, 0, 1]), np.radians(45)), rot)))
    rm_x.add_wp_homomats(homomats=wp_homomats)
    # img_save_dir = fs.Path("C:\\Users\\WRS\\Desktop\\img20220927")
    img_save_dir = fs.Path(f"data_{time.strftime('%Y%m%d%H%M%S')}")
    if not img_save_dir.exists():
        img_save_dir.mkdir()
    from huri.vision.pipeline import vision_pipeline_realsense
    from huri.core.constants import ANNOTATION_0_0_2

    try:
        while True:
            rm_x.goto_next_wp(speed=200)
            j = rbtx.get_jnt_values()
            p, r = rbtx.get_pose()
            pcd, pcd_color, depth_img, color_img = vision_pipeline_realsense(rs_pipe, )

            cv2.imwrite(f"{str(img_save_dir)}/{time.strftime('%Y%m%d%H%M%S')}.jpg", color_img)
            # fs.dump_pickle(tuple(ANNOTATION_0_0_2.IN_HAND_ANNOTATION_SAVE_FORMAT(
            #     version=ANNOTATION_0_0_2.VERSION,
            #     pcd=pcd,
            #     pcd_color=pcd_color,
            #     gray_img=None,
            #     color_img=color_img,
            #     depth_img=depth_img,
            #     rbt_tcp_pos=p,
            #     rbt_tcp_rot=r,
            #     rbt_joints=j,
            #     annotations=None)), img_save_dir.joinpath(
            #     f"{time.strftime('%Y%m%d%H%M%S')}.pkl"), reminder=False)

            cv2.imshow("captured imgs", color_img)
            cv2.waitKey(10)

    except StopIteration:
        print("finished")

    base.run()

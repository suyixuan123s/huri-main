import time
import numpy as np

from robot_con.yumi.autolab_core import RigidTransform
import robot_con.yumi.yumi_robot as yr
import robot_con.yumi.yumi_state as ys
import motion.probabilistic.rrt_connect as rrtc
from motion.trajectory.piecewisepoly import PiecewisePoly


class YumiController:
    """
    A client to control the yumi
    """

    def __init__(self, debug=False):
        self.rbtx = yr.YuMiRobot(debug=debug)
        self._is_add_all = True
        self._traj_opt = PiecewisePoly()

    @property
    def lft_arm_hnd(self):
        return self.rbtx.left

    @property
    def rgt_arm_hnd(self):
        return self.rbtx.right

    def get_pose(self, component_name, return_conf=False):
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm', 'rgt_arm']!")
        pose = armx.get_pose()
        pos = pose.translation
        rot = pose.rotation
        rot[:, 0] = - rot[:, 0]
        rot[:, 1] = - rot[:, 1]
        if not return_conf:
            return pos, rot
        return pos, rot, pose.configuration

    def move_jnts(self, component_name, jnt_vals, speed_n=100, wait=True):
        """
        move one arm joints of the yumi
        :param component_name
        :param jnt_vals: 1x7 np.array
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf

        :return: bool

        author: weiwei
        date: 20170411
        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm', 'rgt_arm']!")
        if speed_n == -1:
            armx.set_speed_max()
        else:
            speed_data = self.rbtx.get_v(speed_n)
            armx.set_speed(speed_data)

        armjnts = np.rad2deg(jnt_vals)
        ajstate = ys.YuMiState(armjnts)
        armx.movetstate_sgl(ajstate, wait_for_res=wait)

    def move_sync(self, lft_jnt_vals, rgt_jnt_vals, speed_n=100):
        # TODO this function is not working now
        self.rbtx.set_v(speed_n)

        armjnts_lft = np.rad2deg(lft_jnt_vals)
        ajstate_lft = ys.YuMiState(armjnts_lft)
        armjnts_rgt = np.rad2deg(rgt_jnt_vals)
        ajstate_rgt = ys.YuMiState(armjnts_rgt)
        self.rbtx.goto_state_sync(left_state=ajstate_lft, right_state=ajstate_rgt)

    def fk(self, component_name: str, jnt_vals: np.ndarray, return_conf: bool = False) -> tuple:
        """
        Forward kinematics of YUMI calculated by the RAPID commands in the real robot
        :param component_name
        :param jnt_vals: 1x7 np.array
        :param return_conf: If True, additionally return robot configuration and external axis value
        :param return_ext_axis: If True, additionally return external axis value
        :return: If return_conf and return_ext_axis is False: 1x3 position, 3x3 rotation
                 If return_conf is True: 1x3 position, 3x3 rotation, 1x4 robot configuration, float external axis value

        Also see 1176 page of https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf
        for robot configuration
        author: hao chen (chen960216@gmail.com)
        date: 20230829
        """
        assert jnt_vals is not None and jnt_vals.shape == (7,), "jnt_vals must be a 1x7 np.ndarray!"
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm', 'rgt_arm']!")
        armjnts = np.rad2deg(jnt_vals)
        ajstate = ys.YuMiState(armjnts)
        pose = armx.fk(ajstate)
        pos = pose.translation
        rot = pose.rotation
        rot[:, 0] = - rot[:, 0]
        rot[:, 1] = - rot[:, 1]
        if not return_conf:
            return pos, rot
        return pos, rot, pose.configuration, np.deg2rad(pose.ext_axis)

    def ik(self, component_name: str,
           pos: np.ndarray,
           rot: np.ndarray,
           conf: np.ndarray = None,
           ext_axis: float = None) -> np.ndarray or None:
        """
        Inverse kinematics of YUMI calculated by the RAPID commands in the real robot
        :param component_name: str, "lft_arm" or "rgt_arm"
        :param pos: 1x3 position
        :param rot: 3x3 rotation
        :param conf: 1x4 robot configuration
        :param ext_axis: float, External axes offset. See `robtarget` in RAPID manual for more details.
                        If None, the robot will use the default value specified in RAPID.
        :return: 1x7 joint values
        Also see 1176 page of https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf
        for robot configuration
        author: hao chen 20230909
        """
        _rot = rot.copy()
        _rot[:, 0] = - _rot[:, 0]
        _rot[:, 1] = - _rot[:, 1]
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm', 'rgt_arm']!")
        _pose = RigidTransform(rotation=_rot, translation=pos, configuration=conf)
        armjnts = armx.ik(_pose, ext_axis=np.rad2deg(ext_axis) if ext_axis is not None else None)
        if armjnts is None:
            return None
        return np.deg2rad(armjnts.joints)

    def move_p(self, component_name: str,
               pos: np.ndarray,
               rot: np.ndarray,
               conf: np.ndarray = None,
               ext_axis: float = None,
               linear: bool = True,
               speed_n: int = 100):
        """
        Move to aa pose
        :param component_name: str, "lft_arm" or "rgt_arm"
        :param pos: 1x3 position
        :param rot: 3x3 rotation
        :param conf: 1x4 robot configuration
        :param ext_axis: float, External axes offset. See `robtarget` in RAPID manual for more details.
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
        :return:
        """
        assert isinstance(pos, np.ndarray) and pos.shape == (3,), "pos must be a 1x3 np.ndarray!"
        assert isinstance(rot, np.ndarray) and rot.shape == (3, 3), "rot must be a 3x3 np.ndarray!"
        assert isinstance(conf, np.ndarray) or conf.shape == (4,), "conf must be a 1x4 np.ndarray!"
        assert isinstance(ext_axis, float) or ext_axis is None, "ext_axis must be a float!"
        assert isinstance(linear, bool), "linear must be a bool!"
        assert isinstance(speed_n, int), "speed_n must be a int!"

        _rot = rot.copy()
        _rot[:, 0] = - _rot[:, 0]
        _rot[:, 1] = - _rot[:, 1]
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm', 'rgt_arm']!")
        if speed_n == -1:
            armx.set_speed_max()
        else:
            speed_data = self.rbtx.get_v(speed_n)
            armx.set_speed(speed_data)
        _pose = RigidTransform(rotation=_rot, translation=pos, configuration=conf, )
        res = armx.goto_pose(pose=_pose, linear=linear, ext_axis=np.rad2deg(ext_axis) if ext_axis is not None else None,
                             wait_for_res=True)

    def contactL(self, component_name, jnt_vals, desired_torque=.5):
        if component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['rgt_arm']!")
        armjnts = np.rad2deg(jnt_vals)
        ajstate = ys.YuMiState(armjnts)
        return armx.contactL(ajstate, desired_torque)

    def get_jnt_values(self, component_name):
        """
        get the joint angles of both arms
        :return: 1x6 array
        author: chen
        """
        if component_name == "all":
            lftjnts = self._get_arm_jnts("lft")
            rgtjnts = self._get_arm_jnts("rgt")
            return np.array(lftjnts + rgtjnts)
        elif component_name in ["lft_arm", "lft_hnd"]:
            return self._get_arm_jnts("lft")
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            return self._get_arm_jnts("rgt")
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")

    def move_jntspace_path(self, component_name, path, speed_n=100, wait=True) -> bool:
        """
        :param speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf

        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        statelist = []
        st = time.time()
        for armjnts in self._traj_opt.interpolate_path(path, num=min(100, int(len(path)))):
            armjnts = np.rad2deg(armjnts)
            ajstate = ys.YuMiState(armjnts)
            statelist.append(ajstate)
        et = time.time()
        print("time calculating sending information", et - st)
        # set the speed of the robot
        if speed_n == -1:
            armx.set_speed_max()
        else:
            speed_data = self.rbtx.get_v(speed_n)
            armx.set_speed(speed_data)
        exec_result = armx.movetstate_cont(statelist, is_add_all=self._is_add_all, wait_for_res=wait)
        return exec_result

    def calibrate_gripper(self):
        """
        Calibrate the gripper
        :param speed : float, optional
            Max speed of the gripper in mm/s.
            Defaults to 10 mm/s. If None, will use maximum speed in RAPID.
        :param force : float, optional
            Hold force used by the gripper in N.
            Defaults to 10 N. If None, will use maximum force the gripper can provide (20N).
        """
        self.rgt_arm_hnd.calibrate_gripper()
        self.lft_arm_hnd.calibrate_gripper()

    def __set_gripper_force(self, component_name, force=10):
        """
        TODO: this program has bug. Fix it later.
        :param force: Hold force by the gripper in Newton.
        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.set_gripper_force(force=force)

    def set_gripper_speed(self, component_name, speed=10):
        """
        :param speed: In mm/s.
        """
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.set_gripper_max_speed(max_speed=speed)

    def set_gripper_width(self, component_name, width):
        """
        Moves the gripper to the given width in meters.
        width : float
                Target width in meters, range[0 , 0.025]
                if you want to fully close or fully open the gripper,
                please use the open_gripper or close_gripper!!
                Otherwise the program may stuck
        """
        assert 0 <= width < yr.YMC.MAX_GRIPPER_WIDTH
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.move_gripper(width=width / 2)

    def open_gripper(self, component_name):
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.open_gripper()

    def close_gripper(self, component_name, force=10):
        assert 0 <= force <= yr.YMC.MAX_GRIPPER_FORCE
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.close_gripper(force=force)

    def set_gripper_force(self, component_name, force=10):
        assert 0 <= force <= yr.YMC.MAX_GRIPPER_FORCE
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        armx.set_gripper_force(force=force)

    def get_gripper_width(self, component_name):
        if component_name in ["lft_arm", "lft_hnd"]:
            armx = self.rbtx.left
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            armx = self.rbtx.right
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        return armx.get_gripper_width() * 2

    def _get_arm_jnts(self, armname):
        if armname == "rgt":
            return np.deg2rad(self.rbtx.right.get_state().joints)
        elif armname == "lft":
            return np.deg2rad(self.rbtx.left.get_state().joints)
        else:
            raise ValueError("Arm name must be right or left!")

    def get_hc_img(self, armname):
        if armname == "rgt":
            self.rbtx.right.write_handcamimg_ftp()
        elif armname == "lft":
            self.rbtx.left.write_handcamimg_ftp()
        else:
            raise ValueError("Arm name must be right or left!")

    def toggle_vac(self, toggletag, armname):
        if armname == "rgt":
            self.rbtx.right.toggle_vacuum(toggletag)
        elif armname == "lft":
            self.rbtx.left.toggle_vacuum(toggletag)

    def get_pressure(self, armname):
        if armname == "rgt":
            return self.rbtx.right.get_pressure()
        elif armname == "lft":
            return self.rbtx.left.get_pressure()

    def go_zero_pose(self):
        self.rbtx.reset_home()

    def stop(self):
        self.rbtx.stop()


def move_rrt(yumi_s: 'robot_sim.robots.yumi.yumi.Yumi',
             yumi_x: YumiController,
             goal_conf: np.ndarray,
             component_name="rgt_arm",
             speed_n=300,
             obstacle_list=None,
             wait=True):
    # initialize the module for RRT
    rrtc_planner = rrtc.RRTConnect(yumi_s)
    if obstacle_list is None:
        obstacle_list = []
    # the left and right arm go initial pose
    rrt_path = None
    if component_name in ["rgt_arm", "rgt_hnd"]:
        rrt_path = rrtc_planner.plan(component_name="rgt_arm",
                                     start_conf=np.array(yumi_x.get_jnt_values("rgt_arm")),
                                     goal_conf=goal_conf,
                                     obstacle_list=obstacle_list,
                                     ext_dist=.01,
                                     max_time=300)
        if len(rrt_path) > 5:
            yumi_x.move_jntspace_path(component_name="rgt_arm", path=rrt_path, speed_n=speed_n, wait=wait)
        else:
            yumi_x.move_jnts(component_name="rgt_arm", jnt_vals=yumi_s.rgt_arm.homeconf, speed_n=speed_n)
    if component_name in ["lft_arm", "lft_hnd"]:
        rrt_path = rrtc_planner.plan(component_name="lft_arm",
                                     start_conf=np.array(yumi_x.get_jnt_values("lft_arm")),
                                     goal_conf=goal_conf,
                                     obstacle_list=obstacle_list,
                                     ext_dist=.01,
                                     max_time=300)
        if len(rrt_path) > 5:
            yumi_x.move_jntspace_path(component_name="lft_arm", path=rrt_path, speed_n=speed_n, wait=wait)
        else:
            yumi_x.move_jnts(component_name="lft_arm", jnt_vals=yumi_s.lft_arm.homeconf, speed_n=speed_n, )
    return rrt_path


def to_homeconf(yumi_s: 'robot_sim.robots.yumi.yumi.Yumi', yumi_x: YumiController, component_name="rgt_arm",
                method="RRT", speed_n=300, wait=True):
    """
    make robot go to init position
    :param yumi_s: Yumi instamce
    :param yumi_x: YumiController instance
    :param component_name: rgt_arm, lft_arm or both. Indicates the arm to go to home configuration
    :param method: rrt
    :param speed_n: -1: full speed, speed number. If speed_n = 100, then speed will be set to the corresponding v100
                specified in RAPID. Loosely, n is translational speed in milimeters per second
                Please refer to page 1186 of
                https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf
    :return:
    """
    if method.lower() == "rrt":
        # initialize the module for RRT
        rrtc_planner = rrtc.RRTConnect(yumi_s)
        # the left and right arm go initial pose
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            rrt_path_rgt = rrtc_planner.plan(component_name="rgt_arm",
                                             start_conf=np.array(yumi_x.get_jnt_values("rgt_arm")),
                                             goal_conf=np.array(yumi_s.rgt_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            if len(rrt_path_rgt) > 5:
                yumi_x.move_jntspace_path(component_name="rgt_arm", path=rrt_path_rgt, speed_n=speed_n, wait=wait)
            else:
                yumi_x.move_jnts(component_name="rgt_arm", jnt_vals=yumi_s.rgt_arm.homeconf, speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            rrt_path_lft = rrtc_planner.plan(component_name="lft_arm",
                                             start_conf=np.array(yumi_x.get_jnt_values("lft_arm")),
                                             goal_conf=np.array(yumi_s.lft_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            if len(rrt_path_lft) > 5:
                yumi_x.move_jntspace_path(component_name="lft_arm", path=rrt_path_lft, speed_n=speed_n, wait=wait)
            else:
                yumi_x.move_jnts(component_name="lft_arm", jnt_vals=yumi_s.lft_arm.homeconf, speed_n=speed_n, )
    else:
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            yumi_x.move_jnts(component_name="rgt_arm", jnt_vals=yumi_s.rgt_arm.homeconf, speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            yumi_x.move_jnts(component_name="lft_arm", jnt_vals=yumi_s.lft_arm.homeconf, speed_n=speed_n)


if __name__ == "__main__":
    yumi_con = YumiController(debug=False)
    # yumi_con.rbtx.calibrate_grippers()
    # print("jnt angle is: ", repr(yumi_con.get_jnt_values("rgt_arm")))
    # yumi_con.calibrate_gripper()
    # yumi_con.set_gripper_force("rgt_arm", 10)
    # yumi_con.set_gripper_width('rgt_arm', 0.02)
    # yumi_con.set_gripper_width(component_name='rgt_arm', width=0.02)
    # yumi_con.open_gripper(component_name='rgt_arm')
    # print(yumi_con.get_gripper_width('rgt_arm'))
    # exit(0)
    # ycc.set_gripper_speed("rgt_arm", 10)

    # ycc.open_gripper("rgt_hnd")
    # a = ycc.get_gripper_width("rgt_hnd")
    # print(a)
    # ycc.close_gripper("rgt_hnd", force=5)
    # a = ycc.get_gripper_width("rgt_hnd")
    # print(a)
    from robot_sim.robots.yumi.yumi import Yumi

    # ycc.move_jnts("rgt_arm", np.array([0.97544, -0.82503, -1.1102, 0.92439, 1.2511, -0.98685, -1.9185]),
    #               )
    yumi_s = Yumi()
    yumi_s.fk("rgt_arm", yumi_con.get_jnt_values("rgt_arm"))
    simulation_rbt = yumi_s.get_gl_tcp("rgt_arm")
    print(simulation_rbt)
    ik_r = yumi_s.ik("rgt_arm", simulation_rbt[0], simulation_rbt[1], seed_jnt_values=yumi_con.get_jnt_values("rgt_arm"))
    print("angle_diff:", ik_r - yumi_con.get_jnt_values("rgt_arm"))
    print("angle diff norm", np.linalg.norm(ik_r - yumi_con.get_jnt_values("rgt_arm")))
    exit(0)

    print(simulation_rbt)
    print(yumi_con.get_pose("rgt_arm"))
    # array([ 0.33997068, -0.1192723 ,  0.09096477])
    pos = np.array([0.33997068, -0.1192723, 0.09096477])
    rot = np.array([[0.96168357, -0.26688774, -0.06273468],
                    [-0.27368392, -0.94804125, -0.16221867],
                    [-0.01618089, 0.1731725, -0.98475858]])
    # ycc.move_l("rgt_arm", pos=pos, rot=rot)
    # print(ycc.get_pose("rgt_arm"))

    # print(np.array([1.5536, -0.63633, -1.9201, 0.6659, 0.93347, -1.0767, 1.9627]))
    # ycc.calibrate_gripper()
    # print(ycc.get_jnt_values("rgt_arm"))
    # ycc.set_gripper_speed("rgt_arm", 10)

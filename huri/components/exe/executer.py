import time
from typing import Union, Optional
import logging

from huri.core.common_import import np
from huri.components.yumi_control.yumi_con import YumiController
from huri.definitions.utils_structure import MotionBatchPPP, MotionElement, MotionBatch
from motion.trajectory.piecewisepoly import PiecewisePoly
import motion.probabilistic.rrt_connect as rrtc
from huri.math.math import perpendicular_vector
from robot_sim.robots.yumi.yumi import Yumi
import basis.robot_math as rm

# Constants
SUCC = 0
INSERT_FAILURE_ERR = 10  # fail to insert the test tube into the rack
INSERT_FAILURE_TUBE_LOSING_ERR = 11

GRASP_COLLISION_ERR = 20  # collision when grasping the test tube
GRASP_NO_TUBE_ERR = 21  # not grasp the test tube

UNKNOWN_ERR = 9999  # unknow error


class MotionExeSingleArm(YumiController):
    GRIPPER_OPEN = 0
    GRIPPER_CLOSE = 1
    GRIPPER_MIDDLE = 2

    def __init__(self,
                 component_name,
                 max_open_gripper_len=0.05,
                 toggle_debug=False,
                 logger=logging.getLogger(__name__)):
        super(MotionExeSingleArm, self).__init__(debug=toggle_debug)
        self.rbt = Yumi()

        self.component_name = component_name
        self.max_open_gripper_len = max_open_gripper_len
        # 0 is open, 1 is close
        self._grasp_force = 14
        self._grasp_speed = 7
        self._min_grasp_dis = .005
        self.close_gripper_delay = 0.5
        self.set_gripper_speed(component_name, self._grasp_speed)

        self._place_contactL_torque = .012

        self.logger_name = "motoion_exe"
        self.logger = logger

        # if the joint configuration distance between the current pose and the first waypoint of a path
        # is larger than the safe margin, the robot will move to the first waypoint by move_jnts first
        self.safe_exe_margin = np.pi / 20
        self.safe_exe_speed = 100

        self.traj_opt = PiecewisePoly()

        self.logger.info(f"[{self.logger_name}] Initialized. "
                         f"Gripper state: {self.gripper_state}")

        self.failed_goback_pose = {
            'rgt_arm': np.array([-0.34906585, -1.57079633, -2.0943951, 0.52359878, 0.,
                                 0.6981317, 0.]),
            'rgt_hnd': np.array([-0.34906585, -1.57079633, -2.0943951, 0.52359878, 0.,
                                 0.6981317, 0.]),
            'lft_arm': np.array([0.34906585, -1.57079633, 2.0943951, 0.52359878, 0.,
                                 0.6981317, 0.]),
            'lft_hnd': np.array([0.34906585, -1.57079633, 2.0943951, 0.52359878, 0.,
                                 0.6981317, 0.])
        }

    def _check_gripper_state(self):
        gripper_len = self.get_gripper_width(self.component_name)
        print(gripper_len)
        if gripper_len >= self.max_open_gripper_len:
            return self.GRIPPER_OPEN
        elif gripper_len <= self._min_grasp_dis:
            return self.GRIPPER_CLOSE
        else:
            return self.GRIPPER_MIDDLE

    @property
    def gripper_state(self):
        return self._check_gripper_state()

    def open_gripper(self, component_name=None):
        if component_name is None:
            component_name = self.component_name
            if self.gripper_state != self.GRIPPER_OPEN:
                super().set_gripper_width(component_name, self.max_open_gripper_len)
                self.logger.info(f"[{self.logger_name}] Gripper opens successfully")
                # if component_name == self.component_name:
                # self.gripper_state = self.GRIPPER_OPEN
            else:
                self.logger.warning(f"[{self.logger_name}] Gripper has already opened")
        else:
            super().set_gripper_width(component_name, self.max_open_gripper_len)

    def close_gripper(self, component_name=None):
        if component_name is None:
            component_name = self.component_name
            if self.gripper_state != self.GRIPPER_CLOSE and self.gripper_state != self.GRIPPER_MIDDLE:
                super().close_gripper(component_name, force=self._grasp_force)
                self.logger.info(f"[{self.logger_name}] Gripper closes successfully")
                # if component_name == self.component_name:
                # self.gripper_state = self.GRIPPER_CLOSE
                time.sleep(self.close_gripper_delay)
            else:
                self.logger.warning(f"[{self.logger_name}] Gripper has already closed")
        else:
            super().close_gripper(component_name, force=self._grasp_force)
            time.sleep(self.close_gripper_delay)

    def acccurate_ik_refine(self, conf_list, component_name=None):
        if component_name is None:
            component_name = self.component_name
        new_conf_list = []
        # recalculate the conf
        for jnt_val in conf_list:
            jnt_val_np = np.asarray(jnt_val)
            self.rbt.fk(component_name, jnt_val_np)
            pos, rot = self.rbt.get_gl_tcp(component_name)
            _, _, conf, ext_axis = self.fk(component_name, jnt_val_np, return_conf=True)
            # solve accurate ik using ik function on the real robot
            accurate_jnt_val = self.ik(component_name, pos, rot, conf, ext_axis=ext_axis)
            new_conf_list.append(accurate_jnt_val)
        print(new_conf_list)
        return new_conf_list

    def refine_traj(self, conf_list, component_name=None):
        if component_name is None:
            component_name = self.component_name
        new_conf_list = []

        self.rbt.fk(component_name, conf_list[-1])
        pos, rot = self.rbt.get_gl_tcp(component_name)
        pos_real, _, conf, ext_axis = self.fk(component_name, conf_list[-1], return_conf=True)
        # recalculate the conf
        diff = np.linalg.norm(pos - pos_real) > (8 / 1000)
        print("REAL POSITION AND FK IN SIMULATION DIFFERENCE IS,", diff)
        if diff:
            for jnt_val in conf_list:
                jnt_val_np = np.asarray(jnt_val)
                self.rbt.fk(component_name, jnt_val_np)
                pos, rot = self.rbt.get_gl_tcp(component_name)
                _, _, conf, ext_axis = self.fk(component_name, jnt_val_np, return_conf=True)
                # solve accurate ik using ik function on the real robot
                accurate_jnt_val = self.ik(component_name, pos, rot, conf, ext_axis=ext_axis)
                new_conf_list.append(accurate_jnt_val)
        return new_conf_list

    def exe_motion_element(self,
                           motion_element: MotionElement,
                           compensate_motion_insert_before: list = None,
                           compensate_motion_insert_after: list = None,
                           speed_n=300,
                           component_name=None,
                           toggle_real_pos=False):
        if component_name is None:
            component_name = self.component_name
        if compensate_motion_insert_before is None:
            compensate_motion_insert_before = []
        if compensate_motion_insert_after is None:
            compensate_motion_insert_after = []
        safe_exe_margin = self.safe_exe_margin
        if isinstance(motion_element, MotionElement):
            hndname = motion_element.hnd_name
            conf_list = compensate_motion_insert_before + motion_element.conf_list + compensate_motion_insert_after
        elif isinstance(motion_element, list) or isinstance(motion_element, np.ndarray):
            hndname = component_name
            conf_list = compensate_motion_insert_before + motion_element + compensate_motion_insert_after
        else:
            raise Exception("Not support motion element")
        if conf_list is None:
            self.logger.warning(f"[{self.logger_name}] Execute the empty motion")
            return

        if toggle_real_pos:
            new_conf_list = []
            # recalculate the conf
            for jnt_val in conf_list:
                jnt_val_np = np.asarray(jnt_val)
                self.rbt.fk(component_name, jnt_val_np)
                pos, rot = self.rbt.get_gl_tcp(component_name)
                _, _, conf, ext_axis = self.fk(component_name, jnt_val_np, return_conf=True)
                # solve accurate ik using ik function on the real robot
                accurate_jnt_val = self.ik(component_name, pos, rot, conf, ext_axis=ext_axis)
                new_conf_list.append(accurate_jnt_val)
            conf_list = new_conf_list

        # safe motion
        if np.linalg.norm(self.get_jnt_values(component_name=hndname) - conf_list[0]) > safe_exe_margin:
            self.move_jnts(component_name=hndname, jnt_vals=conf_list[0], speed_n=self.safe_exe_speed)
        if isinstance(motion_element, MotionElement):
            jawwidth_list = motion_element.jawwidth_list
            if jawwidth_list[0] >= 0.05:
                self.open_gripper()
            if jawwidth_list[0] < 0.05:
                self.close_gripper()
        exec_result = self.move_jntspace_path(path=conf_list, component_name=hndname, speed_n=speed_n)
        return exec_result
        # self.logger.info(f"Motion execute {'successfully' if exec_result else 'failed'}")

    def exe_pick_and_place_motion(self,
                                  motion_batch: MotionBatchPPP,
                                  speed_n_place=100,
                                  is_only_place=False,
                                  toggle_pick_place_path_refine=False):
        if toggle_pick_place_path_refine:
            motion_batch.pick_approach.conf_list = self.acccurate_ik_refine(
                conf_list=motion_batch.pick_approach.conf_list, )
            motion_batch.pick_depart.conf_list = self.acccurate_ik_refine(
                conf_list=motion_batch.pick_depart.conf_list, )
            motion_batch.place_approach.conf_list = self.acccurate_ik_refine(
                conf_list=motion_batch.place_approach.conf_list, )

        if not is_only_place:
            # execute the to pick motion
            # self.exe_motion_element(motion_batch.to_pick)
            self.exe_motion_element(
                self.traj_opt.interpolate_path(motion_batch.to_pick.conf_list), speed_n=speed_n_place)
            self.logger.info(f"[{self.logger_name}] Go to pick the test tube")

            # execute the pick motion
            # self.exe_motion_element(motion_batch.pick_approach[:-2])
            self.exe_motion_element(motion_batch.pick_approach, speed_n=150)  # speed_n=90
            # exe_result = self.contactL(self.component_name,
            #                            jnt_vals=motion_batch.pick_approach.conf_list[0])
            exe_result = True
            if not exe_result:
                self.logger.warning(f"[{self.logger_name}] Gripper collision when grasping the test tube")
                self.exe_motion_element(motion_batch.pick_approach[::-1],
                                        compensate_motion_insert_before=[self.get_jnt_values(self.component_name)],
                                        compensate_motion_insert_after=motion_batch.to_pick[::-1].conf_list)
                return False, GRASP_COLLISION_ERR
            self.logger.info(f"[{self.logger_name}] Pick the test tube")

            # execute the pick depart
            # exe_result = self.contactL(self.component_name,
            #                            jnt_vals=motion_batch.pick_depart.conf_list[0])
            # self.exe_motion_element(motion_batch.pick_depart,
            #                         compensate_motion_insert_before=[self.get_jnt_values(self.component_name)])
            # self.exe_motion_element(motion_batch.pick_depart, )
            self.close_gripper(self.component_name)
            # check if grasp the object when picking
            if self._check_gripper_state() == self.GRIPPER_CLOSE:
                self.open_gripper(self.component_name)
                self._check_gripper_state()
                # self.exe_motion_element(self.traj_opt.interpolate_path(motion_batch.pick_approach[::-1].conf_list),
                #                         compensate_motion_insert_before=[self.get_jnt_values(self.component_name)],
                #                         compensate_motion_insert_after=self.traj_opt.interpolate_path(
                #                             motion_batch.to_pick[::-1].conf_list))
                self.exe_motion_element(self.traj_opt.interpolate_path(motion_batch.pick_approach[::-1].conf_list),
                                        compensate_motion_insert_before=[self.get_jnt_values(self.component_name)], )
                return False, GRASP_NO_TUBE_ERR
            self.exe_motion_element(motion_batch.pick_depart, speed_n=speed_n_place)  # speed_n=250

            # check if grasp the object after departure
            if self._check_gripper_state() == self.GRIPPER_CLOSE:
                self.open_gripper(self.component_name)
                self._check_gripper_state()
                # self.exe_motion_element([],
                #                         compensate_motion_insert_before=[self.get_jnt_values(self.component_name)], )
                return False, GRASP_NO_TUBE_ERR

            self.logger.info(f"[{self.logger_name}] Pick motion finished")

            # execute the to place
            self.exe_motion_element(motion_batch.to_place, speed_n=speed_n_place)  # speed_n=300
            self.logger.info(f"[{self.logger_name}] Go to place the test tube")

        # execute the place approach
        # motion_batch.place_approach.conf_list = self.refine_traj(motion_batch.place_approach.conf_list)

        self.rbt.fk(self.component_name, motion_batch.place_approach.conf_list[-1])
        pos, rot = self.rbt.get_gl_tcp(self.component_name)
        pos_real, _, conf, ext_axis = self.fk(self.component_name, motion_batch.place_approach.conf_list[-1],
                                              return_conf=True)
        diff = np.linalg.norm(pos - pos_real)
        print("REAL POSITION AND FK IN SIMULATION DIFFERENCE IS,", diff)
        super().set_gripper_force(self.component_name, force=4)
        exe_result = self.exe_motion_element(motion_batch.place_approach, speed_n=150)  # speed_n=250
        # self.exe_motion_element(motion_batch.place_approach, speed_n=speed_n_place)
        # exe_result = self.contactL(self.component_name, jnt_vals=motion_batch.place_approach.conf_list[-1],
        #                            desired_torque=self._place_contactL_torque)
        if not exe_result:
            self.logger.warning(f"[{self.logger_name}] Failed to insert the test tube into hole")
            self.move_jnts(component_name=self.component_name, jnt_vals=motion_batch.place_depart.conf_list[-1],
                           speed_n=self.safe_exe_speed)
            # self.exe_motion_element([], compensate_motion_insert_before=[self.get_jnt_values(self.component_name)],
            #                         compensate_motion_insert_after=[motion_batch.place_depart.conf_list[-1], ])
            if self._check_gripper_state() == self.GRIPPER_CLOSE:
                self.logger.warning(f"[{self.logger_name}] FATAL: test tube is missing from hand!!")
                return False, INSERT_FAILURE_TUBE_LOSING_ERR
            return False, INSERT_FAILURE_ERR
        self.logger.info(f"[{self.logger_name}] Place the test tube")

        if motion_batch.to_end is None or motion_batch.to_end.conf_list is None:
            final_path = None
        else:
            final_path = self.traj_opt.interpolate_path(motion_batch.to_end.conf_list, )
        # execute the place depart and to end motion
        # motion_batch.place_depart.conf_list = self.refine_traj(motion_batch.place_depart.conf_list)
        self.exe_motion_element(motion_batch.place_depart,
                                compensate_motion_insert_before=[self.get_jnt_values(self.component_name)],
                                compensate_motion_insert_after=final_path,
                                speed_n=speed_n_place)

        # self.exe_motion_element(motion_batch.to_end)
        self.logger.info(f"[{self.logger_name}] Pick and place motion finished!")

        return True, SUCC


# save check


def motion_executer(yumi_s,
                    yumi_con: Optional[YumiController],
                    motion_batch: MotionBatch,
                    open_gripper_len=0.05,
                    speed_n=100,
                    logger=logging.getLogger(__name__)):
    logger_name = "motion_executer"
    if yumi_con is None:
        print("-------------- no real robot -----------")
        return None
    is_closed = [False]
    is_refined_motion = [False, None]

    def open_gripper():
        if is_closed[0]:
            yumi_con.set_gripper_width(component_name=hndname, width=open_gripper_len)
            time.sleep(.5)
            is_closed[0] = False

    def close_gripper():
        if not is_closed[0]:
            yumi_con.close_gripper(component_name=hndname)
            time.sleep(.5)
            is_closed[0] = True

    for motion_name, motion_element in motion_batch:
        is_exe_success = True
        hndname = motion_element.hnd_name
        conf_list = motion_element.conf_list
        jawwidth_list = motion_element.jawwidth_list
        if is_refined_motion[0]:
            is_refined_motion[0] = False
            conf_list = [is_refined_motion[1]] + conf_list
            is_refined_motion[1] = None
        if np.linalg.norm(yumi_con.get_jnt_values(component_name=hndname) - conf_list[0]) > np.pi / 20:
            yumi_con.move_jnts(component_name=hndname, jnt_vals=conf_list[0], speed_n=speed_n)
        if jawwidth_list[0] >= 0.05:
            open_gripper()
        if jawwidth_list[0] < 0.05:
            close_gripper()
        if motion_name == "place_approach":
            speed_n = 100
        else:
            # if motion_name == "start2pickapproach" or motion_name == "placedepart2goal" or motion_name == "placedepart":
            speed_n = -1
        logger.info(f"[{logger_name}] execute the motion at speed: v{'max' if speed_n == -1 else speed_n}")
        if motion_name == "pick_approach" or motion_name == "place_approach":
            # if motion_name == "place_approach":
            yumi_con.move_jntspace_path(path=conf_list[:-1], component_name=hndname, speed_n=speed_n)
            yumi_con.contactL(hndname, jnt_vals=conf_list[-1])
            open_gripper()
            is_refined_motion[0] = True
            is_refined_motion[1] = yumi_con.get_jnt_values(hndname)
            print(f"the rgt arm joints are: {repr(yumi_con.get_jnt_values(hndname))}")
        else:
            # interpolated_confs, interpolated_spds, interpolated_accs, interpolated_x = \
            #     trajopt_gen.interpolate(conf_list,control_frequency=.8)
            # print(interpolated_confs)
            yumi_con.move_jntspace_path(path=conf_list, component_name=hndname, speed_n=speed_n)
            print(motion_name)
        if is_exe_success:
            logger.info(f"[{logger_name}] execute the motion segment {motion_name} successfully!")
        else:
            logger.info(f"[{logger_name}] execute the motion segment {motion_name} Failed, Error name!")

            # if motion_name == "pick_approach":
            #     yumi_s.fk(hndname, yumi_con.get_jnt_values(hndname))
            #     yumi_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
            #     pose_rgt_raw = yumi_con.rgt_arm_hnd.get_pose()
            #     print("real tcp pose", pose_rgt_raw.translation, pose_rgt_raw.rotation)
            #     print(yumi_s.get_gl_tcp(hndname))
            #     base.run()


from huri.math.fkopt_based_ik import FKOptBasedIK_C, FKOptBasedIK

def feedback_insertion_error(yumi_s,
                             yumi_con: MotionExeSingleArm,
                             tube,
                             tube_pose,
                             pp_planner,
                             obstacle_list=[],
                             toggle_return_motion=False):
    component_name = yumi_con.component_name
    if "rgt" in component_name:
        component_name = "rgt_hnd"
    else:
        component_name = "lft_hnd"
    recover_arm_name = "rgt_hnd"
    if "rgt" in component_name:
        recover_arm_name = "lft_hnd"

    if toggle_return_motion:
        return_false = [False, None, None, None]
    else:
        return_false = False

    component_arm_sim_jnt_val_bk = yumi_s.get_jnt_values(component_name=component_name)
    recover_arm_sim_jnt_val_bk = yumi_s.get_jnt_values(recover_arm_name)

    current_jnt = yumi_con.get_jnt_values(component_name)
    yumi_s.fk(component_name=component_name, jnt_values=current_jnt)
    rbt_pos, rbt_rot = yumi_s.get_gl_tcp(component_name)
    if rbt_pos[1] < -.05:
        rbt_pos[1] = 0
    rbt_pos[0] = .3
    recover_jnt_component_arm = yumi_s.ik(component_name, rbt_pos, rbt_rot, current_jnt)
    if recover_jnt_component_arm is None:
        print("recover failed")
        return return_false
    v = (rm.angle_between_vectors(rbt_rot[:, 0], np.array([0, 1, 0])),
            rm.angle_between_vectors(rbt_rot[:, 0], np.array([0, -1, 0])))
    angle_id = np.argmin(v)
    vector = [np.array([0, 1, 0]), np.array([0, -1, 0])][angle_id]
    angle = v[angle_id]
    if angle > np.pi / 7:
        if np.cross(rbt_rot[:, 0], vector)[2] > 0:
            sign = -1
        else:
            sign = 1
        recover_jnt_component_arm[-1] = recover_jnt_component_arm[-1] + sign* angle
        yumi_s.fk(component_name, recover_jnt_component_arm)
        rbt_pos, rbt_rot = yumi_s.get_gl_tcp(component_name)
    # prepare one hand
    yumi_s.fk(component_name, component_arm_sim_jnt_val_bk)
    # get tube relative pose
    rel_tube_pos = np.linalg.inv(rm.homomat_from_posrot(*yumi_s.get_gl_tcp(component_name))).dot(tube_pose)
    tube_cm = tube.gen_collision_model()
    tube_cm.set_homomat(tube_pose)
    yumi_s.hold(component_name, objcm=tube_cm)
    rrtc_planner = pp_planner.rrtc_planner
    yumi_s.fk(component_name, recover_jnt_component_arm)
    rrt_path_component_arm = rrtc_planner.plan(component_name=component_name,
                                               start_conf=np.array(current_jnt),
                                               goal_conf=np.array(recover_jnt_component_arm),
                                               obstacle_list=obstacle_list,
                                               ext_dist=.04,
                                               max_time=300)
    yumi_s.release(component_name, objcm=tube_cm)
    is_rrt_path = True
    if len(rrt_path_component_arm) < 1:
        is_rrt_path = False
        yumi_s.fk(component_name=component_name, jnt_values=current_jnt)
        rbt_pos, rbt_rot = yumi_s.get_gl_tcp(component_name)
    else:
        yumi_con.exe_motion_element(rrt_path_component_arm, component_name=component_name)
    # prepare other hand
    oik_solver = FKOptBasedIK_C(yumi_s,
                                obstacle_list=obstacle_list + [tube_cm],
                                component_name=recover_arm_name,
                                toggle_debug=False, )

    rbt_rot_recovery = np.eye(3)
    rbt_rot_recovery[:3, 0] = np.cross(perpendicular_vector(rbt_rot[:3, 0]), rbt_rot[:3, 0])
    rbt_pos_recovery = rbt_pos.copy()
    rbt_pos_recovery[2] = rbt_pos_recovery[2] - tube.height / 2
    # tube_pos_new = rm.homomat_from_posrot(rbt_pos, rbt_rot).dot(rel_tube_pos)
    # rbt_pos_recovery = tube_pos_new[:3,3].copy()
    # rbt_pos_recovery = rbt_pos_recovery + tube.height / 2 * tube_pos_new[:3,2]
    # rbt_pos_recovery[2] = 0 - .01

    seed_jnt_value = np.array([-0.99693207, -1.52541777, 1.52925749, 0.3122394, 0.33946654,
                               1.15313904, 0.85468773])
    ik_other = oik_solver.solve(tgt_pos=rbt_pos_recovery,
                                tgt_rotmat=rbt_rot_recovery,
                                seed_jnt_values=seed_jnt_value, )  # recover_arm_sim_jnt_val_bk
    if ik_other[0] is None:
        seed_jnt_value = np.array([-0.69115038, -1.57079633, 1.89211144, 0.67666415, -0.02129302,
                                   0.73094389, 1.23813657])
        ik_other = oik_solver.solve(tgt_pos=rbt_pos_recovery,
                                    tgt_rotmat=rbt_rot_recovery,
                                    seed_jnt_values=seed_jnt_value, )  # recover_arm_sim_jnt_val_bk

    if ik_other[0] is None:
        ik_other = oik_solver.solve(tgt_pos=rbt_pos_recovery,
                                    tgt_rotmat=-rbt_rot_recovery,
                                    seed_jnt_values=seed_jnt_value, )
        if ik_other[0] is None:
            print("ik other failed!!!!!!!!!!!!!!!")
            return return_false
    # yumi_con.move_jntspace_path(component_name, rrt_path_rgt, speed_n=-1)
    yumi_s.fk(recover_arm_name, ik_other[0])
    recover_arm_pos, recover_arm_rot = yumi_s.get_gl_tcp(recover_arm_name)
    recover_arm_pos[1] = rbt_pos[1]
    recover_arm_pos = recover_arm_pos + 0.01 * recover_arm_rot[:3, 2]
    to_recover_motion, approach_recover_motion = pp_planner.gen_approach_motion(component_name=recover_arm_name,
                                                                                goal_tcp_pos=recover_arm_pos,
                                                                                goal_tcp_rotmat=recover_arm_rot,
                                                                                start_conf=recover_arm_sim_jnt_val_bk,
                                                                                approach_direction=recover_arm_rot[:3,
                                                                                                   2],
                                                                                approach_distance=0.04,
                                                                                approach_jawwidth=.5,
                                                                                granularity=tube.height / 6,
                                                                                obstacle_list=obstacle_list,
                                                                                object_list=[],
                                                                                seed_jnt_values=np.array(ik_other[0]),
                                                                                max_time=600)
    if to_recover_motion is None:
        print("to motion failed++++++++++++++++++++++++++++")
        raise Exception
        return return_false
    # component arm move to recovery pos

    # recovery arm move to recovery pos
    yumi_con.exe_motion_element(to_recover_motion.conf_ls, component_name=recover_arm_name)
    yumi_con.set_gripper_width(recover_arm_name, 0.049)
    yumi_con.exe_motion_element(approach_recover_motion.conf_ls, component_name=recover_arm_name)
    yumi_con.close_gripper(recover_arm_name)
    # yumi_con.open_gripper()
    # yumi_con.close_gripper(recover_arm_name)
    # yumi_con.close_gripper()
    yumi_con.set_gripper_width(recover_arm_name, 0.049)
    yumi_con.exe_motion_element(approach_recover_motion.conf_ls[::-1], component_name=recover_arm_name)
    yumi_con.exe_motion_element(to_recover_motion.conf_ls[::-1], component_name=recover_arm_name)
    # component arm move back to origin pos

    if is_rrt_path:
        yumi_con.exe_motion_element(rrt_path_component_arm[::-1])

    yumi_s.fk(component_name=component_name, jnt_values=component_arm_sim_jnt_val_bk)
    yumi_s.fk(component_name=recover_arm_name, jnt_values=recover_arm_sim_jnt_val_bk)
    if not toggle_return_motion:
        return True
    else:
        return True, rrt_path_component_arm, to_recover_motion, approach_recover_motion


if __name__ == "__main__":
    from exe_loggining import exe_logger
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_con = YumiController()
    yumi_con.component_name = 'rgt_arm'
    yumi_robot = ym.Yumi(enable_cc=True)


    class tube:
        height = .1


    feedback_insertion_error(yumi_robot, yumi_con, tube, None, None,
                             )

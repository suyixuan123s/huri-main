import logging
import time

import numpy as np
import manipulation.approach_depart_planner as adp
from huri.components.debug.inspect import save_rbt_info, show_animation, save_error_info, IKError, RbtCollisionError, \
    RbtHandCollisionError
from huri.components.planning.utils import _REQ_PACKET
from huri.components.planning.symmetric_pick_and_place_palnner_c_fund import PickPlacePlanner_CR, motion_seg_avaliable, \
    rm, _MOTION_SEG, motion_seg_comb, motion_seg_to_element
from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper
import huri.core.file_sys as fs
from huri.components.exe.version.grasp_filter import is_grasp_feasible


class CommonGraspIterator:
    def __init__(self,
                 robot_s,
                 hnd,
                 grasp_info_list,
                 hand_name,
                 obj_init_homomat: np.array,
                 sym_obj_tgt_homomats: np.array,
                 pick_approach_pos_offset: np.array,
                 place_approach_pos_offset: np.array,
                 obstacle_list: list,
                 feasible_ranges_rad: list,
                 rack_tf: np.ndarray,
                 toggle_debug=False):
        self.robot_s = robot_s
        self.hnd = hnd
        self.grasp_info_list = grasp_info_list
        # hand_name
        self.hand_name = hand_name
        self.obj_init_homomat = obj_init_homomat
        self.sym_obj_tgt_homomats = sym_obj_tgt_homomats
        self.pick_approach_pos_offset = pick_approach_pos_offset
        self.place_approach_pos_offset = place_approach_pos_offset
        self.obstacle_list = obstacle_list

        self.feasible_ranges_rad = feasible_ranges_rad
        self.rack_tf_inv = np.linalg.inv(rack_tf)

        # record error
        self.ik_failed_init_num = 0
        self.approach_ik_failed_init_num = 0
        self.collision_init_num = 0
        self.apporach_collision_init_num = 0
        self.ik_failed_tgt_num = 0
        self.approach_ik_failed_tgt_num = 0
        self.collision_tgt_num = 0
        self.apporach_collision_tgt_num = 0

        self.toggle_debug = toggle_debug

        self._counter = 0

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        if self._counter >= len(self.grasp_info_list):
            raise StopIteration
        r = self.check_grasps(self._counter)
        self._counter += 1
        return r

    def check_grasps(self, grasp_id) -> (bool, np.array, int, list, list):
        hand_name = self.hand_name
        obj_init_homomat = self.obj_init_homomat
        sym_obj_tgt_homomats = self.sym_obj_tgt_homomats
        pick_approach_pos_offset = self.pick_approach_pos_offset
        place_approach_pos_offset = self.place_approach_pos_offset
        obstacle_list = self.obstacle_list
        # init pos and rotmat
        init_pos = obj_init_homomat[:3, 3]
        init_rotmat = obj_init_homomat[:3, :3]
        # bacakup init jnt angle and jawwidth
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)
        jawwidth_bk = self.robot_s.get_jawwidth(hand_name)
        # self.robot_s.fk("rgt_arm", np.array([0.t, -1.52541777, -1.52925749, 0.3122394, -0.33946654,
        #                                      1.15313904, -0.8545132]))
        # self.robot_s.fk("lft_arm", np.array([-0.99693207, -1.52541777, 1.52925749, 0.3122394, 0.33946654,
        #                                      1.15313904, 0.85468773]))
        # if grasp is feasible at init
        is_init_feasible, init_jnt_val, init_approach_jnt_val, init_jaw_width \
            = self._check_grasp_feasible_init(grasp_id=grasp_id,
                                              hand_name=hand_name,
                                              init_pos=init_pos,
                                              init_rotmat=init_rotmat,
                                              approach_pos_offset=pick_approach_pos_offset,
                                              obstacle_list=obstacle_list,
                                              toggle_debug=self.toggle_debug)
        if self.toggle_debug:
            print(f" Number of feasible IK at the pick pose: {init_jnt_val}")
        if not is_init_feasible:
            self.robot_s.fk(hand_name, jnt_values_bk)
            self.robot_s.jaw_to(hand_name, jawwidth_bk)
            return False, None, None, None, None
        # if grasp is feasible at grasp
        is_tgt_feasible, tgt_jnt_val_list, tgt_approach_jnt_val_list, tgt_jawwidth_list \
            = self._check_grasp_feasible_symmetric_tgt(grasp_id=grasp_id,
                                                       hand_name=hand_name,
                                                       init_ik=init_jnt_val,
                                                       sym_obj_tgt_homomats=sym_obj_tgt_homomats,
                                                       approach_pos_offset=place_approach_pos_offset,
                                                       obstacle_list=obstacle_list,
                                                       toggle_debug=self.toggle_debug)
        if self.toggle_debug:
            print(
                f" Number of feasible IK at the place pose: {len(tgt_jnt_val_list) if tgt_jnt_val_list is not None else 0}")
        if not is_tgt_feasible:
            self.robot_s.fk(hand_name, jnt_values_bk)
            self.robot_s.jaw_to(hand_name, jawwidth_bk)
            return False, None, None, None, None

        self.robot_s.fk(hand_name, jnt_values_bk)
        self.robot_s.jaw_to(hand_name, jawwidth_bk)
        return True, init_jnt_val, init_jaw_width, tgt_jnt_val_list, tgt_jawwidth_list

    def _check_grasp_feasible_init(self,
                                   grasp_id,
                                   hand_name,
                                   init_pos: np.array,
                                   init_rotmat: np.array,
                                   approach_pos_offset: np.array,
                                   obstacle_list=[],
                                   toggle_debug=False) -> (bool, np.array, np.array, int):
        # a variable to store grasp info list
        grasp_info_list = self.grasp_info_list
        robot = self.robot_s
        hnd = self.hnd
        jaw_width_offset = .015
        # grasp info
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info_list[grasp_id]
        # robot ee pose
        init_jaw_center_pos = init_pos + init_rotmat.dot(jaw_center_pos)
        # robot ee rot
        init_jaw_center_rotmat = init_rotmat.dot(jaw_center_rotmat)
        # robot ee approach pose
        if isinstance(approach_pos_offset, float):
            init_jaw_center_approach_pos = init_jaw_center_pos - approach_pos_offset * init_jaw_center_rotmat[:3, 2]
        else:
            len_approach_pos_offset = np.linalg.norm(approach_pos_offset)
            approach_pos_dir = approach_pos_offset / len_approach_pos_offset
            init_jaw_center_approach_pos = init_jaw_center_pos - (
                    approach_pos_dir + init_jaw_center_rotmat[:3, 2]) * len_approach_pos_offset
        # grasp at some pose
        # jaw_width_v = jaw_width + jaw_width_offset
        jaw_width_v = .034
        robot.jaw_to(hand_name, jaw_width_v)
        # check collision feasible
        hnd.grip_at_with_jcpose(gl_jaw_center_pos=init_jaw_center_pos,
                                gl_jaw_center_rotmat=init_jaw_center_rotmat,
                                # jaw_width=jaw_width + jaw_width_offset
                                jaw_width=jaw_width_v,
                                )
        is_hnd_collision = hnd.is_collided(obstacle_list)
        if is_hnd_collision:
            self.collision_init_num += 1
            if toggle_debug:
                save_error_info(RbtHandCollisionError(hnd, gl_jaw_center_pos=init_jaw_center_pos,
                                                      gl_jaw_center_rotmat=init_jaw_center_rotmat,
                                                      jawwidth=jaw_width_v, env_obs=obstacle_list,
                                                      rgba=[1, 0, 0, 1]))
            return False, None, None, -1
        # check jnt values
        jnt_values = robot.ik(hand_name,
                              init_jaw_center_pos,
                              init_jaw_center_rotmat,
                              local_minima="end")
        if jnt_values is None:
            self.ik_failed_init_num += 1
            if toggle_debug:
                save_error_info(IKError(grasp_pos=init_jaw_center_pos,
                                        grasp_rot=init_jaw_center_rotmat,
                                        grasp_jaw=jaw_width,
                                        hnd_instance=None))
            return False, None, None, -1
        approach_jnt_values = robot.ik(hand_name,
                                       init_jaw_center_approach_pos,
                                       init_jaw_center_rotmat,
                                       seed_jnt_values=jnt_values.copy(),
                                       local_minima="end")
        if approach_jnt_values is None:
            self.approach_ik_failed_init_num += 1
            if toggle_debug:
                save_error_info(IKError(grasp_pos=init_jaw_center_approach_pos,
                                        grasp_rot=init_jaw_center_rotmat,
                                        grasp_jaw=jaw_width,
                                        hnd_instance=None,
                                        seed_jnt=np.array(jnt_values)))
            return False, None, None, -1
        # check if the robot is collided at the grasp pos
        robot.fk(hand_name, jnt_values)
        # print(f"diff {(init_jaw_center_pos - self.robot_s.get_gl_tcp(hand_name)[0]) * 1000} mm")
        is_rbt_collided = robot.is_collided(obstacle_list)  # robot_s cd
        if is_rbt_collided:
            self.collision_init_num += 1
            if toggle_debug:
                save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                  hand_name=hand_name,
                                                  jawwidth=jaw_width,
                                                  env_obs=obstacle_list,
                                                  rbt_instance=self.robot_s))
            return False, None, None, -1
        # check if the robot is collided in the approach pos
        robot.fk(hand_name, approach_jnt_values)
        is_rbt_approach_collided = self.robot_s.is_collided(obstacle_list)  # robot_s cd
        if is_rbt_approach_collided:
            self.approach_ik_failed_init_num += 1
            if toggle_debug:
                save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                  hand_name=hand_name,
                                                  jawwidth=jaw_width,
                                                  env_obs=obstacle_list,
                                                  rbt_instance=self.robot_s))
            return False, None, None, -1
        ### DEBUG
        # hnd_instance = self.robot_s.hnd_dict[hand_name]
        # hnd_instance.copy()
        # hnd_instance.grip_at_with_jcpose(init_jaw_center_pos, init_jaw_center_rotmat, jaw_width)
        # hnd_instance.gen_meshmodel().attach_to(base)
        # robot.fk(hand_name,jnt_values)
        # print(jnt_values)
        # print(robot.get_gl_tcp(hand_name))
        # print("---"*20)
        # robot.gen_meshmodel().attach_to(base)
        # base.run()
        # ####
        return True, jnt_values, approach_jnt_values, jaw_width

    def _check_grasp_feasible_symmetric_tgt(self,
                                            grasp_id,
                                            hand_name,
                                            init_ik: np.array,
                                            sym_obj_tgt_homomats: np.array,
                                            approach_pos_offset: np.array,
                                            obstacle_list=[],
                                            toggle_debug=False) -> (bool, list, list, int):
        # a variable to store grasp info list
        grasp_info_list = self.grasp_info_list
        robot = self.robot_s
        hnd = self.hnd
        jaw_width_offset = .015
        # output information
        is_feasible = False
        jnt_values_list = []
        approach_jnt_values_list = []
        jawwidth_list = []
        # grasp info
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info_list[grasp_id]
        # grasp pose for the object
        grasp_homomat = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
        sym_tgt_jaw_center_homomats = np.dot(sym_obj_tgt_homomats.reshape((-1, 4)), grasp_homomat).reshape((-1, 4, 4))
        # robot ee poses
        sym_tgt_jaw_center_poses = sym_tgt_jaw_center_homomats[:, :3, 3]
        # robot ee rotmats
        sym_tgt_jaw_center_rotmats = sym_tgt_jaw_center_homomats[:, :3, :3]
        for sym_id, (tgt_jaw_center_pos, tgt_jaw_center_rotmat) in enumerate(zip(sym_tgt_jaw_center_poses,
                                                                                 sym_tgt_jaw_center_rotmats)):

            if self.feasible_ranges_rad is not None:
                if not is_grasp_feasible(self.rack_tf_inv[:3, :3].dot(tgt_jaw_center_rotmat), self.feasible_ranges_rad):
                    continue
            # check collision feasible
            hnd.grip_at_with_jcpose(gl_jaw_center_pos=tgt_jaw_center_pos,
                                    gl_jaw_center_rotmat=tgt_jaw_center_rotmat,
                                    jaw_width=jaw_width + jaw_width_offset)
            is_hnd_collision = hnd.is_collided(obstacle_list)
            if is_hnd_collision:
                if toggle_debug:
                    save_error_info(RbtHandCollisionError(hnd, gl_jaw_center_pos=tgt_jaw_center_pos,
                                                          gl_jaw_center_rotmat=tgt_jaw_center_rotmat,
                                                          jawwidth=jaw_width + jaw_width_offset,
                                                          env_obs=obstacle_list,
                                                          rgba=[0, 0, 1, 1]))
                continue
            # robot ee depart pose
            tgt_jaw_center_approach_pos = tgt_jaw_center_pos - approach_pos_offset
            robot.jaw_to(hand_name, jaw_width + jaw_width_offset)
            jnt_values = robot.ik(hand_name,
                                  tgt_jaw_center_pos,
                                  tgt_jaw_center_rotmat,
                                  seed_jnt_values=init_ik,
                                  local_minima="end")
            if jnt_values is None:
                self.ik_failed_tgt_num += 1
                if toggle_debug:
                    save_error_info(IKError(grasp_pos=tgt_jaw_center_pos,
                                            grasp_rot=tgt_jaw_center_rotmat,
                                            grasp_jaw=jaw_width,
                                            seed_jnt=init_ik,
                                            hnd_instance=None))
                continue
            approach_jnt_values = robot.ik(hand_name,
                                           tgt_jaw_center_approach_pos,
                                           tgt_jaw_center_rotmat,
                                           seed_jnt_values=jnt_values,
                                           local_minima="end")
            if approach_jnt_values is None:
                self.approach_ik_failed_tgt_num += 1
                if toggle_debug:
                    save_error_info(IKError(grasp_pos=tgt_jaw_center_approach_pos,
                                            grasp_rot=tgt_jaw_center_rotmat,
                                            grasp_jaw=jaw_width,
                                            hnd_instance=None,
                                            seed_jnt=np.array(jnt_values)))
                continue
            # check if the robot is collided at the grasp pos
            robot.fk(hand_name, jnt_values)
            is_rbt_collided = robot.is_collided(obstacle_list)  # robot_s cd
            if is_rbt_collided:
                self.collision_tgt_num += 1
                if toggle_debug:
                    save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                      hand_name=hand_name,
                                                      jawwidth=jaw_width,
                                                      env_obs=obstacle_list,
                                                      rbt_instance=self.robot_s))
                continue
            # check if the robot is collided in the approach pos
            robot.fk(hand_name, approach_jnt_values)
            is_rbt_approach_collided = self.robot_s.is_collided(obstacle_list)  # robot_s cd
            if is_rbt_approach_collided:
                self.apporach_collision_tgt_num += 1
                if toggle_debug:
                    save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                      hand_name=hand_name,
                                                      jawwidth=jaw_width,
                                                      env_obs=obstacle_list,
                                                      rbt_instance=self.robot_s))
                continue
            jnt_values_list.append(jnt_values)
            approach_jnt_values_list.append(jnt_values)
            jawwidth_list.append(jaw_width)

        if len(jnt_values_list) > 0:
            is_feasible = True
        return is_feasible, jnt_values_list, approach_jnt_values_list, jawwidth_list

    def print_error_info(self):
        print('Number of checked grasp poses: ', len(self.grasp_info_list))
        print('-----start-----')
        print('Number of failed IK at grasp: ', self.ik_failed_init_num)
        print('Number of failed IK at approach: ', self.approach_ik_failed_init_num)
        print('Number of collided robots at grasp: ', self.collision_init_num)
        print('Number of collided robots at approach: ', self.apporach_collision_init_num)
        # print('Number of feasible grasps: ', len(init_feasible_grasp_ids))
        print('------end------')
        print('-----target-----')
        print('Number of failed IK at grasp: ', self.ik_failed_tgt_num)
        print('Number of failed IK at approach: ', self.approach_ik_failed_tgt_num)
        print('Number of collided robots at grasp: ', self.collision_tgt_num)
        print('Number of collided robots at approach: ', self.apporach_collision_tgt_num)
        # print('Number of feasible grasps: ', len(init_feasible_grasp_ids))
        print('------end------')


class CommonGraspIteratorGenerator:
    def __init__(self, robot_s):
        self.robot_s = robot_s
        self.hnd = YumiGripper()

    def gen_iterator(self,
                     grasp_info_list,
                     hand_name,
                     obj_init_homomat,
                     sym_obj_tgt_homomats,
                     pick_approach_pos_offset,
                     place_approach_pos_offset,
                     obstacle_list,
                     feasible_ranges_rad,
                     rack_tf,
                     toggle_debug=False):
        return iter(CommonGraspIterator(
            robot_s=self.robot_s,
            hnd=self.hnd,
            grasp_info_list=grasp_info_list,
            hand_name=hand_name,
            obj_init_homomat=obj_init_homomat,
            sym_obj_tgt_homomats=sym_obj_tgt_homomats,
            pick_approach_pos_offset=pick_approach_pos_offset,
            place_approach_pos_offset=place_approach_pos_offset,
            obstacle_list=obstacle_list,
            feasible_ranges_rad=feasible_ranges_rad,
            rack_tf=rack_tf,
            toggle_debug=toggle_debug))


class PickPlacePlanner(PickPlacePlanner_CR):
    NO_PLANNING_RESULT = (None, None, None, None, None, None, None, None)

    def __init__(self, robot_s, common_grasp_solver: 'CommonGraspSolver' = None):
        """
        :param object:
        :param robot_helper:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        super().__init__(robot_s, )
        self._common_grasp_solver = common_grasp_solver
        self._num_workers = self._common_grasp_solver._num_workers

    def visualize_generated_results(self, hand_name, common_grasp_iter, obstacle_list):
        plot_node = [None]
        counter = [0]
        from modeling.model_collection import ModelCollection
        def update(hand_name, task):
            if base.inputmgr.keymap["space"]:
                base.inputmgr.keymap["space"] = False
                if counter[0] > len(common_grasp_iter) - 1:
                    counter[0] = 0
                    return task.again
                else:
                    if plot_node[0] is not None:
                        plot_node[0].detach()
                    plot_node[0] = ModelCollection()
                    (grasp_id, init_jnt_val, init_jaw_width, tgt_jnt_val_list, tgt_jawwidth_list) \
                        = common_grasp_iter[counter[0]]
                    print(f" init jnt val is: {init_jnt_val}")
                    print(f" tgt jnt val is: {tgt_jnt_val_list}")
                    print("****" * 10)
                    self.robot_s.fk(hand_name, np.array(init_jnt_val))
                    self.robot_s.jaw_to(hand_name, init_jaw_width)
                    self.robot_s.gen_meshmodel(rgba=[0, 1, 0, 1]).attach_to(plot_node[0])
                    for tgt_val in tgt_jnt_val_list:
                        self.robot_s.fk(hand_name, tgt_val)
                    self.robot_s.gen_meshmodel(rgba=[1, 0, 0, 1]).attach_to(plot_node[0])
                    for _ in obstacle_list:
                        _.attach_to(plot_node[0])
                        _.show_cdprimit()
                    plot_node[0].attach_to(base)
                    counter[0] += 1
            return task.again

        taskMgr.doMethodLater(0.02, update, "update", extraArgs=[hand_name], appendTask=True)
        base.run()

    def gen_pick_and_place_motion(self,
                                  hnd_name,
                                  tube,
                                  tube_cm,
                                  start_conf,
                                  end_conf,
                                  grasp_info_list,
                                  goal_homomat_list,
                                  approach_direction_list,
                                  approach_distance_list,
                                  depart_direction_list,
                                  depart_distance_list,
                                  common_grasp_iter=None,
                                  approach_jawwidth=None,
                                  depart_jawwidth=None,
                                  ad_granularity=.01,
                                  use_rrt=True,
                                  obstacle_list=[],
                                  grasp_obstacle_list=[],
                                  rrt_obstacle_list=[],  # obstacles for rrt
                                  use_incremental=False,
                                  feasible_ranges_rad=None,
                                  rack_tf=None,
                                  logger=logging.getLogger(__name__),
                                  toggle_debug=False):
        # TODO backup the jawwidth and joint value and set the jaw with and joint value back to initial value when finished
        """

        :param hnd_name:
        :param tube_cm:
        :param grasp_info_list:
        :param goal_homomat_list:
        :param start_conf: RRT motion between start_state and pre_approach; No RRT motion if None
        :param end_conf: RRT motion between post_depart and end_conf; Noe RRT motion if None
        :param approach_direction_list: the first element will be the pick approach direction
        :param approach_distance_list: the first element will be the pick approach direction
        :param depart_direction_list: the last element will be the release depart direction
        :param depart_distance_list: the last element will be the release depart direction
        :param approach_jawwidth:
        :param depart_jawwidth:
        :param ad_granularity:
        :param use_rrt:
        :param obstacle_list:
        :param use_incremental:
        :return:
        author: weiwei
        date: 20191122, 20200105
        """
        logger_name = "gen_pick_and_place"
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]
        if depart_jawwidth is None:
            depart_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]

        obj_init_homomat = goal_homomat_list[0]
        obj_tgt_homomat = goal_homomat_list[1]

        first_goal_pos = goal_homomat_list[0][:3, 3]
        first_goal_rotmat = goal_homomat_list[0][:3, :3]

        # find common grasps planning
        if common_grasp_iter is None:
            chunk_len = max(int(np.ceil(len(grasp_info_list) / self._num_workers)), 1)
            datalist = [_REQ_PACKET(chunk_init_id=_c * chunk_len,
                                    grasp_info_list=grasp_info_list_chunk,
                                    hnd_name=hnd_name,
                                    obj_init_homomat=obj_init_homomat.copy(),
                                    sym_obj_tgt_homomats=tube.get_symmetry_homomat(15, obj_tgt_homomat),
                                    pick_approach_pos_offset=approach_direction_list[0] * approach_distance_list[0] if
                                    approach_direction_list[0] is not None else approach_distance_list[0],
                                    place_approach_pos_offset=approach_direction_list[1] * approach_distance_list[1] if
                                    approach_direction_list[1] is not None else approach_distance_list[1],
                                    obstacle_list=obstacle_list,
                                    feasible_ranges_rad=feasible_ranges_rad,
                                    rack_tf=rack_tf)
                        for _c, grasp_info_list_chunk in
                        enumerate(
                            [grasp_info_list[x:x + chunk_len] for x in range(0, len(grasp_info_list), chunk_len)])]
            common_grasp_iter = self._common_grasp_solver.send_recv_data(datalist)
        # logging
        if toggle_debug:
            self.visualize_generated_results(hnd_name, common_grasp_iter, obstacle_list)
        logger.info(
            f"[{logger_name}]: Number of feasible common grasps {len(common_grasp_iter)}: {[_[0] for _ in common_grasp_iter]}.")
        if len(common_grasp_iter) < 1:
            return [*self.NO_PLANNING_RESULT, False, True]
        # motion planning
        if toggle_debug:
            debug_data_list = []
        is_pick_feasible = None
        is_place_feasible = None
        for iter_id, (grasp_id, init_jnt_val, init_jaw_width, tgt_jnt_val_list, tgt_jawwidth_list) in enumerate(
                common_grasp_iter):
            is_pick_feasible = True
            if toggle_debug:
                debug_data_list.append([init_jnt_val, init_jaw_width, tgt_jnt_val_list, tgt_jawwidth_list])
            # jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            # approach
            self.robot_s.fk(hnd_name, init_jnt_val)
            first_jaw_center_pos, first_jaw_center_rotmat = self.robot_s.get_gl_tcp(hnd_name)
            # print("???", grasp_id)
            # print((obj_init_homomat[:3,3] - first_jaw_center_pos + np.array(grasp_info_list[grasp_id][1]))*1000)
            # exit(0)
            # objcm as an obstacle
            objcm_copy = tube_cm.copy()
            objcm_copy.set_pos(first_goal_pos)
            objcm_copy.set_rotmat(first_goal_rotmat)
            start2pickapproach_motion_seg, pickapproach_motion_seg = \
                self.gen_approach_motion(component_name=hnd_name,
                                         goal_tcp_pos=first_jaw_center_pos,
                                         goal_tcp_rotmat=first_jaw_center_rotmat,
                                         start_conf=start_conf,
                                         approach_direction=approach_direction_list[0],
                                         approach_distance=approach_distance_list[0],
                                         approach_jawwidth=approach_jawwidth,
                                         granularity=ad_granularity,
                                         obstacle_list=obstacle_list + grasp_obstacle_list + rrt_obstacle_list,
                                         object_list=[objcm_copy],
                                         seed_jnt_values=init_jnt_val)
            if start2pickapproach_motion_seg is None:
                print("Cannot generate the pick motion!")
                is_pick_feasible = False
                continue
                # middle
            # middle motion
            for tgt_id, tgt_jnt_val in enumerate(tgt_jnt_val_list):
                is_place_feasible = True
                last_goal_pos = obj_tgt_homomat[:3, 3]
                last_goal_rotmat = obj_tgt_homomat[:3, :3]
                pickdepart_motion_seg, pickdepart2placeapproach_motion_seg, placeapproach_motion_seg = \
                    self.gen_holding_moveto(hand_name=hnd_name,
                                            objcm=tube_cm,
                                            jaw_width=tgt_jawwidth_list[tgt_id],
                                            obj_pose_list=[goal_homomat_list[0], goal_homomat_list[1]],
                                            depart_direction_list=depart_direction_list,
                                            approach_direction_list=approach_direction_list,
                                            depart_distance_list=depart_distance_list,
                                            approach_distance_list=approach_distance_list,
                                            first_conf=pickapproach_motion_seg.conf_ls[-1],
                                            goal_conf=tgt_jnt_val,
                                            ad_granularity=ad_granularity,
                                            use_rrt=use_rrt,
                                            obstacle_list=obstacle_list + grasp_obstacle_list, )
                if pickdepart_motion_seg is None:
                    print("Cannot generate place depart motion")
                    is_pick_feasible = False
                    break
                if pickdepart2placeapproach_motion_seg is None:
                    print("Cannot generate the hold moving motion")
                    is_place_feasible = False
                    continue
                if placeapproach_motion_seg is None:
                    print("Cannot generate the place approach motion")
                    is_place_feasible = False
                    continue
                # departure
                self.robot_s.fk(hnd_name, tgt_jnt_val)
                last_jaw_center_pos, last_jaw_center_rotmat = self.robot_s.get_gl_tcp(hnd_name)
                # objcm as an obstacle
                objcm_copy.set_pos(last_goal_pos)
                objcm_copy.set_rotmat(last_goal_rotmat)
                placedepart_motion_seg, placedepart2goal_motion_seg = \
                    self.gen_depart_motion(component_name=hnd_name,
                                           start_tcp_pos=last_jaw_center_pos,
                                           start_tcp_rotmat=last_jaw_center_rotmat,
                                           end_conf=end_conf,
                                           depart_direction=depart_direction_list[0],
                                           depart_distance=depart_distance_list[0],
                                           depart_jawwidth=depart_jawwidth,
                                           granularity=ad_granularity,
                                           obstacle_list=obstacle_list + grasp_obstacle_list,
                                           object_list=[objcm_copy],
                                           seed_jnt_values=placeapproach_motion_seg.conf_ls[-1])
                if placedepart_motion_seg is None:
                    print("Cannot generate the release motion!")
                    is_place_feasible = False
                    continue

                if motion_seg_avaliable(start2pickapproach_motion_seg):
                    start2pickapproach_motion_seg = _MOTION_SEG(
                        name=start2pickapproach_motion_seg.name,
                        conf_ls=start2pickapproach_motion_seg.conf_ls,
                        jaww_ls=start2pickapproach_motion_seg.jaww_ls,
                        objpose_ls=self.gen_object_motion(component_name=hnd_name,
                                                          conf_list=start2pickapproach_motion_seg.conf_ls,
                                                          obj_pos=first_goal_pos,
                                                          obj_rotmat=first_goal_rotmat,
                                                          type='absolute'))

                pickapproach_motion_seg = _MOTION_SEG(
                    name=pickapproach_motion_seg.name,
                    conf_ls=pickapproach_motion_seg.conf_ls,
                    jaww_ls=pickapproach_motion_seg.jaww_ls,
                    objpose_ls=self.gen_object_motion(component_name=hnd_name,
                                                      conf_list=pickapproach_motion_seg.conf_ls,
                                                      obj_pos=first_goal_pos,
                                                      obj_rotmat=first_goal_rotmat,
                                                      type='absolute'))
                placedepart_motion_seg = _MOTION_SEG(
                    name=placedepart_motion_seg.name,
                    conf_ls=placedepart_motion_seg.conf_ls,
                    jaww_ls=placedepart_motion_seg.jaww_ls,
                    objpose_ls=self.gen_object_motion(component_name=hnd_name,
                                                      conf_list=placedepart_motion_seg.conf_ls,
                                                      obj_pos=last_goal_pos,
                                                      obj_rotmat=last_goal_rotmat,
                                                      type='absolute'))

                if motion_seg_avaliable(placedepart2goal_motion_seg):
                    placedepart2goal_motion_seg = _MOTION_SEG(
                        name=placedepart2goal_motion_seg.name,
                        conf_ls=placedepart2goal_motion_seg.conf_ls,
                        jaww_ls=placedepart2goal_motion_seg.jaww_ls,
                        objpose_ls=self.gen_object_motion(component_name=hnd_name,
                                                          conf_list=placedepart2goal_motion_seg.conf_ls,
                                                          obj_pos=last_goal_pos,
                                                          obj_rotmat=last_goal_rotmat,
                                                          type='absolute'))
                logger.info(
                    f"[{logger_name}]: SELECTED grasp: {grasp_id}.")
                if toggle_debug:
                    fs.dump_pickle(debug_data_list, fs.workdir.joinpath("components", "exe",
                                                                        "debug_data", "generated_jnts",
                                                                        f"{time.strftime('%Y%m%d-%H%M%S')}_jnts.pkl"))
                print(f"SELECTED grasp: {grasp_id}.")
                return start2pickapproach_motion_seg, pickapproach_motion_seg, pickdepart_motion_seg, \
                    pickdepart2placeapproach_motion_seg, placeapproach_motion_seg, placedepart_motion_seg, \
                    placedepart2goal_motion_seg, \
                    common_grasp_iter[iter_id:], \
                    is_pick_feasible, is_place_feasible

        return [*self.NO_PLANNING_RESULT, is_pick_feasible, is_place_feasible]


if __name__ == '__main__':
    from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
    from huri.core.common_import import *
    from huri.core.print_tool import text_pd

    # create the virtual environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    # generate the yumi robot and its mesh model
    ym_rbt = ym.Yumi(enable_cc=True)
    ym_rbt.gen_meshmodel().attach_to(base)
    hnd_name = "rgt_hnd"

    # generate a rack from a prototype
    rack_1 = Rack_Hard_Proto.copy()
    rack_1.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, -.2, 0])))
    # insert test tube to the rack
    rack_1.insert_tube(slot_id=np.array([0, 3]), tube=TubeType.TUBE_TYPE_3)
    rack_1.insert_tube(slot_id=np.array([4, 1]), tube=TubeType.TUBE_TYPE_3)
    rack_1.insert_tube(slot_id=np.array([3, 2]), tube=TubeType.TUBE_TYPE_3)
    rack_1.insert_tube(slot_id=np.array([2, 3]), tube=TubeType.TUBE_TYPE_3)

    # move test tube
    remove_ind = np.array([0, 3])
    tube_type, tube_homomat_gl = rack_1.remove_slot(slot_id=remove_ind)
    tube_move = TubeType.gen_tube_by_tubetype(tube_type)
    tube_move.set_homomat(tube_homomat_gl)
    tube_cm_mdl = tube_move.gen_collision_model()
    tube_cm_mdl.set_rgba([1, 0, 0, .7])
    tube_cm_mdl.attach_to(base)

    # generate the collision model in the environment
    obs_rack_1, obs_tube_collection = rack_1.gen_collision_model(gen_tube=True)
    obs_rack_1.attach_to(base)
    obs_tube_collection.attach_to(base)
    # [_.attach_to(base) for _ in obs_tube_collection]

    print("The state after inserting the test tube rack: ")
    print(text_pd(rack_1.rack_status))
    print("-" * 30)

    obj_init_homomat = tube_homomat_gl
    obj_tgt_homomat = rack_1.get_slot_homomat(np.array([4, 9]))
    goal_place_offset_dis = .04
    obj_tgt_homomat[:3, 3] = obj_tgt_homomat[:3, 3] + rack_1.get_homomat()[:3, 2] * goal_place_offset_dis

    # planner = PickPlacePlanner(ym_rbt)

    common_grasp_iter = CommonGraspIteratorGenerator(robot_s=ym_rbt)

    grasp_iter = common_grasp_iter.gen_iterator(
        grasp_info_list=fs.load_json(TubeType.WHITE_CAP_TUBE.grasps_path),
        hand_name=hnd_name,
        obj_init_homomat=obj_init_homomat,
        sym_obj_tgt_homomats=TubeType.WHITE_CAP_TUBE.get_symmetry_homomat(3, obj_tgt_homomat),
        pick_approach_pos_offset=-rack_1.get_homomat()[:3, 2],
        place_approach_pos_offset=-rack_1.get_homomat()[:3, 2],
        obstacle_list=obs_tube_collection.cm_list, )
    feasible_num = 0
    for _ in grasp_iter:
        print(grasp_iter._counter)
        if _[0]:
            feasible_num += 1

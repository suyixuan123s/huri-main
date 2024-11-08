import math
import numpy as np
import basis.robot_math as rm
import basis.data_adapter as da
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc
import manipulation.approach_depart_planner as adp
import modeling.model_collection as mc
from huri.components.multiprocess.pick_and_place import get_available_graspids_ik
import modeling.geometric_model as gm
import time
from huri.components.debug.inspect import save_rbt_info, show_animation, save_error_info, IKError, RbtCollisionError


class ReplanningFlag(Exception):
    def __init__(self, msg):
        super(ReplanningFlag, self).__init__(msg)


class PickPlacePlanner(adp.ADPlanner):

    def gen_object_motion(self, component_name, conf_list, obj_pos, obj_rotmat, type='absolute'):
        """
        :param conf_list:
        :param obj_pos:
        :param obj_rotmat:
        :param type: 'absolute' or 'relative'
        :return:
        author: weiwei
        date: 20210125
        """
        objpose_list = []
        if type == 'absolute':
            for _ in conf_list:
                objpose_list.append(rm.homomat_from_posrot(obj_pos, obj_rotmat))
        elif type == 'relative':
            jnt_values_bk = self.robot_s.get_jnt_values(component_name)
            for conf in conf_list:
                self.robot_s.fk(component_name, conf)
                gl_obj_pos, gl_obj_rotmat = self.robot_s.cvt_loc_tcp_to_gl(component_name, obj_pos, obj_rotmat)
                objpose_list.append(rm.homomat_from_posrot(gl_obj_pos, gl_obj_rotmat))
            self.robot_s.fk(component_name, jnt_values_bk)
        else:
            raise ValueError('Type must be absolute or relative!')
        return objpose_list

    def find_common_graspids_symmetric(self,
                                       hand_name,  # TODO hnd is on  a manipulator
                                       tube,
                                       grasp_info_list,
                                       obj_init_homomat,
                                       obj_goal_homomat,
                                       approach_direction_list,
                                       approach_distance_list,
                                       obstacle_list=[],
                                       symmetry_discrete_factor=10,
                                       toggle_debug=True):
        """
        find the common collision free and IK feasible graspids
        :param hand_name: a component may have multiple hands
        :param grasp_info_list: a list like [[jawwidth, gl_jaw_center_pos, pos, rotmat], ...]
        :param goal_homomat_list: [homomat, ...]
        :param obstacle_list
        :return: [final_available_graspids, intermediate_available_graspids]
        author: weiwei
        date: 20210113, 20210125
        """
        # start reasoning
        previously_available_graspids = range(len(grasp_info_list))

        # counter to record the failed num at start
        ikfailed_grasps_num = 0
        ikfailed_grasps_approach_num = 0
        rbtcollided_grasps_num = 0
        rbtcollided_grasps_approach_num = 0

        # hnd instance for error
        hnd_instance = self.robot_s.hnd_dict[hand_name]
        grasp_offset = .015

        # bacakup init jnt angle and jawwidth
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)
        jawwidth_bk = self.robot_s.get_jawwidth(hand_name)

        # lists to store the feasible grasp id
        init_feasible_grasp_ids = []
        init_feasible_iks = []
        init_feasible_approach_iks = []
        goal_feasible_grasp_ids = []
        feasible_symmetry_obj_goal_hommat_dict = {}

        # init
        init_pos = obj_init_homomat[:3, 3]
        init_rotmat = obj_init_homomat[:3, :3]
        graspid_and_graspinfo_list = list(zip(previously_available_graspids,  # need .copy()?
                                              [grasp_info_list[i] for i in previously_available_graspids]))
        # TODO check the kinematically feasible and collision free at pick and place pose
        for idx, (graspid, grasp_info) in enumerate(graspid_and_graspinfo_list):
            # grasp info
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            # robot ee pose
            init_jaw_center_pos = init_pos + init_rotmat.dot(jaw_center_pos)
            # robot ee rot
            init_jaw_center_rotmat = init_rotmat.dot(jaw_center_rotmat)
            # robot ee approach pose
            init_jaw_center_approach_pos = init_jaw_center_pos - approach_direction_list[0] * approach_distance_list[0]

            # grasp at some pose
            self.robot_s.jaw_to(hand_name, jaw_width + grasp_offset)

            ##############
            # check 1 jnt value, 2 jnt value approach 3 rbt collision
            #
            ##############
            # initialization
            jnt_values = None
            approach_jnt_values = None
            is_rbt_collided = False
            ####
            jnt_values = self.robot_s.ik(hand_name,
                                         init_jaw_center_pos,
                                         init_jaw_center_rotmat,
                                         local_minima="end")
            if jnt_values is None:
                ikfailed_grasps_num += 1
                if toggle_debug:
                    save_error_info(IKError(grasp_pos=init_jaw_center_pos,
                                            grasp_rot=init_jaw_center_rotmat,
                                            grasp_jaw=jaw_width,
                                            hnd_instance=hnd_instance.copy()))
                continue

            approach_jnt_values = self.robot_s.ik(hand_name,
                                                  init_jaw_center_approach_pos,
                                                  init_jaw_center_rotmat,
                                                  seed_jnt_values=jnt_values.copy(),
                                                  local_minima="end")
            if approach_jnt_values is None:
                ikfailed_grasps_approach_num += 1
                if toggle_debug:
                    save_error_info(IKError(grasp_pos=init_jaw_center_approach_pos,
                                            grasp_rot=init_jaw_center_rotmat,
                                            grasp_jaw=jaw_width,
                                            hnd_instance=hnd_instance.copy(),
                                            seed_jnt=np.array(jnt_values)))
                continue

            # check if the robot is collided at the grasp pos
            self.robot_s.fk(hand_name, jnt_values)
            #print(f"diff {(init_jaw_center_pos - self.robot_s.get_gl_tcp(hand_name)[0]) * 1000} mm")
            is_rbt_collided = self.robot_s.is_collided(obstacle_list)  # robot_s cd
            if is_rbt_collided:
                rbtcollided_grasps_num += 1
                if toggle_debug:
                    save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                      hand_name=hand_name,
                                                      jawwidth=jaw_width,
                                                      env_obs=obstacle_list,
                                                      rbt_instance=self.robot_s))
                continue
            # check if the robot is collided in the approach pos
            self.robot_s.fk(hand_name, approach_jnt_values)
            is_rbt_collided = is_rbt_collided or self.robot_s.is_collided(obstacle_list)  # robot_s cd
            if is_rbt_collided:
                rbtcollided_grasps_approach_num += 1
                if toggle_debug:
                    save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                      hand_name=hand_name,
                                                      jawwidth=jaw_width,
                                                      env_obs=obstacle_list,
                                                      rbt_instance=self.robot_s))
                continue

            # hnd cdfree, rbs ikf/cdfree, obj cdfree
            init_feasible_grasp_ids.append(graspid)
            init_feasible_iks.append(jnt_values)
            init_feasible_approach_iks.append(approach_jnt_values)

        print('-----start-----')
        print('Number of failed IK at grasp: ', ikfailed_grasps_num)
        print('Number of failed IK at approach: ', ikfailed_grasps_approach_num)
        print('Number of collided robots at grasp: ', rbtcollided_grasps_num)
        print('Number of collided robots at approach: ', rbtcollided_grasps_approach_num)
        print('Number of feasible grasps: ', len(init_feasible_grasp_ids))
        print('------end------')

        if toggle_debug:
            show_animation(info_type="cd_error")
            base.run()

        # counter to record the failed num at goal
        ikfailed_grasps_num = 0
        ikfailed_grasps_approach_num = 0
        rbtcollided_grasps_num = 0
        rbtcollided_grasps_approach_num = 0

        if len(init_feasible_grasp_ids) > 10:
            symmetry_discrete_factor = min(3, symmetry_discrete_factor)
        if len(init_feasible_grasp_ids) < 5:
            symmetry_discrete_factor = max(10, symmetry_discrete_factor)

        # goal considers symmetric
        # TODO check this function
        symmetry_obj_goal_homomat = tube.get_symmetry_homomat(discrete_factor=symmetry_discrete_factor,
                                                              rot_mat=obj_goal_homomat)
        graspid_and_graspinfo_list = list(zip(init_feasible_grasp_ids,  # need .copy()?
                                              [grasp_info_list[i] for i in init_feasible_grasp_ids],
                                              init_feasible_approach_iks))

        for idx, (graspid, grasp_info, init_feasible_ik) in enumerate(graspid_and_graspinfo_list):
            # grasp info
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            # grasp pose for the object
            grasp_homomat = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
            symmetry_goal_jaw_homomats = np.dot(symmetry_obj_goal_homomat.reshape((-1, 4)), grasp_homomat).reshape(
                (-1, 4, 4))
            # robot ee pose
            symmetry_goal_jaw_center_pos = symmetry_goal_jaw_homomats[:, :3, 3]
            # robot ee rot
            symmetry_goal_jaw_center_rotmat = symmetry_goal_jaw_homomats[:, :3, :3]
            is_add_goal_feasible_grasp = False
            for _symmetry_ind, (_goal_jaw_center_pos, _goal_jaw_center_rotmat) in enumerate(
                    zip(symmetry_goal_jaw_center_pos,
                        symmetry_goal_jaw_center_rotmat)):
                # robot ee approach pose
                goal_jaw_center_approach_pos = _goal_jaw_center_pos - approach_direction_list[1] * \
                                               approach_distance_list[1]
                # grasp at some pose
                self.robot_s.jaw_to(hand_name, jaw_width + grasp_offset)

                ##############
                # check 1 jnt value, 2 jnt value approach 3 rbt collision
                #
                ##############
                # initialization
                jnt_values = None
                approach_jnt_values = None
                is_rbt_collided = False
                ####
                jnt_values = self.robot_s.ik(hand_name,
                                             _goal_jaw_center_pos,
                                             _goal_jaw_center_rotmat,
                                             seed_jnt_values=np.array(init_feasible_ik),
                                             local_minima="end")
                if jnt_values is None:
                    ikfailed_grasps_num += 1
                    if toggle_debug:
                        save_error_info(IKError(grasp_pos=_goal_jaw_center_pos,
                                                grasp_rot=_goal_jaw_center_rotmat,
                                                grasp_jaw=jaw_width,
                                                hnd_instance=hnd_instance.copy(),
                                                seed_jnt=np.array(init_feasible_ik)))
                    continue

                # TODO: only do the ik-feasible-checking for the approach pose to save time of planning
                approach_jnt_values = self.robot_s.ik(hand_name,
                                                      goal_jaw_center_approach_pos,
                                                      _goal_jaw_center_rotmat,
                                                      seed_jnt_values=jnt_values.copy(),
                                                      local_minima="end")
                if approach_jnt_values is None:
                    ikfailed_grasps_approach_num += 1
                    if toggle_debug:
                        save_error_info(IKError(grasp_pos=goal_jaw_center_approach_pos,
                                                grasp_rot=_goal_jaw_center_rotmat,
                                                grasp_jaw=jaw_width,
                                                hnd_instance=hnd_instance.copy(),
                                                seed_jnt=np.array(jnt_values)))
                    continue

                # check if the robot is collided at the grasp pos
                self.robot_s.fk(hand_name, jnt_values)
                is_rbt_collided = self.robot_s.is_collided(obstacle_list)  # robot_s cd
                if is_rbt_collided:
                    rbtcollided_grasps_num += 1
                    if toggle_debug:
                        save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                          hand_name=hand_name,
                                                          jawwidth=jaw_width,
                                                          env_obs=obstacle_list,
                                                          rbt_instance=self.robot_s))
                    continue
                # check if the robot is collided in the approach pos
                self.robot_s.fk(hand_name, approach_jnt_values)
                is_rbt_collided = is_rbt_collided or self.robot_s.is_collided(obstacle_list)  # robot_s cd
                if is_rbt_collided:
                    rbtcollided_grasps_approach_num += 1
                    if toggle_debug:
                        save_error_info(RbtCollisionError(jnts=self.robot_s.get_jnt_values(component_name=hand_name),
                                                          hand_name=hand_name,
                                                          jawwidth=jaw_width,
                                                          env_obs=obstacle_list,
                                                          rbt_instance=self.robot_s))
                    continue

                # hnd cdfree, rbs ikf/cdfree, obj cdfree
                if not is_add_goal_feasible_grasp:
                    feasible_symmetry_obj_goal_hommat_dict[graspid] = []
                    goal_feasible_grasp_ids.append(graspid)
                    is_add_goal_feasible_grasp = True
                feasible_symmetry_obj_goal_hommat_dict[graspid].append(
                    symmetry_obj_goal_homomat[_symmetry_ind])

        print('-----goal-----')
        print('Number of failed IK at grasp: ', ikfailed_grasps_num)
        print('Number of failed IK at approach: ', ikfailed_grasps_approach_num)
        print('Number of collided robots at grasp: ', rbtcollided_grasps_num)
        print('Number of collided robots at approach: ', rbtcollided_grasps_approach_num)
        print('Number of feasible grasps: ', len(goal_feasible_grasp_ids))
        print('------end------')

        final_available_graspids = goal_feasible_grasp_ids
        self.robot_s.fk(hand_name, jnt_values_bk)
        self.robot_s.jaw_to(hand_name, jawwidth_bk)

        return final_available_graspids, \
               feasible_symmetry_obj_goal_hommat_dict, \
               init_feasible_iks, \
               init_feasible_approach_iks

    def gen_approach_motion(self,
                            component_name,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            start_conf=None,
                            approach_direction=None,  # np.array([0, 0, -1])
                            approach_distance=.1,
                            approach_jawwidth=.05,
                            granularity=.03,
                            obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                            object_list=[],  # target objects, will be checked by rrt, but not by linear
                            seed_jnt_values=None,
                            toggle_end_grasp=False,
                            end_jawwidth=.0):
        if seed_jnt_values is None:
            seed_jnt_values = start_conf
        if approach_direction is None:
            approach_direction = goal_tcp_rotmat[:, 2]
        conf_list, jawwidth_list = self.gen_approach_linear(component_name,
                                                            goal_tcp_pos,
                                                            goal_tcp_rotmat,
                                                            approach_direction,
                                                            approach_distance,
                                                            approach_jawwidth,
                                                            granularity,
                                                            [],
                                                            seed_jnt_values,
                                                            toggle_end_grasp,
                                                            end_jawwidth)
        if conf_list is None:
            print("ADPlanner: Cannot gen approach linear!")
            return None, None
        if start_conf is not None:
            start2approach_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                              start_conf=start_conf,
                                                              goal_conf=conf_list[0],
                                                              obstacle_list=obstacle_list + object_list,
                                                              ext_dist=.05,
                                                              max_time=300)
            if start2approach_conf_list is None:
                print("ADPlanner: Cannot plan approach motion!")
                return None, None
            start2approach_jawwidth_list = self.gen_jawwidth_motion(start2approach_conf_list, approach_jawwidth)
        return start2approach_conf_list + conf_list, start2approach_jawwidth_list + jawwidth_list

    def gen_depart_motion(self,
                          component_name,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          end_conf=None,
                          depart_direction=None,  # np.array([0, 0, 1])
                          depart_distance=.1,
                          depart_jawwidth=.05,
                          granularity=.03,
                          obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                          object_list=[],  # target objects, will be checked by rrt, but not by linear
                          seed_jnt_values=None,
                          toggle_begin_grasp=False,
                          begin_jawwidth=.0):
        if seed_jnt_values is None:
            seed_jnt_values = end_conf
        if depart_direction is None:
            depart_direction = start_tcp_rotmat[:, 2]
        conf_list, jawwidth_list = self.gen_depart_linear(component_name,
                                                          start_tcp_pos,
                                                          start_tcp_rotmat,
                                                          depart_direction,
                                                          depart_distance,
                                                          depart_jawwidth,
                                                          granularity,
                                                          [],
                                                          seed_jnt_values,
                                                          toggle_begin_grasp,
                                                          begin_jawwidth)
        if conf_list is None:
            print("ADPlanner: Cannot gen depart linear!")
            return None, None
        if end_conf is not None:
            depart2goal_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                           start_conf=conf_list[-1],
                                                           goal_conf=end_conf,
                                                           obstacle_list=obstacle_list + object_list,
                                                           ext_dist=.05,
                                                           max_time=300)
            if depart2goal_conf_list is None:
                print("ADPlanner: Cannot plan depart motion!")
                return None, None
            depart2goal_jawwidth_list = self.gen_jawwidth_motion(depart2goal_conf_list, depart_jawwidth)
        else:
            depart2goal_conf_list = []
            depart2goal_jawwidth_list = []
        return conf_list + depart2goal_conf_list, jawwidth_list + depart2goal_jawwidth_list

    def gen_holding_moveto(self,
                           hand_name,
                           objcm,
                           grasp_info,
                           obj_pose_list,
                           depart_direction_list,
                           depart_distance_list,
                           approach_direction_list,
                           approach_distance_list,
                           ad_granularity=.007,
                           use_rrt=True,
                           obstacle_list=[],
                           seed_jnt_values=None):
        """
        hold and move an object to multiple poses
        :param hand_name:
        :param grasp_info:
        :param obj_pose_list:
        :param depart_direction_list: the last element will be ignored
        :param depart_distance_list: the last element will be ignored
        :param approach_direction_list: the first element will be ignored
        :param approach_distance_list: the first element will be ignored
        :param ad_granularity:
        :param obstacle_list:
        :param seed_jnt_values:
        :return:
        """
        jnt_values_bk = self.robot_s.get_jnt_values(hand_name)
        jawwidth_bk = self.robot_s.get_jawwidth(hand_name)
        # final
        conf_list = []
        jawwidthlist = []
        objpose_list = []
        # hold object
        jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
        first_obj_pos = obj_pose_list[0][:3, 3]
        first_obj_rotmat = obj_pose_list[0][:3, :3]
        first_jaw_center_pos = first_obj_rotmat.dot(jaw_center_pos) + first_obj_pos
        first_jaw_center_rotmat = first_obj_rotmat.dot(jaw_center_rotmat)
        first_conf = self.robot_s.ik(hand_name,
                                     first_jaw_center_pos,
                                     first_jaw_center_rotmat,
                                     seed_jnt_values=seed_jnt_values)
        if first_conf is None:
            print("Cannot solve the ik at the first grasping pose!")
            return None, None, None
        self.robot_s.fk(component_name=hand_name, jnt_values=first_conf)
        # set a copy of the object to the start pose, hold the object, and move it to goal object pose
        objcm_copy = objcm.copy()
        objcm_copy.set_pos(first_obj_pos)
        objcm_copy.set_rotmat(first_obj_rotmat)
        rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hand_name, objcm_copy, jaw_width)
        seed_conf = first_conf
        for i in range(len(obj_pose_list) - 1):
            # get start and goal object poses
            start_obj_pos = obj_pose_list[i][:3, 3]
            start_obj_rotmat = obj_pose_list[i][:3, :3]
            goal_obj_pos = obj_pose_list[i + 1][:3, 3]
            goal_obj_rotmat = obj_pose_list[i + 1][:3, :3]
            # transform grasps
            start_jaw_center_pos = start_obj_rotmat.dot(jaw_center_pos) + start_obj_pos
            start_jaw_center_rotmat = start_obj_rotmat.dot(jaw_center_rotmat)
            goal_jaw_center_pos = goal_obj_rotmat.dot(jaw_center_pos) + goal_obj_pos
            goal_jaw_center_rotmat = goal_obj_rotmat.dot(jaw_center_rotmat)
            depart_direction = depart_direction_list[i]
            if depart_direction is None:
                depart_direction = -start_jaw_center_rotmat[:, 2]
            depart_distance = depart_distance_list[i]
            if depart_distance is None:
                depart_distance = 0
            approach_direction = approach_direction_list[i + 1]
            if approach_direction is None:
                approach_direction = goal_jaw_center_rotmat[:, 2]
            approach_distance = approach_distance_list[i + 1]
            if approach_distance is None:
                approach_distance = 0
            # depart linear
            conf_list_depart = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                    goal_tcp_pos=start_jaw_center_pos,
                                                                    goal_tcp_rotmat=start_jaw_center_rotmat,
                                                                    direction=depart_direction,
                                                                    distance=depart_distance,
                                                                    obstacle_list=[],
                                                                    # The test tube can be contacted with each other at begin
                                                                    granularity=ad_granularity,
                                                                    seed_jnt_values=seed_conf,
                                                                    toggle_debug=False,
                                                                    type='source')
            if conf_list_depart is None:
                print(f"Cannot generate the linear part of the holding depart motion!")
                # uncomment to debug
                # self.robot_s.gen_meshmodel().attach_to(base)
                # self.robot_s.show_cdprimit()
                # for i in obstacle_list:
                #     i.attach_to(base)
                #     i.show_cdprimit()
                # base.run()
                self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                return None, None, None
            jawwidthlist_depart = self.gen_jawwidth_motion(conf_list_depart, jaw_width)
            objpose_list_depart = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_depart,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')
            if use_rrt:  # if use rrt, we shall find start and goal conf first and then perform rrt
                # approach linear
                seed_conf = conf_list_depart[-1]
                conf_list_approach = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                          goal_tcp_pos=goal_jaw_center_pos,
                                                                          goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                          direction=approach_direction,
                                                                          distance=approach_distance,
                                                                          obstacle_list=obstacle_list,
                                                                          granularity=ad_granularity,
                                                                          seed_jnt_values=seed_conf,
                                                                          type='sink')
                if conf_list_approach is None:
                    print(f"Cannot generate the linear part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
                conf_list_middle = self.rrtc_planner.plan(component_name=hand_name,
                                                          start_conf=conf_list_depart[-1],
                                                          goal_conf=conf_list_approach[0],
                                                          obstacle_list=obstacle_list,
                                                          otherrobot_list=[],
                                                          ext_dist=.07,
                                                          max_iter=300)
                if conf_list_middle is None:
                    print(f"Cannot generate the rrtc part of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
            else:  # if do not use rrt, we start from depart end to mid end and then approach from mid end to goal
                seed_conf = conf_list_depart[-1]
                self.robot_s.fk(component_name=hand_name, jnt_values=seed_conf)
                mid_start_tcp_pos, mid_start_tcp_rotmat = self.robot_s.get_gl_tcp(hand_name)
                mid_goal_tcp_pos = goal_jaw_center_pos - approach_direction * approach_distance
                mid_goal_tcp_rotmat = goal_jaw_center_rotmat
                conf_list_middle = self.inik_slvr.gen_linear_motion(component_name=hand_name,
                                                                    start_tcp_pos=mid_start_tcp_pos,
                                                                    start_tcp_rotmat=mid_start_tcp_rotmat,
                                                                    goal_tcp_pos=mid_goal_tcp_pos,
                                                                    goal_tcp_rotmat=mid_goal_tcp_rotmat,
                                                                    obstacle_list=obstacle_list,
                                                                    granularity=ad_granularity,
                                                                    seed_jnt_values=seed_conf)
                if conf_list_middle is None:
                    print(f"Cannot generate the linear motion middle of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
                # approach linear
                seed_conf = conf_list_middle[-1]
                conf_list_approach = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                          goal_tcp_pos=goal_jaw_center_pos,
                                                                          goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                          direction=approach_direction,
                                                                          distance=approach_distance,
                                                                          obstacle_list=[],
                                                                          # The test tube can be contacted with other test tubes at goal
                                                                          granularity=ad_granularity,
                                                                          seed_jnt_values=seed_conf,
                                                                          type='sink')
                if conf_list_approach is None:
                    print(f"Cannot generate the linear motion approach of the {i}th holding approach motion!")
                    self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                    self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                    return None, None, None
            jawwidthlist_approach = self.gen_jawwidth_motion(conf_list_approach, jaw_width)
            objpose_list_approach = self.gen_object_motion(component_name=hand_name,
                                                           conf_list=conf_list_approach,
                                                           obj_pos=rel_obj_pos,
                                                           obj_rotmat=rel_obj_rotmat,
                                                           type='relative')
            jawwidthlist_middle = self.gen_jawwidth_motion(conf_list_middle, jaw_width)
            objpose_list_middle = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_middle,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')
            conf_list = conf_list + conf_list_depart + conf_list_middle + conf_list_approach
            jawwidthlist = jawwidthlist + jawwidthlist_depart + jawwidthlist_middle + jawwidthlist_approach
            objpose_list = objpose_list + objpose_list_depart + objpose_list_middle + objpose_list_approach
            seed_conf = conf_list[-1]
        self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
        self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
        return conf_list, jawwidthlist, objpose_list

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
                                  approach_jawwidth=None,
                                  depart_jawwidth=None,
                                  ad_granularity=.01,
                                  use_rrt=True,
                                  obstacle_list=[],
                                  grasp_obstacle_list=[],
                                  use_incremental=False):
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
        if approach_jawwidth is None:
            approach_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]
        if depart_jawwidth is None:
            depart_jawwidth = self.robot_s.hnd_dict[hnd_name].jawwidth_rng[1]
        first_goal_pos = goal_homomat_list[0][:3, 3]
        first_goal_rotmat = goal_homomat_list[0][:3, :3]

        common_grasp_id_list, \
        symmetry_obj_goal_homomat_dict, \
        init_feasible_iks, \
        init_feasible_approach_iks = self.find_common_graspids_symmetric(hnd_name,
                                                                         tube,
                                                                         grasp_info_list,
                                                                         obj_init_homomat=
                                                                         goal_homomat_list[
                                                                             0],
                                                                         obj_goal_homomat=
                                                                         goal_homomat_list[
                                                                             1],
                                                                         toggle_debug=False,
                                                                         approach_direction_list=approach_direction_list,
                                                                         approach_distance_list=approach_distance_list,
                                                                         symmetry_discrete_factor=3,
                                                                         obstacle_list=obstacle_list)
        if len(common_grasp_id_list) == 0:
            print("No common grasp id at the given goal homomats!")
            return None, None, None, init_feasible_iks

        start_time = time.time()
        for _grasp_id_ind, grasp_id in enumerate(common_grasp_id_list):
            grasp_info = grasp_info_list[grasp_id]
            jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat = grasp_info
            # approach
            first_jaw_center_pos = first_goal_rotmat.dot(jaw_center_pos) + first_goal_pos
            first_jaw_center_rotmat = first_goal_rotmat.dot(jaw_center_rotmat)
            # objcm as an obstacle
            objcm_copy = tube_cm.copy()
            objcm_copy.set_pos(first_goal_pos)
            objcm_copy.set_rotmat(first_goal_rotmat)
            conf_list_approach, jawwidthlist_approach = \
                self.gen_approach_motion(component_name=hnd_name,
                                         goal_tcp_pos=first_jaw_center_pos,
                                         goal_tcp_rotmat=first_jaw_center_rotmat,
                                         start_conf=start_conf,
                                         approach_direction=approach_direction_list[0],
                                         approach_distance=approach_distance_list[0],
                                         approach_jawwidth=approach_jawwidth,
                                         granularity=ad_granularity,
                                         obstacle_list=obstacle_list,
                                         object_list=[objcm_copy],
                                         seed_jnt_values=init_feasible_iks[_grasp_id_ind])
            if conf_list_approach is None:
                print("Cannot generate the pick motion!")
                continue
            # middle
            for _symmetry_obj_goal_homomat in symmetry_obj_goal_homomat_dict[grasp_id]:
                last_goal_pos = _symmetry_obj_goal_homomat[:3, 3]
                last_goal_rotmat = _symmetry_obj_goal_homomat[:3, :3]
                conf_list_middle, jawwidthlist_middle, objpose_list_middle = \
                    self.gen_holding_moveto(hand_name=hnd_name,
                                            objcm=tube_cm,
                                            grasp_info=grasp_info,
                                            obj_pose_list=[goal_homomat_list[0], _symmetry_obj_goal_homomat],
                                            depart_direction_list=depart_direction_list,
                                            approach_direction_list=approach_direction_list,
                                            depart_distance_list=depart_distance_list,
                                            approach_distance_list=approach_distance_list,
                                            ad_granularity=ad_granularity,
                                            use_rrt=use_rrt,
                                            obstacle_list=obstacle_list,
                                            seed_jnt_values=conf_list_approach[-1])
                if conf_list_middle is None:
                    print("Cannot generate the hold moving motion")
                    continue
                # departure
                last_jaw_center_pos = last_goal_rotmat.dot(jaw_center_pos) + last_goal_pos
                last_jaw_center_rotmat = last_goal_rotmat.dot(jaw_center_rotmat)
                # objcm as an obstacle
                objcm_copy.set_pos(last_goal_pos)
                objcm_copy.set_rotmat(last_goal_rotmat)
                conf_list_depart, jawwidthlist_depart = \
                    self.gen_depart_motion(component_name=hnd_name,
                                           start_tcp_pos=last_jaw_center_pos,
                                           start_tcp_rotmat=last_jaw_center_rotmat,
                                           end_conf=end_conf,
                                           depart_direction=depart_direction_list[0],
                                           depart_distance=depart_distance_list[0],
                                           depart_jawwidth=depart_jawwidth,
                                           granularity=ad_granularity,
                                           obstacle_list=obstacle_list,
                                           object_list=[objcm_copy],
                                           seed_jnt_values=conf_list_middle[-1])
                if conf_list_depart is None:
                    print("Cannot generate the release motion!")
                    continue
                objpose_list_approach = self.gen_object_motion(component_name=hnd_name,
                                                               conf_list=jawwidthlist_approach,
                                                               obj_pos=first_goal_pos,
                                                               obj_rotmat=first_goal_rotmat,
                                                               type='absolute')
                objpose_list_depart = self.gen_object_motion(component_name=hnd_name,
                                                             conf_list=conf_list_depart,
                                                             obj_pos=last_goal_pos,
                                                             obj_rotmat=last_goal_rotmat,
                                                             type='absolute')
                return [conf_list_approach, conf_list_middle, conf_list_depart], \
                       [jawwidthlist_approach, jawwidthlist_middle, jawwidthlist_depart], \
                       [objpose_list_approach, objpose_list_middle, objpose_list_depart], \
                       init_feasible_iks

            # return conf_list_approach + conf_list_middle + conf_list_depart, \
            #        jawwidthlist_approach + jawwidthlist_middle + jawwidthlist_depart, \
            #        objpose_list_approach +  objpose_list_middle + objpose_list_depart
        print(time.time() - start_time)
        return None, None, None, init_feasible_iks


if __name__ == '__main__':
    import time
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    import modeling.collision_model as cm
    import grasping.annotation.utils as gutil
    import numpy as np
    import basis.robot_math as rm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    objcm = cm.CollisionModel('tubebig.stl')
    robot_s = ym.Yumi(enable_cc=True)
    manipulator_name = 'rgt_arm'
    hand_name = 'rgt_hnd'
    start_conf = robot_s.get_jnt_values(manipulator_name)
    goal_homomat_list = []
    for i in range(4):
        goal_pos = np.array([.55, -.1, .3]) - np.array([i * .1, i * .1, 0])
        # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
        goal_rotmat = np.eye(3)
        goal_homomat_list.append(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm = objcm.copy()
        tmp_objcm.set_rgba([1, 0, 0, .3])
        tmp_objcm.set_homomat(rm.homomat_from_posrot(goal_pos, goal_rotmat))
        tmp_objcm.attach_to(base)
    grasp_info_list = gutil.load_pickle_file(objcm_name='tubebig', file_name='yumi_tube_big.pickle')
    grasp_info = grasp_info_list[0]
    pp_planner = PickPlacePlanner(robot_s=robot_s)
    # goal_homomat_list = [np.array([[0.99797, 2.7551e-21, -0.063678, 0.34823],
    #                                [-2.5763e-05, 1, -0.00040375, -0.086992],
    #                                [0.063678, 0.00040458, 0.99797, -0.0092632],
    #                                [0, 0, 0, 1]]), np.array([[1, 0, 0, 0.30699],
    #                                                          [0, 1, 0, -0.17727],
    #                                                          [0, 0, 1, -0.010472],
    #                                                          [0, 0, 0, 1]])]
    conf_list, jawwidth_list, objpose_list = \
        pp_planner.gen_pick_and_place_motion(hnd_name=hand_name,
                                             tube_cm=objcm,
                                             grasp_info_list=grasp_info_list,
                                             goal_homomat_list=goal_homomat_list,
                                             start_conf=None,
                                             end_conf=None,
                                             depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
                                             approach_direction_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
                                             # depart_distance_list=[None] * len(goal_homomat_list),
                                             # approach_distance_list=[None] * len(goal_homomat_list),
                                             depart_distance_list=[.2] * len(goal_homomat_list),
                                             approach_distance_list=[.2] * len(goal_homomat_list),
                                             approach_jawwidth=None,
                                             depart_jawwidth=None,
                                             ad_granularity=.003,
                                             use_rrt=True,
                                             obstacle_list=[],
                                             use_incremental=False)
    # for grasp_info in grasp_info_list:
    #     conf_list, jawwidth_list, objpose_list = \
    #         pp_planner.gen_holding_moveto(hnd_name=hnd_name,
    #                                       objcm=objcm,
    #                                       grasp_info=grasp_info,
    #                                       obj_pose_list=goal_homomat_list,
    #                                       depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
    #                                       approach_direction_list=[np.array([0, 0, -1])] * len(goal_homomat_list),
    #                                       # depart_distance_list=[None] * len(goal_homomat_list),
    #                                       # approach_distance_list=[None] * len(goal_homomat_list),
    #                                       depart_distance_list=[.2] * len(goal_homomat_list),
    #                                       approach_distance_list=[.2] * len(goal_homomat_list),
    #                                       ad_granularity=.003,
    #                                       use_rrt=True,
    #                                       obstacle_list=[],
    #                                       seed_jnt_values=start_conf)
    #     print(robot_s.rgt_oih_infos, robot_s.lft_oih_infos)
    #     if conf_list is not None:
    #         break

    # animation
    robot_attached_list = []
    object_attached_list = []
    counter = [0]


    def update(robot_s,
               hand_name,
               objcm,
               robot_path,
               jawwidth_path,
               obj_path,
               robot_attached_list,
               object_attached_list,
               counter,
               task):
        if counter[0] >= len(robot_path):
            counter[0] = 0
        if len(robot_attached_list) != 0:
            for robot_attached in robot_attached_list:
                robot_attached.detach()
            for object_attached in object_attached_list:
                object_attached.detach()
            robot_attached_list.clear()
            object_attached_list.clear()
        pose = robot_path[counter[0]]
        robot_s.fk(hand_name, np.array(pose))
        robot_s.jaw_to(hand_name, jawwidth_path[counter[0]])
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        robot_attached_list.append(robot_meshmodel)
        obj_pose = obj_path[counter[0]]
        objb_copy = objcm.copy()
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        object_attached_list.append(objb_copy)
        counter[0] += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[robot_s,
                                     hand_name,
                                     objcm,
                                     conf_list,
                                     jawwidth_list,
                                     objpose_list,
                                     robot_attached_list,
                                     object_attached_list,
                                     counter],
                          appendTask=True)
    base.run()

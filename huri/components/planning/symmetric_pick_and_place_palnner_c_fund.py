import numpy as np

from huri.components.planning.symmetric_pick_and_place_planner import PickPlacePlanner, rm
from collections import namedtuple
from huri.definitions.utils_structure import MotionElement

from huri.components.planning.utils import _MOTION_SEG


def motion_seg_avaliable(motion_seg: _MOTION_SEG):
    if motion_seg.conf_ls is not None:
        return True
    else:
        return False


def motion_seg_comb(*args: _MOTION_SEG):
    new_name = ""
    new_conf_ls = []
    new_jaww_ls = []
    new_objpose_ls = []
    for _ in args:
        new_conf_ls += _.conf_ls
        new_jaww_ls += _.jaww_ls
        new_objpose_ls += _.objpose_ls
        new_name = f"{new_name}_{_.name}"
    return _MOTION_SEG(name=new_name,
                       conf_ls=new_conf_ls,
                       jaww_ls=new_jaww_ls,
                       objpose_ls=new_objpose_ls)


def motion_seg_to_element(motion_seg: _MOTION_SEG,
                          component_name: str,
                          obj_cm=None,
                          obs_list=None, ) -> MotionElement:
    return MotionElement(obj_cm=obj_cm,
                         objpose_list=motion_seg.objpose_ls,
                         conf_list=motion_seg.conf_ls,
                         jawwidth_list=motion_seg.jaww_ls,
                         hnd_name=component_name,
                         obs_list=obs_list, )


class PickPlacePlanner_CR(PickPlacePlanner):
    NO_APPROACH_MOTION = (None, None)
    NO_DEPART_MOTION = (None, None)
    NO_HOLD_MOTION = (None, None, None)
    RRT_EXT_DIS = .02

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
                            end_jawwidth=.0,
                            max_time=300) -> (_MOTION_SEG, _MOTION_SEG):
        if seed_jnt_values is None:
            seed_jnt_values = start_conf
        if approach_direction is None:
            approach_direction = goal_tcp_rotmat[:, 2]

        # if (np.rad2deg(rm.angle_between_vectors(approach_direction, -goal_tcp_rotmat[:, 0])) < np.deg2rad(50)) or\
        #         (np.rad2deg(rm.angle_between_vectors(approach_direction, goal_tcp_rotmat[:, 0])) < np.deg2rad(50)):
        #     approach_direction = approach_direction

        # approach_direction rm.unit_vector(approach_direction + goal_tcp_rotmat[:, 2]),
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
            return self.NO_APPROACH_MOTION
        pickapproach_motion_seg = _MOTION_SEG(name="pick_approach",
                                              conf_ls=conf_list,
                                              jaww_ls=jawwidth_list,
                                              objpose_ls=None)
        if start_conf is not None:
            start2approach_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                              start_conf=start_conf,
                                                              goal_conf=conf_list[0],
                                                              obstacle_list=obstacle_list + object_list,
                                                              ext_dist=self.RRT_EXT_DIS,
                                                              max_time=max_time,
                                                              smoothing_iterations=40, )
            if start2approach_conf_list is None:
                print("ADPlanner: Cannot plan approach motion!")
                return self.NO_APPROACH_MOTION
            start2approach_jawwidth_list = self.gen_jawwidth_motion(start2approach_conf_list, approach_jawwidth)
            start2pickapproach_motion_seg = _MOTION_SEG(name="start2pickapproach",
                                                        conf_ls=start2approach_conf_list,
                                                        jaww_ls=start2approach_jawwidth_list,
                                                        objpose_ls=None)
        else:
            start2pickapproach_motion_seg = _MOTION_SEG(name="start2pickapproach",
                                                        conf_ls=None,
                                                        jaww_ls=None,
                                                        objpose_ls=None)
        return start2pickapproach_motion_seg, pickapproach_motion_seg

    def gen_holding_moveto(self,
                           hand_name,
                           objcm,
                           jaw_width,
                           obj_pose_list,
                           depart_direction_list,
                           depart_distance_list,
                           approach_direction_list,
                           approach_distance_list,
                           first_conf,
                           goal_conf,
                           ad_granularity=.007,
                           use_rrt=True,
                           obstacle_list=[], ) -> (_MOTION_SEG, _MOTION_SEG, _MOTION_SEG):
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

        # hold object
        first_obj_pos = obj_pose_list[0][:3, 3]
        first_obj_rotmat = obj_pose_list[0][:3, :3]

        self.robot_s.fk(component_name=hand_name, jnt_values=goal_conf)
        goal_jaw_center_pos, goal_jaw_center_rotmat = self.robot_s.get_gl_tcp(hand_name)
        self.robot_s.fk(component_name=hand_name, jnt_values=first_conf)
        start_jaw_center_pos, start_jaw_center_rotmat = self.robot_s.get_gl_tcp(hand_name)
        # set a copy of the object to the start pose, hold the object, and move it to goal object pose
        objcm_copy = objcm.copy()
        objcm_copy.set_pos(first_obj_pos)
        objcm_copy.set_rotmat(first_obj_rotmat)
        rel_obj_pos, rel_obj_rotmat = self.robot_s.hold(hand_name, objcm_copy, jaw_width)
        seed_conf = first_conf

        depart_direction = depart_direction_list[0]
        if depart_direction is None:
            depart_direction = -start_jaw_center_rotmat[:, 2]
        depart_distance = depart_distance_list[0]
        if depart_distance is None:
            depart_distance = 0
        approach_direction = approach_direction_list[1]
        if approach_direction is None:
            approach_direction = goal_jaw_center_rotmat[:, 2]
        approach_distance = approach_distance_list[1]
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
            return self.NO_HOLD_MOTION
        jawwidthlist_depart = self.gen_jawwidth_motion(conf_list_depart, jaw_width)
        objpose_list_depart = self.gen_object_motion(component_name=hand_name,
                                                     conf_list=conf_list_depart,
                                                     obj_pos=rel_obj_pos,
                                                     obj_rotmat=rel_obj_rotmat,
                                                     type='relative')
        pickdepart_motion_seg = _MOTION_SEG(name="pickdepart",
                                            conf_ls=conf_list_depart,
                                            jaww_ls=jawwidthlist_depart,
                                            objpose_ls=objpose_list_depart)
        seed_conf = conf_list_depart[-1]
        ##place motion start
        if use_rrt:  # if use rrt, we shall find start and goal conf first and then perform rrt
            # approach linear
            conf_list_approach = self.inik_slvr.gen_rel_linear_motion(component_name=hand_name,
                                                                      goal_tcp_pos=goal_jaw_center_pos,
                                                                      goal_tcp_rotmat=goal_jaw_center_rotmat,
                                                                      direction=approach_direction,
                                                                      distance=approach_distance,
                                                                      obstacle_list=[],
                                                                      granularity=ad_granularity,
                                                                      seed_jnt_values=seed_conf,
                                                                      type='sink')
            if conf_list_approach is None:
                print(f"Cannot generate the linear part of the holding approach place motion!")
                self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                return (pickdepart_motion_seg, *self.NO_HOLD_MOTION[1:])
            jawwidthlist_approach = self.gen_jawwidth_motion(conf_list_approach, jaw_width)
            objpose_list_approach = self.gen_object_motion(component_name=hand_name,
                                                           conf_list=conf_list_approach,
                                                           obj_pos=rel_obj_pos,
                                                           obj_rotmat=rel_obj_rotmat,
                                                           type='relative')
            placeapproach_motion_seg = _MOTION_SEG(name="place_approach",
                                                   conf_ls=conf_list_approach,
                                                   jaww_ls=jawwidthlist_approach,
                                                   objpose_ls=objpose_list_approach)
            # rrtc
            conf_list_middle = self.rrtc_planner.plan(component_name=hand_name,
                                                      start_conf=conf_list_depart[-1],
                                                      goal_conf=conf_list_approach[0],
                                                      obstacle_list=[],
                                                      otherrobot_list=[],
                                                      ext_dist=self.RRT_EXT_DIS,
                                                      max_iter=300,
                                                      smoothing_iterations=30, )
            if conf_list_middle is None:
                print(f"Cannot generate the rrtc part of the holding approach motion!")
                self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                return (pickdepart_motion_seg, *self.NO_HOLD_MOTION[1:])
            jawwidthlist_middle = self.gen_jawwidth_motion(conf_list_middle, jaw_width)
            objpose_list_middle = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_middle,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')
            pickdepart2placeapproach_motion_seg = _MOTION_SEG(name="pickdepart2placeapproach",
                                                              conf_ls=conf_list_middle,
                                                              jaww_ls=jawwidthlist_middle,
                                                              objpose_ls=objpose_list_middle)
                # return self.NO_HOLD_MOTION
        else:  # if do not use rrt, we start from depart end to mid end and then approach from mid end to goal
            self.robot_s.fk(component_name=hand_name, jnt_values=seed_conf)
            mid_start_tcp_pos, mid_start_tcp_rotmat = self.robot_s.get_gl_tcp(hand_name)
            mid_goal_tcp_pos = goal_jaw_center_pos - approach_direction * approach_distance
            mid_goal_tcp_rotmat = goal_jaw_center_rotmat
            conf_list_middle = self.inik_slvr.gen_linear_motion(component_name=hand_name,
                                                                start_tcp_pos=mid_start_tcp_pos,
                                                                start_tcp_rotmat=mid_start_tcp_rotmat,
                                                                goal_tcp_pos=mid_goal_tcp_pos,
                                                                goal_tcp_rotmat=mid_goal_tcp_rotmat,
                                                                obstacle_list=[],
                                                                granularity=ad_granularity,
                                                                seed_jnt_values=seed_conf)
            if conf_list_middle is None:
                print(f"Cannot generate the linear motion middle of the holding approach pick motion!")
                self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                return (pickdepart_motion_seg, *self.NO_HOLD_MOTION[1:])
            jawwidthlist_middle = self.gen_jawwidth_motion(conf_list_middle, jaw_width)
            objpose_list_middle = self.gen_object_motion(component_name=hand_name,
                                                         conf_list=conf_list_middle,
                                                         obj_pos=rel_obj_pos,
                                                         obj_rotmat=rel_obj_rotmat,
                                                         type='relative')
            pickdepart2placeapproach_motion_seg = _MOTION_SEG(name="pickdepart2placeapproach",
                                                              conf_ls=conf_list_middle,
                                                              jaww_ls=jawwidthlist_middle,
                                                              objpose_ls=objpose_list_middle)
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
                print(f"Cannot generate the linear motion approach of the holding approach pick motion!")
                self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
                self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
                return (pickdepart_motion_seg, *self.NO_HOLD_MOTION[1:])
            jawwidthlist_approach = self.gen_jawwidth_motion(conf_list_approach, jaw_width)
            objpose_list_approach = self.gen_object_motion(component_name=hand_name,
                                                           conf_list=conf_list_approach,
                                                           obj_pos=rel_obj_pos,
                                                           obj_rotmat=rel_obj_rotmat,
                                                           type='relative')
            placeapproach_motion_seg = _MOTION_SEG(name="place_approach",
                                                   conf_ls=conf_list_approach,
                                                   jaww_ls=jawwidthlist_approach,
                                                   objpose_ls=objpose_list_approach)

        self.robot_s.release(hand_name, objcm_copy, jawwidth_bk)
        self.robot_s.fk(component_name=hand_name, jnt_values=jnt_values_bk)
        return pickdepart_motion_seg, pickdepart2placeapproach_motion_seg, placeapproach_motion_seg

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
                          begin_jawwidth=.0) -> (_MOTION_SEG, _MOTION_SEG):
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
            return self.NO_DEPART_MOTION
        placedepart_motion_seg = _MOTION_SEG(name="placedepart",
                                             conf_ls=conf_list,
                                             jaww_ls=jawwidth_list,
                                             objpose_ls=None)
        if end_conf is not None:
            depart2goal_conf_list = self.rrtc_planner.plan(component_name=component_name,
                                                           start_conf=conf_list[-1],
                                                           goal_conf=end_conf,
                                                           obstacle_list=obstacle_list + object_list,
                                                           ext_dist=self.RRT_EXT_DIS,
                                                           max_time=300,
                                                           smoothing_iterations=30, )
            if depart2goal_conf_list is None:
                print("ADPlanner: Cannot plan depart motion!")
                return self.NO_DEPART_MOTION
            depart2goal_jawwidth_list = self.gen_jawwidth_motion(depart2goal_conf_list, depart_jawwidth)
            placedepart2goal_motion_seg = _MOTION_SEG(name="placedepart2goal",
                                                      conf_ls=depart2goal_conf_list,
                                                      jaww_ls=depart2goal_jawwidth_list,
                                                      objpose_ls=None)
        else:
            placedepart2goal_motion_seg = _MOTION_SEG(name="placedepart2goal",
                                                      conf_ls=None,
                                                      jaww_ls=None,
                                                      objpose_ls=None)
        return placedepart_motion_seg, placedepart2goal_motion_seg

import pdb

from utils import init_env_rbt, np, MotionElement, MotionBatch
from huri.definitions.rack_def import TubeRack, Rack_Hard_Proto, TubeType
from tubesolseq import ArrangeTubeSolSeq
from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver
from huri.core.print_tool import text_pd, print_with_border
import huri.core.file_sys as fs
import huri.components.planning.symmetric_pick_and_place_planner_simple as ppp
from huri.learning.env.arrangement_planning_rack.utils import isdone
from animation import show_animation
from dual_d405_manager import RealSenseD405
import basis.robot_math as rm
from vision_sys import RackLocator, detect_tube, is_pose_err
from huri.math.math import combination
from huri.components.vision.tube_detector import TestTubeDetector, extract
import modeling.geometric_model as gm
import threading
from queue import Queue
from constants20240215 import (TCP2EYE_MAT_DEVICE_1,
                               TCP2EYE_MAT_DEVICE_2,
                               HEIGHT_RANGE,
                               GOAL_PATTERN,
                               GOAL_PATTERN1,
                               GOAL_PATTERN2,
                               GOAL_UP_DISTANCE,
                               APPROACH_DISTANCE_LIST,
                               DEPART_DISTANCE_LIST,
                               SPEED,
                               ACC,
                               TOGGLE_RACK_LOCATOR,
                               OBS_SLOT_1,
                               OBS_SLOT_2,
                               TOGGLE_YOLO,
                               OBS_HEIGHT)
# GUI Related
from direct.gui.DirectGui import *
from direct.gui.DirectFrame import DirectFrame
from panda3d.core import (Filename, TextNode,
                          LPoint3f,
                          LVecBase3f,
                          LVecBase4f,
                          TextNode)
from direct.stdpy import threading
from huri.components.gui.select_pattern import PatternSelectGui

# import pydevd_pycharm

GOAL_PATTERN = GOAL_PATTERN
TOGGLE_VIS_DEBUG = True


def _E(state, goal_pattern, number_class):
    entropy = np.zeros(len(number_class), dtype=int)
    for i, _ in enumerate(number_class):
        entropy[i] = len(np.where(state[goal_pattern == _] == _)[0])
    return entropy


def E(state, goal_pattern):
    number_class = np.unique(state)
    number_class = number_class[number_class > 0]
    return _E(state, goal_pattern, number_class)


def task_planning(tube_rack: TubeRack, solver: D3QNSolver, condition_set,
                  infeasible_info_dict: dict, ) -> ArrangeTubeSolSeq:
    raw_solution, solution = solver.solve(init_state=tube_rack.rack_status,
                                          max_iter=600,
                                          condition_set=condition_set,
                                          infeasible_action_dict=infeasible_info_dict,
                                          toggle_show=False)
    task_sols = ArrangeTubeSolSeq(rack=tube_rack,
                                  solution=solution,
                                  tf=tube_rack.get_homomat())
    return task_sols


def init_vision_system(rl: RackLocator, tube_detector, rbtx, add_to_scene_func=None, toggle_debug=False):
    # x_range = np.array([.3])
    # y_range = np.array([0.07])
    # z_range = np.array([OBS_HEIGHT])
    # r = combination([x_range, y_range, z_range]).tolist()
    # rl.add_obs_pos(obs_pos=r)
    rbtx.homeconf()     # move to home position
    c_pcd, pcds = rl.scan(toggle=False) # scan the scene, c_pcd is colorful point cloud, pcds is the point cloud
    rack, rack_tf = rl.locate_rack(c_pcd, Rack_Hard_Proto, height_range=HEIGHT_RANGE,
                                   downsampling_voxelsize=.005,
                                   toggle_debug=toggle_debug)   # rack -> huri/definitions/rack_def.py, rack_tf -> homomat
    # pcd_rgba, im_center = rl.obs_in_rack_center(rack_tf, toggle=TOGGLE_RACK_LOCATOR)

    obs_p_1 = rack.get_slot_homomat(OBS_SLOT_1)
    new_rbt_obs_pos_1 = rl.get_obs_xy_point(obs_p_1[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_1[:3, 3],
                new_rbt_obs_pos_1[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_1, im_1 = rl.get_pcd_im()

    obs_p_2 = rack.get_slot_homomat(OBS_SLOT_2)
    new_rbt_obs_pos_2 = rl.get_obs_xy_point(obs_p_2[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_2[:3, 3],
                new_rbt_obs_pos_2[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_2, im_2 = rl.get_pcd_im()

    rack_instance = Rack_Hard_Proto
    pcd_w = np.vstack((pcd_1, pcd_2,))
    rack, rack_tf = rl.locate_rack(pcd_w, Rack_Hard_Proto, height_range=HEIGHT_RANGE, )
    for pcd, im in [(pcd_1, im_1), (pcd_2, im_2), ]:
        rack_instance, rack_tf = detect_tube(pcd[:, :3], im,
                                             rack_proto=rack_instance,
                                             rack_tf=rack_tf,
                                             detector=tube_detector,
                                             toggle_yolo=False,
                                             toggle_detect_tube_pos=True)
        add_to_scene_func(gm.gen_rgb_pointcloud(pcd_w))
    # if np.sum(E(rack_instance.rack_status, GOAL_PATTERN2)) > np.sum(E(rack_instance.rack_status, GOAL_PATTERN1)):
    #     rack_instance.rack_status = np.fliplr(rack_instance.rack_status)
    #     rack_instance.rack_status_confidence = np.fliplr(rack_instance.rack_status_confidence)
    #     rack_instance.tubes_pose = np.fliplr(rack_instance.tubes_pose)

    if TOGGLE_VIS_DEBUG:
        fs.dump_pickle([(pcd_1, im_1), (pcd_2, im_2), ], "pcd_data_debug", reminder=False)
    return rack_instance, rack_tf


def next_vision_system(rack_tf, rl, detector, rbtx, add_to_scene_func=None, remove_scene_func=None, toggle_debug=False):
    if remove_scene_func is not None:
        remove_scene_func()
    # pcd_rgba, im = rl.obs_in_rack_center(rack_tf, toggle=False)
    # rack, rack_tf = rl.locate_rack(pcd_rgba[:, :3], Rack_Hard_Proto, height_range=HEIGHT_RANGE,
    #                                toggle_debug=toggle_debug)
    # pcd_rgba, im_center = rl.obs_in_rack_center(rack_tf, toggle=TOGGLE_RACK_LOCATOR)
    rack = Rack_Hard_Proto.copy()
    rack.set_homomat(rack_tf)

    obs_p_1 = rack.get_slot_homomat(OBS_SLOT_1)
    new_rbt_obs_pos_1 = rl.get_obs_xy_point(obs_p_1[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_1[:3, 3],
                new_rbt_obs_pos_1[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_1, im_1 = rl.get_pcd_im()

    obs_p_2 = rack.get_slot_homomat(OBS_SLOT_2)
    new_rbt_obs_pos_2 = rl.get_obs_xy_point(obs_p_2[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_2[:3, 3],
                new_rbt_obs_pos_2[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_2, im_2 = rl.get_pcd_im()

    rack_instance = Rack_Hard_Proto
    pcd_w = np.vstack((pcd_1, pcd_2,))
    rack, rack_tf = rl.locate_rack(pcd_w, Rack_Hard_Proto, height_range=HEIGHT_RANGE, )
    for pcd, im in [(pcd_1, im_1), (pcd_2, im_2), ]:
        rack_instance, rack_tf = detect_tube(pcd[:, :3], im,
                                             rack_proto=rack_instance,
                                             rack_tf=rack_tf,
                                             detector=detector,
                                             toggle_yolo=False,
                                             toggle_detect_tube_pos=True)
        add_to_scene_func(gm.gen_rgb_pointcloud(pcd_w))
    # if np.sum(E(rack_instance.rack_status, GOAL_PATTERN2)) > np.sum(E(rack_instance.rack_status, GOAL_PATTERN1)):
    #     pass
    return rack_instance, rack_tf


def vision_system(rl, detector):
    pcd, im = rl.get_pcd_im()
    rack, rack_tf = rl.locate_rack(pcd, Rack_Hard_Proto, height_range=HEIGHT_RANGE,
                                   toggle_debug=False)
    rack_instance, rack_tf = detect_tube(pcd, im,
                                         rack_proto=Rack_Hard_Proto,
                                         rack_tf=rack_tf,
                                         detector=detector,
                                         toggle_yolo=TOGGLE_YOLO)
    return rack_instance, rack_tf


class MotionRunner(threading.Thread):
    def __init__(self, rbthp, rl, data_queue, ret_queue, detector, grasp_cache_dict, pp_planner):
        super().__init__()
        self.rbthp = rbthp
        self.rbt_x = rbthp.rbt_con
        self.rbt_s = rbthp.sim_rbt
        self.rl = rl
        self.data_queue = data_queue
        self.ret_queue = ret_queue
        self.detector = detector
        self.grasp_cache_dict = grasp_cache_dict
        self.pp_planner = pp_planner
        self.tube_rack_exe = None
        self.add_plot_func = None
        self.remove_plot_func = None

    def run(self):
        while True:
            # try:
            #     data = self.data_queue.get(False)
            # except queue.Empty:
            #     time.sleep(.1)
            #     continue
            data = self.data_queue.get()
            if data == "end":
                self.ret_queue.put(False)
                # break
            if isinstance(data, str):
                print(data)
                continue
            el = data
            need_replanning, el, rack_tf = self.pick_motion(el)
            if need_replanning:
                with self.data_queue.mutex:
                    self.data_queue.queue.clear()
                self.ret_queue.put(True)

            else:
                need_replanning, el = self.place_motion(el, rack_tf)
            if need_replanning:
                with self.data_queue.mutex:
                    self.data_queue.queue.clear()
                self.ret_queue.put(True)
            psg.set_bg(GOAL_PATTERN)
            psg.set_fg(self.tube_rack_exe.rack_status)

    def pick_motion(self, el):
        rl = self.rl
        detector = self.detector
        tube_rack_exe = self.tube_rack_exe
        rbthp = self.rbthp
        grasp_cache_dict = self.grasp_cache_dict
        pp_planner = self.pp_planner
        for i in range(3):
            h1 = rbthp.hnd_model_at(el.pick_approach[:3, 3], el.pick_approach[:3, :3])
            self.add_plot_func(h1.gen_meshmodel(rgba=[191 / 255, 164 / 255, 111 / 255, .7]))
            need_replanning = True
            self.open_gripper()
            self.rbt_x.move_p(el.pick_approach[:3, 3],
                              el.pick_approach[:3, :3],
                              speed=350,
                              mvacc=ACC)

            _rack, _rack_tf, pcd_w, im = self.rl.track(Rack_Hard_Proto,
                                                       rack_init_pose=self.tube_rack_exe.get_homomat(),
                                                       height_range=HEIGHT_RANGE,
                                                       rack_state=tube_rack_exe.rack_status,
                                                       toggle_pcd=False)
            if TOGGLE_VIS_DEBUG:
                fs.dump_pickle([tube_rack_exe.rack_status, el.rack_tf], "rack_states", reminder=False)
            # exit(0)
            _rack, __rack_tf = detect_tube(pcd_w[:, :3],
                                           im,
                                           rack_proto=Rack_Hard_Proto,
                                           rack_tf=_rack_tf,
                                           detector=detector,
                                           toggle_yolo=TOGGLE_YOLO,
                                           toggle_detect_tube_pos=True)
            if _rack.rack_status[tuple(el.init_slot_id)] != el.tube.type:
                new_rbt_obs_pos = rl.get_obs_xy_point(_rack.get_slot_homomat(el.init_slot_id)[:2, 3], )
                self.rbt_x.move_p(new_rbt_obs_pos[:3, 3],
                                  new_rbt_obs_pos[:3, :3],
                                  speed=SPEED,
                                  mvacc=ACC)
                _rack, _rack_tf, pcd_w, im = rl.track(Rack_Hard_Proto,
                                                      rack_init_pose=el.rack_tf,
                                                      height_range=HEIGHT_RANGE,
                                                      rack_state=tube_rack_exe.rack_status,
                                                      toggle_pcd=False)
                _rack, __rack_tf = detect_tube(pcd_w[:, :3],
                                               im,
                                               rack_proto=Rack_Hard_Proto,
                                               rack_tf=_rack_tf,
                                               detector=detector,
                                               toggle_yolo=TOGGLE_YOLO,
                                               toggle_detect_tube_pos=True)
                # Rack status
                if _rack.rack_status[tuple(el.init_slot_id)] != el.tube.type:
                    print("NEED REPLANNING")
                    need_replanning = True
                    break
            tube_homomat = _rack.get_slot_homomat(el.init_slot_id)
            grasp = el.grasp_info_list[el.grasp_id]
            init_jaw_center_pos = tube_homomat[:3, 3] + tube_homomat[:3, :3].dot(grasp[1])
            init_jaw_center_rotmat = tube_homomat[:3, :3].dot(grasp[2])
            _rack_cm, _tube_cm = _rack.gen_collision_model(gen_tube=True, center=el.init_slot_id)
            _, _potential_tube_cm = tube_rack_exe.gen_collision_model(gen_tube=True, center=el.init_slot_id)
            is_collded = rbthp.check_grasp_collision_at_pose(init_jaw_center_pos, init_jaw_center_rotmat,
                                                             [_rack_cm, *_tube_cm.cm_list,
                                                              *_potential_tube_cm.cm_list], toggle=False)
            if is_collded:
                grasp_info_list = grasp_cache_dict[el.tube.type]

                approach_direction_list = [-_rack_tf[:3, 2],
                                           rm.unit_vector(el.pick[:3, 3] - el.pick_approach[:3, 3])]
                depart_distance_list = DEPART_DISTANCE_LIST,
                # DEPART_DISTANCE_LIST
                approach_distance_list = APPROACH_DISTANCE_LIST[el.tube.type]
                _goal_pos = tube_rack_exe.get_slot_homomat(el.goal_slot_id)
                _goal_pos[:3, 3] = _goal_pos[:3, 3] + _rack_tf[:3, 2] * GOAL_UP_DISTANCE[el.tube.type]
                _common_grasp_iter = pp_planner.common_grasp_iter_gen.gen_iterator(grasp_info_list=grasp_info_list,
                                                                                   hand_name="arm",
                                                                                   obj_init_homomat=tube_homomat,
                                                                                   sym_obj_tgt_homomats_angles=el.tube.get_symmetry_homomat_angle(
                                                                                       3, rot_mat=_goal_pos),
                                                                                   pick_approach_pos_offset=
                                                                                   approach_direction_list[0] *
                                                                                   approach_distance_list[0] if
                                                                                   approach_direction_list[
                                                                                       0] is not None else
                                                                                   approach_distance_list[0],
                                                                                   place_approach_pos_offset=
                                                                                   approach_direction_list[1] *
                                                                                   approach_distance_list[1] if
                                                                                   approach_direction_list[
                                                                                       1] is not None else
                                                                                   approach_distance_list[1],
                                                                                   obstacle_list=[_rack_cm,
                                                                                                  *_tube_cm.cm_list,
                                                                                                  *
                                                                                                  tube_rack_exe.gen_collision_model(
                                                                                                      gen_tube=True)[
                                                                                                      1].cm_list],
                                                                                   toggle_debug=False)
                _common_grasp_iter.set_start_cnt(el.grasp_id + 1)
                for gid, pp in enumerate(_common_grasp_iter):
                    if pp is None:
                        continue
                    if pp[0]:
                        grasp_id = gid
                        init_tcp_homomat, init_tcp_approach_homomat, init_jaw_width, tgt_tcp_homomat_list, tgt_approach_jnt_val_list, tgt_tcp_sym_angles, tgt_jawwidth_list = pp[
                                                                                                                                                                              1:]

                        el = MotionElement(init_homo=init_tcp_homomat,
                                           init_approach_homo=init_tcp_approach_homomat,
                                           init_jawwidth=init_jaw_width,
                                           tgt_homo_list=tgt_tcp_homomat_list,
                                           tgt_approach_homo_list=tgt_approach_jnt_val_list,
                                           tgt_jawwidth_list=tgt_jawwidth_list,
                                           tgt_tcp_sym_angles=tgt_tcp_sym_angles,
                                           grasp_id=grasp_id,
                                           move_tube=el.tube,
                                           grasp_info_list=grasp_info_list,
                                           rack_tf=_rack.get_homomat(),
                                           init_slot_id=el.init_slot_id,
                                           goal_slot_id=el.goal_slot_id,
                                           hnd_name="arm",
                                           obs_list=None, )

                        tube_homomat = _rack.get_slot_homomat(el.init_slot_id)
                        grasp = grasp_cache_dict[el.tube.type][el.grasp_id]
                        init_jaw_center_pos = tube_homomat[:3, 3] + tube_homomat[:3, :3].dot(grasp[1])
                        init_jaw_center_rotmat = tube_homomat[:3, :3].dot(grasp[2])
                        break
                else:
                    print("NEED REPLANNING")
                    need_replanning = True
                    # TODO need replanning and add constraints to task planning

            self.remove_plot_func()
            rbthp.sync()
            self.add_plot_func(rbthp.gen_mesh_model())
            self.add_plot_func(gm.gen_rgb_pointcloud(pcd_w))
            mdl = _rack.gen_mesh_model(gen_tube=True)
            self.add_plot_func(self.tube_rack_exe.gen_mesh_model(gen_tube=True, alpha=.5))
            self.add_plot_func(mdl)

            new_approach_pos = init_jaw_center_pos + el.pick_approach[:3, 3] - el.pick[:3, 3]
            new_depart_pos = init_jaw_center_pos + _rack_tf[:3, 2] * DEPART_DISTANCE_LIST[0]

            # plot robot move
            # j1 = rbthp.sim_rbt.ik("arm", new_approach_pos, init_jaw_center_rotmat)
            # h1 = rbthp.hnd_model_at(new_approach_pos, init_jaw_center_rotmat)
            # j2 = rbthp.sim_rbt.ik("arm", init_jaw_center_pos, init_jaw_center_rotmat)
            # h2 = rbthp.hnd_model_at(init_jaw_center_pos, init_jaw_center_rotmat)
            # j3 = rbthp.sim_rbt.ik("arm", new_depart_pos, init_jaw_center_rotmat)
            # h3 = rbthp.hnd_model_at(new_depart_pos, init_jaw_center_rotmat)
            # rbthp.sim_rbt.fk("arm", j1)
            # self.add_plot_func(h1.gen_meshmodel(rgba=[214 / 255, 198 / 255, 75 / 255, .5]))
            # self.add_plot_func(h2.gen_meshmodel())
            # rbthp.sim_rbt.fk("arm", j2)
            # rbthp.sim_rbt.fk("arm", j3)
            # self.add_plot_func(h3)
            h3 = self.rbthp.hnd_model_at(el.place_approach[:3, 3], el.place_approach[:3, :3])
            self.add_plot_func(h3.gen_meshmodel(rgba=[191 / 255, 164 / 255, 111 / 255, .7]))
            # robot move
            rbthp.rbt_con.move_p(new_approach_pos,
                                 init_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
            rbthp.rbt_con.move_p(init_jaw_center_pos, init_jaw_center_rotmat, speed=100, mvacc=ACC)
            self.close_gripper()
            if rbthp.rbt_con.get_gripper_width() < 0.005:
                continue
            rbthp.rbt_con.move_p(new_depart_pos, init_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
            is_not_grasped = rbthp.rbt_con.get_gripper_width() < 0.005
            if is_not_grasped:
                continue
            need_replanning = False
            break
        else:
            self.open_gripper()
            rbthp.rbt_con.move_p(new_depart_pos, init_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
        if not need_replanning:
            self.tube_rack_exe.set_homomat(_rack_tf)
        return need_replanning, el, _rack_tf

    def place_motion(self, el, rack_tf):
        rbt_x = self.rbt_x
        rl = self.rl
        grasp = el.grasp_info_list[el.grasp_id]
        tube_rack_exe = self.tube_rack_exe

        rbt_x.move_p(el.place_approach[:3, 3], el.place_approach[:3, :3], speed=300, mvacc=ACC)

        _rack, _rack_tf, pcd_w, im = rl.track(Rack_Hard_Proto, rack_init_pose=rack_tf,
                                              rack_state=tube_rack_exe.rack_status,
                                              height_range=HEIGHT_RANGE,
                                              maximum_distance=.001)
        if is_pose_err(rack_tf, _rack_tf):
            new_rbt_obs_pos = rl.get_obs_xy_point(_rack.get_slot_homomat(el.goal_slot_id)[:2, 3], )
            self.rbt_x.move_p(new_rbt_obs_pos[:3, 3],
                              new_rbt_obs_pos[:3, :3],
                              speed=SPEED,
                              mvacc=ACC)
            _rack, _rack_tf, pcd_w, im = rl.track(Rack_Hard_Proto, rack_init_pose=rack_tf,
                                                  rack_state=tube_rack_exe.rack_status,
                                                  height_range=HEIGHT_RANGE)
        goal_slot_p = _rack.get_slot_homomat(el.goal_slot_id)

        self.remove_plot_func()
        self.rbthp.sync()
        self.add_plot_func(self.rbthp.gen_mesh_model())
        self.add_plot_func(gm.gen_rgb_pointcloud(pcd_w))
        self.add_plot_func(gm.gen_arrow(spos=goal_slot_p[:3, 3] + goal_slot_p[:3, 2] * .12,
                                        epos=goal_slot_p[:3, 3] + goal_slot_p[:3, 2] * .05))
        mdl = _rack.gen_mesh_model(gen_tube=False)
        self.add_plot_func(self.tube_rack_exe.gen_mesh_model(gen_tube=True, alpha=.5))
        self.add_plot_func(mdl)

        tube_homomat = _rack.get_slot_homomat(el.goal_slot_id)

        tube_homomat = el.tube.get_symmetry_homomat_at_angle(el.tgt_tcp_sym_angles[0], tube_homomat)

        goal_jaw_center_pos = tube_homomat[:3, 3] + tube_homomat[:3, :3].dot(grasp[1]) + _rack_tf[:3,
                                                                                         2] * GOAL_UP_DISTANCE[
                                  el.tube.type]
        goal_jaw_center_rotmat = tube_homomat[:3, :3].dot(grasp[2])
        tgt_new_approach_pos = goal_jaw_center_pos + el.place_approach[:3, 3] - el.place[:3, 3]

        # plot robot move
        # j1 = self.rbthp.sim_rbt.ik("arm", tgt_new_approach_pos, goal_jaw_center_rotmat)
        # j2 = self.rbthp.sim_rbt.ik("arm", goal_jaw_center_pos, goal_jaw_center_rotmat)
        # j3 = self.rbthp.sim_rbt.ik("arm", tgt_new_approach_pos, goal_jaw_center_rotmat)
        # self.rbthp.sim_rbt.fk("arm", j1)
        # self.add_plot_func(self.rbthp.gen_mesh_model(alpha=.3))
        # self.rbthp.sim_rbt.fk("arm", j2)
        # self.add_plot_func(self.rbthp.gen_mesh_model(alpha=.3))
        # self.rbthp.sim_rbt.fk("arm", j3)
        # self.add_plot_func(self.rbthp.gen_mesh_model(alpha=.3))

        # h1 = self.rbthp.hnd_model_at(tgt_new_approach_pos, goal_jaw_center_rotmat)
        # h2 = self.rbthp.hnd_model_at(goal_jaw_center_pos, goal_jaw_center_rotmat)
        # self.add_plot_func(h1.gen_meshmodel(rgba=[191 / 255, 164 / 255, 111 / 255, .3]))
        # self.add_plot_func(h2.gen_meshmodel(rgba=[114 / 255, 98 / 255, 80 / 255, .3]))

        # robot move
        rbt_x.move_p(tgt_new_approach_pos,
                     goal_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
        rbt_x.move_p(goal_jaw_center_pos, goal_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
        self.open_gripper()
        rbt_x.move_p(tgt_new_approach_pos, goal_jaw_center_rotmat, speed=SPEED, mvacc=ACC)

        # update tube_rack exe
        tube_rack_exe.remove_slot(el.init_slot_id)
        tube_rack_exe.insert_tube(el.goal_slot_id, el.tube)
        # pydevd_pycharm.settrace('localhost', port=54321, stdoutToServer=True, stderrToServer=True)
        return False, el

    def open_gripper(self):
        self.rbt_x.set_gripper_width(0.025, speed=300)

    def close_gripper(self):
        self.rbt_x.close_gripper(speed=80)

    def update_tube_rack_exe(self, tube_rack_exe):
        self.tube_rack_exe = tube_rack_exe

    def update_add_plot_func(self, func, remove_func):
        self.add_plot_func = func
        self.remove_plot_func = remove_func


import os


def to_gui_size(tgt_width, img_wh):
    gui_height = img_wh[1] / (img_wh[0] / tgt_width)
    return (tgt_width, 1, gui_height)


class HLabGUI(object):
    """
    the graphical user interface of the application

    author: weiwei
    date: 20180925
    """

    def __init__(self, scenarioctrl=None):
        self.scctrl = scenarioctrl
        this_dir, this_filename = os.path.split(__file__)
        # self.imageObject = OnscreenImage(image="./gui/banner250x1080.png", pos=(1.55, 0, 0), scale=(250 / 1080.0, 1, 1))

        self.pg234 = DirectFrame(
            frameSize=(0, 1, 0, 1),
            frameColor=(1.0, .01, 1.0, 1.0),
            pos=LPoint3f(1.25, 0, 1),
            parent=None,
        )
        self.pg234.setTransparency(0)

        tgt_width = 0.23
        wrs_logo_2022_size = to_gui_size(tgt_width, (1805, 1331))
        self.wrs_logo = OnscreenImage(image="./gui/wrs_logo_2022.png",
                                      pos=(wrs_logo_2022_size[0] + .1, 0, -wrs_logo_2022_size[0]),
                                      scale=wrs_logo_2022_size,
                                      parent=self.pg234)
        self.wrs_logo.setTransparency(1)

        self.huxosaka_u_logo = OnscreenImage(image="./gui/hu_osaka_u_logo.png",
                                             pos=(0, 1, -1 + 176 / 1080 / 2),
                                             scale=(1450 / 1980, 1, 176 / 1980),
                                             parent=None)
        self.huxosaka_u_logo.setTransparency(1)
        left_margin = .02
        brmappath = Filename.fromOsSpecific(os.path.join(this_dir, "gui", "buttonrun_maps.egg"))
        maps = loader.loadModel(brmappath)
        self.run_btn = DirectButton(frameSize=(-1, 1, -.25, .25), geom=(maps.find('**/buttonrun_ready'),
                                                                        maps.find('**/buttonrun_click'),
                                                                        maps.find('**/buttonrun_rollover')),
                                    pos=(0.32 + left_margin, 0, -.5), scale=(.06, .12, .12),
                                    command=self.execplan,
                                    parent=self.pg234)

        brmappath = Filename.fromOsSpecific(os.path.join(this_dir, "gui", "buttondelete_maps.egg"))
        maps = loader.loadModel(brmappath)
        self.del_btn = DirectButton(frameSize=(-1, 1, -.25, .25), geom=(maps.find('**/buttondelete_ready'),
                                                                        maps.find('**/buttondelete_click'),
                                                                        maps.find('**/buttondelete_rollover')),
                                    pos=(0.47 + left_margin, 0, -.5), scale=(.06, .12, .12),
                                    command=self.deleteCapture,
                                    parent=self.pg234)

        brmappath = Filename.fromOsSpecific(os.path.join(this_dir, "gui", "buttonrecog_maps.egg"))
        maps = loader.loadModel(brmappath)
        self.recog_btn = DirectButton(frameSize=(-1, 1, -.25, .25), geom=(maps.find('**/buttonrecog_ready'),
                                                                          maps.find('**/buttonrecog_click'),
                                                                          maps.find('**/buttonrecog_rollover')),
                                      pos=(0.17 + left_margin, 0, -.5), scale=(.06, .12, .12),
                                      command=self.recognize,
                                      parent=self.pg234)

        self.nposes = 0
        self.textNPose = OnscreenText(text='#Poses: ' + str(self.nposes), pos=(1.45, -.9, 0), scale=0.03,
                                      fg=(1., 1., 1., 1),
                                      align=TextNode.ALeft, mayChange=1)
        self.textCaptured = OnscreenText(text='Ready to capture', pos=(1.45, -.95, 0), scale=0.03, fg=(1., 1., 1., 1),
                                         align=TextNode.ALeft, mayChange=1)

    def execplan(self):
        self.run_btn['state'] = DGG.DISABLED
        thread = threading.Thread(target=run, args=(self.scctrl,))
        thread.start()
        # thread.join()
        print("thread finished...exiting")

    def deleteCapture(self):
        self.run_btn['state'] = DGG.NORMAL
        self.recog_btn['state'] = DGG.NORMAL
        self.scctrl.remove_plot()
        self.scctrl.plot_rbt_sync()

    def recognize(self):
        self.recog_btn['state'] = DGG.DISABLED
        self.scctrl.vision()


# TODO Task Planning has some problems
# TODO Vision System needs to calibrate
# TODO Finger is to thin, needs to redeisgned
# TODO Motion Planning needs to improve speed
# Feedback System ?

class Execution(object):
    def __init__(self):
        rbthp = init_env_rbt()
        self.rbthp = rbthp
        # Change IK func
        rbthp.sim_rbt.ik = lambda hand_name, pos, rot, *args, **kwargs: rbthp.ik(pos, rot)
        rbthp.rbt_con.calibrate_gripper()

        self.component_name = "arm"
        rs_pipe = RealSenseD405()
        self.rs_pipe = rs_pipe

        rl = RackLocator(rbthp.rbt_con, rs_pipe,
                         origin_obs_pose=np.array([0.27, 0.07, OBS_HEIGHT]),
                         hand_to_eye_mat=TCP2EYE_MAT_DEVICE_1,
                         toggle_debug=False)
        self.rl = rl
        detector = TestTubeDetector(affine_mat_path=None)
        self.detector = detector

        # init the pp planner
        pp_planner = ppp.PickPlacePlanner(robot_s=rbthp.sim_rbt)
        self.pp_planner = pp_planner

        # Load Grasps
        grasp_cache_dict = {
            # purple ring
            1: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6_gp2", "purple_cap_tube_grasps.json")),
            # blue
            3: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6_gp2", "blue_cap_tube_grasps.json")),
            # white
            2: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6_gp2", "purple_cap_tube_grasps.json")),
        }  # restore grasps

        self.grasp_cache_dict = grasp_cache_dict

        data_queue = Queue()
        ret_queue = Queue()
        motion_runner = MotionRunner(rbthp=rbthp, rl=rl, data_queue=data_queue, ret_queue=ret_queue, detector=detector,
                                     grasp_cache_dict=grasp_cache_dict, pp_planner=pp_planner)
        motion_runner.start()
        self.task_solver = D3QNSolver(GOAL_PATTERN)
        self.data_queue = data_queue
        self.ret_queue = ret_queue
        self.motion_runner = motion_runner

        self.tube_rack = None
        self.rack_tf = None

        # plot
        self.plot_gm = []

        rbthp.sync()
        self.add_plot(rbthp.gen_mesh_model())
        self.motion_runner.update_add_plot_func(self.add_plot, self.remove_plot)

    def add_plot(self, cm):
        self.plot_gm.append(cm)
        cm.attach_to(base)

    def remove_plot(self):
        for _ in self.plot_gm:
            if _ is not None:
                _.remove()
        self.plot_gm = []

    def plot_rbt_sync(self):
        self.rbthp.sync()
        self.add_plot(self.rbthp.gen_mesh_model())

    def vision(self):
        self.remove_plot()

        rl = self.rl
        detector = self.detector
        rbthp = self.rbthp

        tube_rack, rack_tf = init_vision_system(rl, detector, rbthp.rbt_con, self.add_plot)

        psg.set_bg(GOAL_PATTERN)
        psg.set_fg(tube_rack.rack_status)

        self.tube_rack = tube_rack
        self.rack_tf = rack_tf

        rbthp.sync()
        self.add_plot(rbthp.gen_mesh_model())
        mdl = self.tube_rack.gen_mesh_model(gen_tube=True)
        self.add_plot(mdl)

        print(repr(tube_rack.rack_status))


def run(self):
    '''The main function'''
    """
    Initialization
    """

    '''
    Search Motion
    '''
    print("?")
    if self.rack_tf is None or self.tube_rack is None:
        self.vision()
    tube_rack = self.tube_rack
    rack_tf = self.rack_tf
    rl = self.rl
    detector = self.detector
    rbthp = self.rbthp
    data_queue = self.data_queue
    ret_queue = self.ret_queue
    motion_runner = self.motion_runner
    task_solver = self.task_solver
    grasp_cache_dict = self.grasp_cache_dict
    pp_planner = self.pp_planner
    component_name = self.component_name

    infeasible_info_dict = {}  # restore infeasible action during task planning
    infeasible_local_pattern = {}
    rack_state_history = []

    # init task planning solver
    local_geomtry = None
    for i in range(100):
        motion_batch = MotionBatch()
        if i > 0:
            # pcd_rgba, im = rl.obs_in_rack_center(rack_tf, toggle=False)
            tube_rack, rack_tf = next_vision_system(rack_tf, rl, detector, rbthp.rbt_con,
                                                    add_to_scene_func=self.add_plot,
                                                    remove_scene_func=self.remove_plot, toggle_debug=False)
            self.plot_rbt_sync()
            if local_geomtry is not None:
                print("LOCAL GEMOTRY EXIST", local_geomtry)
                tube_rack.update_local_state_data(*local_geomtry)
            # tube_rack.gen_mesh_model(gen_tube=True).attach_to(base)
            # base.run()
        tube_rack_exe = tube_rack.copy()

        motion_runner.update_tube_rack_exe(tube_rack_exe)
        psg.set_bg(GOAL_PATTERN)
        psg.set_fg(tube_rack_exe.rack_status)

        condition_set = np.ones(([*GOAL_PATTERN.shape[:2], 6]), dtype=int)

        while not isdone(node=tube_rack.rack_status, goalpattern=GOAL_PATTERN):
            task_sols = task_planning(tube_rack=tube_rack, solver=task_solver,
                                      condition_set=condition_set,
                                      infeasible_info_dict=infeasible_info_dict, )
            sol_length = len(task_sols)
            for sol_id, sol in enumerate(task_sols):
                print(f"------- Remaining unsolved path length is {sol_length - sol_id - 1} -------")
                print_with_border("Rack State matrix", width=38)
                print(text_pd(tube_rack.rack_status))
                is_replanning_flag = False
                moved_tube = sol.tube
                moved_tube_cm = moved_tube.gen_collision_model()
                # get grasp of moved tube
                grasp_info_list = grasp_cache_dict[moved_tube.type]
                # grasp_info_list = tube_rack.get_grasps_local_state(sol.init_slot_id, grasp_info_list)
                # collision model of the moved tube
                init_slot_id = sol.init_slot_id
                goal_slot_id = sol.goal_slot_id

                print_with_border(f"Moving: the test tube is {moved_tube.type}"
                                  f"            init coordinate is {init_slot_id}"
                                  f"              goal coordinate is {goal_slot_id}"
                                  f"              Number of grasps: {len(grasp_info_list)}", width=38)

                # test tubes' collision model
                obs_testubes_cmlist = sol.obs_tube_cm_collection.cm_list
                obj_init_homomat = sol.init_homomat.copy()
                obj_goal_homomat = sol.goal_homomat.copy()

                up_dis = GOAL_UP_DISTANCE[moved_tube.type]

                obj_goal_homomat[:3, 3] = obj_goal_homomat[:3, 3] + rack_tf[:3, 2] * up_dis
                homomat_list = [obj_init_homomat, obj_goal_homomat]
                grasp_direction = sol.grasp_direction
                insert_direction = sol.insert_direction
                print("insert d", insert_direction)
                insert_direction = np.array([0, 0, -1])
                # up distance
                tube_rack_obs_cm = tube_rack.gen_collision_model()
                grasp_id, init_tcp_homomat, init_tcp_approach_homomat, init_jaw_width, tgt_tcp_homomat_list, tgt_approach_jnt_val_list, tgt_tcp_sym_angles, tgt_jawwidth_list, is_pick_feasible = \
                    pp_planner.gen_pick_and_place_motion(hnd_name="arm",
                                                         tube=moved_tube,
                                                         tube_cm=moved_tube_cm,
                                                         grasp_info_list=grasp_info_list,
                                                         goal_homomat_list=homomat_list,
                                                         start_conf=rbthp.get_jnt_values(),
                                                         end_conf=None,
                                                         depart_direction_list=[rack_tf[:3, 2]] * len(
                                                             homomat_list),
                                                         approach_direction_list=[grasp_direction,
                                                                                  insert_direction],
                                                         depart_distance_list=DEPART_DISTANCE_LIST,
                                                         # DEPART_DISTANCE_LIST
                                                         approach_distance_list=APPROACH_DISTANCE_LIST[
                                                             moved_tube.type],
                                                         # APPROACH_DISTANCE_LIST
                                                         common_grasp_iter=None,
                                                         approach_jawwidth=None,
                                                         depart_jawwidth=None,
                                                         use_rrt=False,
                                                         obstacle_list=obs_testubes_cmlist,
                                                         grasp_obstacle_list=[tube_rack_obs_cm],
                                                         use_incremental=False,
                                                         # logger=exe_logger
                                                         )
                # Planning successfully
                if init_tcp_homomat is not None:
                    tube_rack.remove_slot(init_slot_id)
                    tube_rack.insert_tube(goal_slot_id, moved_tube)
                    print("---" * 17)
                    print("Planning Successfully!")
                    el = MotionElement(init_homo=init_tcp_homomat,
                                       init_approach_homo=init_tcp_approach_homomat,
                                       init_jawwidth=init_jaw_width,
                                       tgt_homo_list=tgt_tcp_homomat_list,
                                       tgt_approach_homo_list=tgt_approach_jnt_val_list,
                                       tgt_jawwidth_list=tgt_jawwidth_list,
                                       tgt_tcp_sym_angles=tgt_tcp_sym_angles,
                                       grasp_id=grasp_id,
                                       grasp_info_list=grasp_info_list,
                                       move_tube=moved_tube,
                                       rack_tf=rack_tf,
                                       init_slot_id=init_slot_id,
                                       goal_slot_id=goal_slot_id,
                                       hnd_name=component_name,
                                       obs_list=obs_testubes_cmlist, )
                    motion_batch.append(el)
                    data_queue.put(el)
                # Planning Failed
                else:
                    print("---" * 17)
                    print("Planning Failed!! Restart Search a New Path")
                    # planning failed
                    current_state = tube_rack.rack_status.copy()
                    next_state = current_state.copy()
                    next_state[tuple(init_slot_id)], next_state[tuple(goal_slot_id)] = next_state[
                        tuple(goal_slot_id)], \
                        next_state[
                            tuple(init_slot_id)]
                    # weight_matrix = bad_status_weight_matrix.get(str(current_state),
                    #                                              np.zeros_like(tube_rack.rack_status)).copy()
                    # weight_matrix[tuple(init_slot_id)] = 1
                    # weight_matrix[tuple(goal_slot_id)] = 1
                    # bad_status_weight_matrix[str(current_state)] = weight_matrix
                    # print failed reason
                    # print(common_grasp_iter_info)
                    # add infeasible state to dict
                    # pdb.set_trace()
                    task_solver.cal_constraints(current_state,
                                                init_slot_id,
                                                goal_slot_id,
                                                is_pick_feasible,
                                                True,
                                                condition_set,
                                                infeasible_action_dict=infeasible_info_dict)
                    print("Current SLot maps are", np.sum(condition_set, axis=2))
                    is_replanning_flag = True
                # planning failed or execution failed
                if is_replanning_flag:
                    break

        data_queue.put("end")
        need_replanning = ret_queue.get()
        with ret_queue.mutex:
            ret_queue.queue.clear()
        if not need_replanning:
            break
    # show_animation(rbthp.sim_rbt, motion_batch, base)

    # for _ in


if __name__ == "__main__":
    # main()
    from huri.core.common_import import wd

    base = wd.World(cam_pos=(1, -0.6, 0.7), lookat_pos=[.2, 0, .1], w=1920,
                    h=1080, )
    from huri.core.base_boost import zoombase, boost_base

    base = boost_base(base)

    func = Execution()
    HLabGUI(func)
    psg = PatternSelectGui(base, rack_size=(5, 10))
    base.run()

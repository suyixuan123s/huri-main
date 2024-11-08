import pickle
import queue
import time

from utils import init_env_rbt, np, MotionElement, MotionBatch
from huri.definitions.rack_def import TubeRack, Rack_Hard_Proto, TubeType
from tubesolseq import ArrangeTubeSolSeq
from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
from huri.core.print_tool import text_pd, print_with_border
import huri.core.file_sys as fs
import huri.components.planning.symmetric_pick_and_place_planner_simple as ppp
from huri.learning.env.arrangement_planning_rack.utils import isdone
from animation import show_animation
from dual_d405_manager import RealSenseD405Dual
import basis.robot_math as rm
from vision_sys import RackLocatorDual, detect_tube
from huri.math.math import combination
from huri.components.vision.tube_detector import TestTubeDetector, extract
import modeling.geometric_model as gm
import threading
from queue import Queue

from constants import (HEIGHT_RANGE,
                       GOAL_PATTERN,
                       GOAL_UP_DISTANCE,
                       APPROACH_DISTANCE_LIST,
                       DEPART_DISTANCE_LIST,
                       SPEED,
                       ACC,
                       TOGGLE_RACK_LOCATOR,
                       TOGGLE_YOLO)
import constants


def task_planning(tube_rack: TubeRack, solver: DQNSolver, infeasible_info_dict: dict,
                  infeasible_local_pattern: dict) -> ArrangeTubeSolSeq:
    task_sols = ArrangeTubeSolSeq(rack=tube_rack,
                                  solution=solver.solve(tube_rack.rack_status,
                                                        GOAL_PATTERN,
                                                        infeasible_info_dict,
                                                        infeasible_local_pattern,
                                                        toggle_result=False),
                                  tf=tube_rack.get_homomat())
    return task_sols


def init_vision_system(rl, detector, rbtx, toggle_debug=False):
    # x_range = np.array([.24])
    # y_range = np.array([0])
    # z_range = np.array([0.15])
    # r = combination([x_range, y_range, z_range]).tolist()
    # rl.add_obs_pos(obs_pos=r)
    rbtx.homeconf()
    c_pcd, pcds = rl.scan(toggle=False)
    rack, rack_tf = rl.locate_rack(c_pcd, Rack_Hard_Proto, height_range=HEIGHT_RANGE,
                                   toggle_debug=toggle_debug)
    # pcd_rgba, im_center = rl.obs_in_rack_center(rack_tf, toggle=TOGGLE_RACK_LOCATOR)

    obs_p_1 = rack.get_slot_homomat((2, 1))
    new_rbt_obs_pos_1 = rl.get_obs_xy_point(obs_p_1[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_1[:3, 3],
                new_rbt_obs_pos_1[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_1, im_1 = rl.get_pcd_im_1()
    pcd_2, im_2 = rl.get_pcd_im_2()

    obs_p_2 = rack.get_slot_homomat((2, 8))
    new_rbt_obs_pos_2 = rl.get_obs_xy_point(obs_p_2[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_2[:3, 3],
                new_rbt_obs_pos_2[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_3, im_3 = rl.get_pcd_im_1()
    pcd_4, im_4 = rl.get_pcd_im_2()

    rack_instance = Rack_Hard_Proto
    for pcd, im in [(pcd_1, im_1), (pcd_2, im_2), (pcd_3, im_1), (pcd_4, im_2)]:
        rack_instance, rack_tf = detect_tube(pcd[:, :3], im,
                                             rack_proto=rack_instance,
                                             rack_tf=rack_tf,
                                             detector=detector,
                                             toggle_yolo=False,
                                             toggle_detect_tube_pos=True)
        # gm.gen_rgb_pointcloud(np.vstack((pcd_1, pcd_2, pcd_3, pcd_4)))
    return rack_instance, rack_tf


def next_vision_system(rack_tf, rl, detector, rbtx, toggle_debug=False):
    pcd_rgba, im = rl.obs_in_rack_center(rack_tf, toggle=False)
    rack, rack_tf = rl.locate_rack(pcd_rgba[:, :3], Rack_Hard_Proto, height_range=HEIGHT_RANGE,
                                   toggle_debug=toggle_debug)
    pcd_rgba, im_center = rl.obs_in_rack_center(rack_tf, toggle=TOGGLE_RACK_LOCATOR)

    obs_p_1 = rack.get_slot_homomat((2, 1))
    new_rbt_obs_pos_1 = rl.get_obs_xy_point(obs_p_1[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_1[:3, 3],
                new_rbt_obs_pos_1[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_1, im_1 = rl.get_pcd_im()

    obs_p_2 = rack.get_slot_homomat((2, 8))
    new_rbt_obs_pos_2 = rl.get_obs_xy_point(obs_p_2[:2, 3], )
    rbtx.move_p(new_rbt_obs_pos_2[:3, 3],
                new_rbt_obs_pos_2[:3, :3], speed=SPEED, mvacc=ACC)
    pcd_2, im_2 = rl.get_pcd_im()

    rack_instance = Rack_Hard_Proto
    for pcd, im in [(pcd_rgba, im_center), (pcd_1, im_1), (pcd_2, im_2)]:
        rack_instance, rack_tf = detect_tube(pcd[:, :3], im,
                                             rack_proto=rack_instance,
                                             rack_tf=rack_tf,
                                             detector=detector,
                                             toggle_yolo=False,
                                             toggle_detect_tube_pos=True)
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
        threading.Thread.__init__(self)
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
                break
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

    def pick_motion(self, el):
        rl = self.rl
        detector = self.detector
        tube_rack_exe = self.tube_rack_exe
        rbthp = self.rbthp
        grasp_cache_dict = self.grasp_cache_dict
        pp_planner = self.pp_planner
        for i in range(3):
            need_replanning = True
            self.open_gripper()
            self.rbt_x.move_p(el.pick_approach[:3, 3],
                              el.pick_approach[:3, :3],
                              speed=350,
                              mvacc=ACC)

            _rack, _rack_tf, pcd_w, im = self.rl.track(Rack_Hard_Proto,
                                                       rack_init_pose=el.rack_tf,
                                                       height_range=HEIGHT_RANGE,
                                                       rack_state=tube_rack_exe.rack_status,
                                                       toggle_pcd=False)
            # fs.dump_pickle([tube_rack_exe.rack_status, el.rack_tf], "rack_states", reminder=False)
            # exit(0)
            _rack, __rack_tf = detect_tube(pcd_w,
                                           im,
                                           rack_proto=Rack_Hard_Proto,
                                           rack_tf=_rack_tf,
                                           detector=detector,
                                           toggle_yolo=TOGGLE_YOLO,
                                           toggle_detect_tube_pos=False)
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
                _rack, __rack_tf = detect_tube(pcd_w,
                                               im,
                                               rack_proto=Rack_Hard_Proto,
                                               rack_tf=_rack_tf,
                                               detector=detector,
                                               toggle_yolo=TOGGLE_YOLO,
                                               toggle_detect_tube_pos=False)
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

            new_approach_pos = init_jaw_center_pos + el.pick_approach[:3, 3] - el.pick[:3, 3]
            new_depart_pos = init_jaw_center_pos + _rack_tf[:3, 2] * DEPART_DISTANCE_LIST[0]

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
        return need_replanning, el, _rack_tf

    def place_motion(self, el, rack_tf):
        rbt_x = self.rbt_x
        rl = self.rl
        grasp = el.grasp_info_list[el.grasp_id]
        tube_rack_exe = self.tube_rack_exe

        rbt_x.move_p(el.place_approach[:3, 3], el.place_approach[:3, :3], speed=300, mvacc=ACC)

        _rack, _rack_tf, pcd_w, im = rl.track(Rack_Hard_Proto, rack_init_pose=rack_tf,
                                              rack_state=tube_rack_exe.rack_status,
                                              height_range=HEIGHT_RANGE)

        tube_homomat = _rack.get_slot_homomat(el.goal_slot_id)

        tube_homomat = el.tube.get_symmetry_homomat_at_angle(el.tgt_tcp_sym_angles[0], tube_homomat)

        goal_jaw_center_pos = tube_homomat[:3, 3] + tube_homomat[:3, :3].dot(grasp[1]) + _rack_tf[:3,
                                                                                         2] * GOAL_UP_DISTANCE[
                                  el.tube.type]
        goal_jaw_center_rotmat = tube_homomat[:3, :3].dot(grasp[2])
        tgt_new_approach_pos = goal_jaw_center_pos + el.place_approach[:3, 3] - el.place[:3, 3]
        rbt_x.move_p(tgt_new_approach_pos,
                     goal_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
        rbt_x.move_p(goal_jaw_center_pos, goal_jaw_center_rotmat, speed=SPEED, mvacc=ACC)
        self.open_gripper()
        rbt_x.move_p(tgt_new_approach_pos, goal_jaw_center_rotmat, speed=SPEED, mvacc=ACC)

        # update tube_rack exe
        tube_rack_exe.remove_slot(el.init_slot_id)
        tube_rack_exe.insert_tube(el.goal_slot_id, el.tube)
        return False, el

    def open_gripper(self):
        self.rbt_x.open_gripper(speed=300)

    def close_gripper(self):
        self.rbt_x.close_gripper(speed=80)

    def update_tube_rack_exe(self, tube_rack_exe):
        self.tube_rack_exe = tube_rack_exe


# TODO Task Planning has some problems
# TODO Vision System needs to calibrate
# TODO Finger is to thin, needs to redeisgned
# TODO Motion Planning needs to improve speed
# Feedback System ?

def main():
    '''The main function'''
    """
    Initialization
    """
    rbthp = init_env_rbt()
    # Change IK func
    rbthp.sim_rbt.ik = lambda hand_name, pos, rot, *args, **kwargs: rbthp.ik(pos, rot)
    rbthp.rbt_con.calibrate_gripper()
    component_name = "arm"
    rs_pipe = RealSenseD405Dual()

    rl = RackLocatorDual(rbthp.rbt_con, rs_pipe,
                         origin_obs_pose=np.array([0.27, 0.07, constants.OBS_HEIGHT]),
                         hand_to_eye_mat_1=constants.TCP2EYE_MAT_DEVICE_1,
                         hand_to_eye_mat_2=constants.TCP2EYE_MAT_DEVICE_2,
                         toggle_debug=False)
    detector = TestTubeDetector(affine_mat_path=None)
    tube_rack, rack_tf = init_vision_system(rl, detector, rbthp.rbt_con)

    print(repr(tube_rack.rack_status))
    # init the pp planner
    pp_planner = ppp.PickPlacePlanner(robot_s=rbthp.sim_rbt)

    # Load Grasps
    grasp_cache_dict = {
        # purple ring
        1: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "purple_ring_cap_tube_grasps.json")),
        # blue
        2: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "blue_cap_tube_grasps.json")),
        # white
        3: fs.load_json(path=fs.workdir_data.joinpath("grasps", "lite6", "white_cap_tube_grasps.json")),
    }  # restore grasps

    infeasible_info_dict = {}  # restore infeasible action during task planning
    infeasible_local_pattern = {}
    rack_state_history = []

    task_solver = DQNSolver()
    data_queue = Queue()
    ret_queue = Queue()
    motion_runner = MotionRunner(rbthp=rbthp, rl=rl, data_queue=data_queue, ret_queue=ret_queue, detector=detector,
                                 grasp_cache_dict=grasp_cache_dict, pp_planner=pp_planner)
    motion_runner.start()
    '''
    Search Motion
    '''

    # init task planning solver
    local_geomtry = None
    for i in range(100):
        motion_batch = MotionBatch()
        if i > 0:
            # pcd_rgba, im = rl.obs_in_rack_center(rack_tf, toggle=False)
            tube_rack, rack_tf = next_vision_system(rack_tf, rl, detector, rbthp.rbt_con, toggle_debug=False)
            if local_geomtry is not None:
                print("LOCAL GEMOTRY EXIST", local_geomtry)
                tube_rack.update_local_state_data(*local_geomtry)
            # tube_rack.gen_mesh_model(gen_tube=True).attach_to(base)
            # base.run()
        tube_rack_exe = tube_rack.copy()
        motion_runner.update_tube_rack_exe(tube_rack_exe)
        while not isdone(node=tube_rack.rack_status, goalpattern=GOAL_PATTERN):
            local_geomtry = None
            task_sols = task_planning(tube_rack=tube_rack, solver=task_solver,
                                      infeasible_info_dict=infeasible_info_dict,
                                      infeasible_local_pattern=infeasible_local_pattern)
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
                grasp_info_list = tube_rack.get_grasps_local_state(sol.init_slot_id, grasp_info_list)
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
                # up distance
                tube_rack_obs_cm = tube_rack.gen_collision_model()
                grasp_id, init_tcp_homomat, init_tcp_approach_homomat, init_jaw_width, tgt_tcp_homomat_list, tgt_approach_jnt_val_list, tgt_tcp_sym_angles, tgt_jawwidth_list, is_init_feasible = \
                    pp_planner.gen_pick_and_place_motion(hnd_name="arm",
                                                         tube=moved_tube,
                                                         tube_cm=moved_tube_cm,
                                                         grasp_info_list=grasp_info_list,
                                                         goal_homomat_list=homomat_list,
                                                         start_conf=rbthp.get_jnt_values(),
                                                         end_conf=None,
                                                         depart_direction_list=[rack_tf[:3, 2]] * len(
                                                             homomat_list),
                                                         approach_direction_list=[grasp_direction, insert_direction],
                                                         depart_distance_list=DEPART_DISTANCE_LIST,
                                                         # DEPART_DISTANCE_LIST
                                                         approach_distance_list=APPROACH_DISTANCE_LIST[moved_tube.type],
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
                    next_state[tuple(init_slot_id)], next_state[tuple(goal_slot_id)] = next_state[tuple(goal_slot_id)], \
                                                                                       next_state[tuple(init_slot_id)]
                    # weight_matrix = bad_status_weight_matrix.get(str(current_state),
                    #                                              np.zeros_like(tube_rack.rack_status)).copy()
                    # weight_matrix[tuple(init_slot_id)] = 1
                    # weight_matrix[tuple(goal_slot_id)] = 1
                    # bad_status_weight_matrix[str(current_state)] = weight_matrix
                    # print failed reason
                    if not is_init_feasible:
                        print("Infeasible Reason: No grasp at init position")
                    # print(common_grasp_iter_info)
                    # add infeasible state to dict
                    infeasible_actions = infeasible_info_dict.get(str(current_state), [])
                    is_remove_all_init_slot_id_actions = not is_init_feasible
                    infeasible_actions.extend(task_solver.action_between_states(current_state,
                                                                                next_state,
                                                                                remove_action_group=is_remove_all_init_slot_id_actions))
                    print(rack_state_history)
                    for prev_state in rack_state_history:
                        diff = current_state - prev_state
                        if len(diff[diff != 0]) != 2:  # only one step different
                            continue
                        infeasible_actions.extend(task_solver.action_between_states(current_state, prev_state))

                    infeasible_info_dict[str(current_state)] = list(set(infeasible_actions))

                    # infeaible local pattern
                    if not is_init_feasible:
                        # TODO bug still there
                        infeasible_local = infeasible_local_pattern.get(tuple(init_slot_id), {})
                        local_info = str(np.pad(current_state, 1)[init_slot_id[0]:init_slot_id[0] + 3,
                                         init_slot_id[1]:init_slot_id[1] + 3])
                        local_infeasible_actions = infeasible_local.get(local_info, [])
                        local_infeasible_actions.extend(task_solver.action_between_states(current_state,
                                                                                          next_state,
                                                                                          remove_action_group=is_remove_all_init_slot_id_actions))
                        infeasible_local[local_info] = local_infeasible_actions
                        infeasible_local_pattern[tuple(init_slot_id)] = infeasible_local

                    # print("Weight matrix of the state is ...\n", weight_matrix)
                    print("Infeasible action of the state is ...", infeasible_actions)
                    print(f"Remove all the action that pick {init_slot_id} ...", is_remove_all_init_slot_id_actions)
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
    base.run()
    # show_animation(rbthp.sim_rbt, motion_batch, base)

    # for _ in


if __name__ == "__main__":
    main()

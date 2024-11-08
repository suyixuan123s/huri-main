"""
作者: Hao Chen (chen960216@gmail.com)
创建日期: 20231130大阪
"""

__VERSION__ = "0.0.2_debug"

import time

from huri.core.common_import import wd, ym, fs, np
from huri.core.constants import SENSOR_INFO
from huri.components.exe.utils import gen_camera_obs, capture_vision_info, init_real_rbt, change_led_power, \
    vision_read_data
from huri.definitions.rack_def import TubeRack
from huri.core.print_tool import text_pd, print_with_border
from huri.definitions.utils_structure import MotionElement, MotionBatch, MotionBatchPPP
# 运动规划
from huri.components.planning.common_grasp_seq import CommonGraspSolver, CommonGraspSolverDebug
import huri.components.planning.symmetric_pick_and_place_planner_c as ppp
# 视觉系统
from huri.definitions.rack_def import ArrangeTubeSolSeq, Rack_Hard_Proto
from huri.components.vision.tube_detector import TestTubeDetector
# 任务规划
from huri.learning.env.arrangement_planning_rack.utils import isdone
from huri.components.task_planning.tube_puzzle_learning_solver import D3QNSolver
# 机器人执行
import huri.components.yumi_control.yumi_con as yc
# 错误反馈
from huri.components.exe.executer import (feedback_insertion_error,
                                          INSERT_FAILURE_ERR,
                                          GRASP_COLLISION_ERR,
                                          INSERT_FAILURE_TUBE_LOSING_ERR,
                                          GRASP_NO_TUBE_ERR,
                                          UNKNOWN_ERR)
from huri.components.exe.constants import APPROACH_DISTANCE_LIST, DEPART_DISTANCE_LIST, TABLE_HEIGHT, RACK_HEIGHT
from huri.components.exe.utils import create_directory
from huri.learning.method.AlphaZero.log import CsvWriter, write_to_csv
from huri.learning.method.AlphaZero.utils import get_time_stamp, delete_all_files_in_directory, is_folder_empty
import uuid
from huri.components.exe.version.grasp_filter import grasp_filter, get_feasible_ranges_rad
from huri.components.planning.symmetric_pick_and_place_palnner_c_fund import PickPlacePlanner_CR
import modeling.collision_model as cm

# 目标模式（GOAL_PATTERN）定义
GOAL_PATTERN = np.array([[1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ],
                         [1, 0, 2, 0, 3, 0, 4, 0, 5, 0, ]])

# 全局变量
ITERATION_NUM = [0]
MOTION_PLANNING_SOL_LOCAL_COUNTER = [0]
MOTION_PLANNING_LOCAL_COUNTER = [0]
VISION_SYSTEM_LOCAL_COUNTER = [0]
INSERTION_FEEDBACK_COUNTER = [0]
WORK_DIRECTORY = create_directory('./run')
if not is_folder_empty(WORK_DIRECTORY):
    delete_all_files_in_directory(WORK_DIRECTORY)
WORK_DIRECTORY_VISION = create_directory('./run/vision')
WORK_DIRECTORY_TP = create_directory('./run/taskplanning')
WORK_DIRECTORY_MP = create_directory('./run/motionplanning')

LOGGER = CsvWriter(str(WORK_DIRECTORY.joinpath("running_log.csv")))

# 任务规划
def task_planning(tube_rack: TubeRack, solver: D3QNSolver, condition_set, infeasible_action_dict) -> ArrangeTubeSolSeq:
    ##### 日志记录
    st = time.time()
    save_path = WORK_DIRECTORY_TP.joinpath(f'TASK_PLANNING_DATA_ITER_{ITERATION_NUM[0]}.pkl')
    ###############
    raw_solution, solution = solver.solve(tube_rack.rack_status,
                                          condition_set=condition_set,
                                          infeasible_action_dict=infeasible_action_dict,
                                          max_iter=800,
                                          toggle_show=False)
    task_sols = ArrangeTubeSolSeq(rack=tube_rack,
                                  solution=solution,
                                  tf=tube_rack.get_homomat())
    return task_sols

# 视觉系统
def vision_system(detector: TestTubeDetector,
                  toggle_yolo=False,
                  toggle_save=False,
                  debug_filepath=None,
                  debug_program=DEBUG_PROGRAM) -> (TubeRack, np.ndarray):
    toggle_debug = True if debug_filepath is not None else False
    ##### 日志记录
    st = time.time()
    save_path = WORK_DIRECTORY_VISION.joinpath(
        f'VISION_DATA_ITER_{ITERATION_NUM[0]}_{VISION_SYSTEM_LOCAL_COUNTER[0]}.pkl')
    #####
    # 通过视觉传感器捕捉数据
    if debug_program:
        filename = fs.workdir / "data" / "vision_exp" / "20231130-163116.pkl"
        pcd, img, depth_img, _, _ = vision_read_data(filename)
    else:
        pcd, img, depth_img = capture_vision_info(ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG,
                                                  debug_filename=debug_filepath,
                                                  save_path=save_path,
                                                  toggle_save=False,
                                                  toggle_debug=toggle_debug, )  # TODO 调试代码修改
    detected_test_tubes, tube_rack, rack_tf, yolo_img = detector.analyze_scene(rack_proto=Rack_Hard_Proto,
                                                                               pcd=pcd,
                                                                               texture_img=img[:, :, 0],
                                                                               std_out=None,
                                                                               toggle_detect_tube_pos=True,
                                                                               toggle_yolo=True,
                                                                               save_detect=toggle_save)
    fs.dump_pickle([uuid.uuid4().hex, pcd, img], r"vision_data.pkl",
                   reminder=False)

    return tube_rack, rack_tf

# 主执行流程
def main(open_jaw_width=.034,
         depth_sensor_debug_path=None,
         debug=False,
         retry_num=3,  # 抓取失败时的重试次数
         goal_place_offset_dis=.021,
         is_vision_feedback=True,
         toggle_save=False,
         exe_speed=200,
         rrt_ext_dis=.008,
         VISION_FEEDBACK_INTERVAL=3, ):
    # 初始化仿真机器人
    yumi_robot = ym.Yumi(enable_cc=True)
    # 初始化抓取与放置规划器
    num_workers = 6
    if not DEBUG_PROGRAM:
        common_grasp_solver = CommonGraspSolver(num_workers=num_workers)
    else:
        common_grasp_solver = CommonGraspSolverDebug()
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot, common_grasp_solver=common_grasp_solver)

    # 仿真环境
    PickPlacePlanner_CR.RRT_EXT_DIS = rrt_ext_dis
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    collision_box_for_rrt = cm.gen_box(extent=[5, 5, .12], homomat=np.eye(4))
    collision_box_for_rrt.set_pos(np.array([0, 0, .06]))
    collision_box_for_rrt.attach_to(base)

    change_led_power(ip_adr=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG, led_power=2000)
    component_name = "rgt_hnd"
    yumi_robot_arm = yumi_robot.rgt_arm if "rgt" in component_name else yumi_robot.lft_arm

    # 初始化真实机器人
    yumi_con = init_real_rbt(gripper_len=open_jaw_width, toggle_debug=debug, use_motion_exe=True)
    if not debug:
        yc.to_homeconf(yumi_s=yumi_robot, yumi_x=yumi_con, component_name="both")
        yumi_robot.fk('rgt_arm', yumi_con.get_jnt_values('rgt_arm'))
        yumi_robot.fk('lft_arm', yumi_con.get_jnt_values('lft_arm'))

    # 设置Phoxi相机的碰撞模型
    camera_obs = gen_camera_obs()
    camera_obs.attach_to(base)

    # 视觉系统
    detector = TestTubeDetector(affine_mat_path=SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH,
                                rack_height_lower=TABLE_HEIGHT + RACK_HEIGHT - .005,
                                rack_height_upper=TABLE_HEIGHT + RACK_HEIGHT + .005, )
    tube_rack, rack_tf = vision_system(detector, debug_filepath=depth_sensor_debug_path,
                                       toggle_save=toggle_save)

    # 初始化任务规划求解器
    task_solver = D3QNSolver(goal_pattern=GOAL_PATTERN)

    ## 运动规划初始化
    motion_batch = MotionBatch()
    grasp_cache_dict = {}  # 保存抓取
    infeasible_info_dict = {}  # 保存不可行的操作
    infeasible_local_pattern = {}
    rack_state_history = []
    bad_status_weight_matrix = {}

    condition_set = np.ones(([*GOAL_PATTERN.shape[:2], 6]), dtype=int)
    infeasible_action_dict = {}

    animation_cnt = 0
    need_vision_feedback = False
    need_grasp_vision_feedback = False
    need_vision_counter = 0

    while not isdone(node=tube_rack.rack_status, goalpattern=GOAL_PATTERN):
        print(len(np.where(tube_rack.rack_status == 1)[0]))
        print(len(np.where(tube_rack.rack_status == 2)[0]))
        print(len(np.where(tube_rack.rack_status == 3)[0]))

        if not task_solver.is_state_valid(tube_rack.rack_status):
            raise Exception("状态包含不可解的模式！")

        # 重新计算操作
        task_sols = task_planning(tube_rack=tube_rack,
                                  solver=task_solver,
                                  condition_set=condition_set,
                                  infeasible_action_dict=infeasible_action_dict)

        sol_length = len(task_sols)
        for sol_id, sol in enumerate(task_sols):
            need_vision_counter += 1
            if need_vision_counter % VISION_FEEDBACK_INTERVAL == 0:
                need_vision_feedback = True

            print(f"------- 剩余未解决的路径长度为 {sol_length - sol_id - 1} -------")
            print_with_border("试管架状态矩阵", width=38)
            print(text_pd(tube_rack.rack_status))
            # 生成并执行抓取与放置动作
            common_grasp_iter_info = None
            is_replanning_flag = False
            rack_state_history.append(tube_rack.rack_status.copy())
            moved_tube = sol.tube
            moved_tube_cm = moved_tube.gen_collision_model()
            init_slot_id = sol.init_slot_id
            goal_slot_id = sol.goal_slot_id
            print_with_border(f"正在移动：试管类型 {moved_tube.type}，起始坐标 {init_slot_id}，目标坐标 {goal_slot_id}，抓取数: {len(grasp_cache_dict.get(moved_tube.type, []))}", width=38)

            ### 省略部分中间代码 ###

            if not is_retry:
                break

        # 规划或执行失败时，重新规划
        if is_replanning_flag:
            break

    print_with_border("所有试管已排列完成", width=70)
    common_grasp_solver.stop()
    if debug:
        pass
    else:
        yumi_con.stop()


if __name__ == '__main__':
    main(open_jaw_width=.031,
         debug=False,
         depth_sensor_debug_path=None,
         retry_num=5,
         goal_place_offset_dis=.025,
         is_vision_feedback=True,
         toggle_save=True,
         exe_speed=600,
         rrt_ext_dis=.025,
         VISION_FEEDBACK_INTERVAL=2)

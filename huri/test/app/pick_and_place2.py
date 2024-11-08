import copy

from huri.components.pipeline.data_pipeline import RenderController, MotionController
from huri.definitions.tube_def import TubeType, TestTube
from huri.core.file_sys import load_json
from huri.core.common_import import *
# import huri.component.planning.pick_place_planner as ppp
import manipulation.pick_place_planner as ppp
from typing import List
from huri.definitions.utils_structure import MotionElement, MotionBatch


def get_objcm_grasps(tube: TestTube):
    objcm = cm.CollisionModel(initor=tube.model_path)
    grasp_info_list = load_json(path=tube.grasps_path)
    return objcm, grasp_info_list


def test(tuberacksolution,
         std_out: RenderController = None):
    tube_list, tube_init_pos_list, tube_goal_pos_list = [], [], []
    for result in tuberacksolution:
        tube, tube_init_pos, tube_goal_pos = result
        tube_list.append(tube)
        tube_init_pos_list.append(tube_init_pos)
        tube_goal_pos_list.append(tube_goal_pos)

    yumi_robot = ym.Yumi(enable_cc=True)
    std_out.attach(node=yumi_robot.gen_meshmodel(), name="yumi")
    hand_name = 'rgt_hnd'
    start_conf = yumi_robot.get_jnt_values(hand_name)
    pp_planner = ppp.PickPlacePlanner(robot_s=yumi_robot)
    motions = MotionBatch()

    obstacle_list_pos = copy.deepcopy(tube_init_pos_list)
    for tube_idx in range(len(tube_list)):
        tube = tube_list[tube_idx]
        tube_init_pos = tube_init_pos_list[tube_idx]
        tube_goal_pos = tube_goal_pos_list[tube_idx]
        goal_homomat_list = [tube_init_pos, tube_goal_pos]
        objcm_tmp, grasp_info_list = get_objcm_grasps(tube=tube)
        obstacle_list = []
        for obs_tube_idx, obs_tube in enumerate(tube_list):
            if tube_idx == obs_tube_idx:
                continue
            obs_tube_cm = cm.CollisionModel(initor=obs_tube.model_path, cdprimit_type="cylinder")
            print(obstacle_list_pos[obs_tube_idx])
            obs_tube_cm.set_homomat(obstacle_list_pos[obs_tube_idx])
            # obs_tube_cm.attach_to(base)
            # obs_tube_cm.set_rgba([1,0,0,1])
            obstacle_list.append(obs_tube_cm)
        print(f"     :: number of grasps:{len(grasp_info_list)}")
        objcm_tmp.set_homomat(tube_init_pos)
        objcm_tmp.set_rgba(tube.color)
        objcm_tmp.show_localframe()
        if std_out is not None:
            std_out.attach(node=objcm_tmp, name="obj at start")
            objcm_tmp_goal = objcm_tmp.copy()
            objcm_tmp_goal.set_homomat(tube_goal_pos)
            objcm_tmp_goal.show_localframe()
            std_out.attach(node=objcm_tmp_goal, name="obj at goal")
            print(goal_homomat_list)
            # base.startTk()
            # base.tkRoot.withdraw()
            # base.run()

        conf_list_tmp, jawwidth_list_tmp, objpose_list_tmp = \
            pp_planner.gen_pick_and_place_motion(hnd_name=hand_name,
                                                 objcm=objcm_tmp,
                                                 grasp_info_list=grasp_info_list,
                                                 goal_homomat_list=goal_homomat_list,
                                                 start_conf=None,
                                                 end_conf=start_conf if tube_idx == len(tube_list)-1 else None,
                                                 depart_direction_list=[np.array([0, 0, 1])] * len(goal_homomat_list),
                                                 approach_direction_list=[np.array([0, 0, -1])] * len(
                                                     goal_homomat_list),
                                                 depart_distance_list=[.2] * len(goal_homomat_list),
                                                 approach_distance_list=[.2] * len(goal_homomat_list),
                                                 approach_jawwidth=None,
                                                 depart_jawwidth=None,
                                                 ad_granularity=.01,
                                                 use_rrt=False,
                                                 obstacle_list=obstacle_list,
                                                 use_incremental=False, )
        if conf_list_tmp is not None:
            # reset the pose of the obstacle
            obstacle_list_pos[tube_idx] = tube_goal_pos
            # add motions into motion batch
            for i in range(len(conf_list_tmp)):
                motions.append(MotionElement(obj_cm=objcm_tmp.copy(),
                                             objpose_list=objpose_list_tmp[i],
                                             conf_list=conf_list_tmp[i],
                                             jawwidth_list=jawwidth_list_tmp[i],
                                             hnd_name=hand_name))


    motion_batch = iter(motions)

    def update(robot_s: ym.Yumi,
               motion_batch: MotionBatch,
               task):
        if base.inputmgr.keymap["space"]:
            base.inputmgr.keymap["space"] = False
            try:
                motion_element = next(motion_batch)
            except StopIteration:
                return task.again
        else:
            motion_element = motion_batch.current
        try:
            objcm, obj_pose, pose, jawwdith, hand_name = next(motion_element)
        except StopIteration:
            return task.again
        robot_s.fk(hand_name, pose)
        robot_s.jaw_to(hand_name, jawwdith)
        robot_meshmodel = robot_s.gen_meshmodel()
        robot_meshmodel.attach_to(base)
        objb_copy = objcm.copy()
        objb_copy.set_homomat(obj_pose)
        objb_copy.attach_to(base)
        motion_element.reset_robot_gm(robot_meshmodel)
        motion_element.reset_obj_gm(objb_copy)
        return task.again

    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[yumi_robot,
                                     motion_batch
                                     ],
                          appendTask=True)
    return motions


if __name__ == "__main__":
    from huri.core.file_sys import workdir, load_pickle
    from huri.components.vision.tube_detector import TestTubeDetector
    from huri.components.yumi_control.yumi_con import YumiController
    from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
    from time import strftime

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    std_out = RenderController(root=base.tkRoot, base=base)
    detector = TestTubeDetector(workdir / "data/calibration/affine_mat_20210730-004129.json")
    # load test data
    filename = workdir / "data" / "vision_exp" / "20210726-164310.pkl"
    # _, _, img, _, pcd = load_pickle(filename)
    # pcd, img = load_pickle(filename)
    pcd, img = vision_pipeline(SensorMarkerHandler(),
                               workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl")
    detected_test_tube, tuberacksolution = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0], std_out=std_out)
    # plot tubes
    rack_cm = cm.CollisionModel(initor=tuberacksolution.rack_status.model_path)
    rack_cm.set_homomat(tuberacksolution.tf)
    rack_cm.set_rgba(tuberacksolution.rack_status.color)
    std_out.attach(node=rack_cm, name="tube rack")
    # find tube_list, tube init pos list, tube goal pos list
    # get motions
    motionbatch = test(tuberacksolution=tuberacksolution,
                       std_out=std_out)

    # execution
    try:
        yumi_con = YumiController()
        mc = MotionController(root=base.tkRoot, yumi_con=yumi_con)
        mc.add_batch(motionbatch)
    except:
        pass
    base.startTk()
    base.tkRoot.withdraw()
    base.run()

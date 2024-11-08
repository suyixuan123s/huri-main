"""
This is an example to
1. generate the pick-and-place motion to move a test tube from a rack to another rack
"""
from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
from huri.core.common_import import *
from huri.core.print_tool import text_pd
from itertools import product
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot and its mesh model
ym_rbt = ym.Yumi(enable_cc=True)
component_name = "rgt_arm"

# initialize the module to generate the linear motion
inik_svlr = inik.IncrementalNIK(robot_s=ym_rbt)

# initialize the module for RRT
rrtc_planner = rrtc.RRTConnect(ym_rbt)

# generate a rack from a prototype
rack_1 = Rack_Hard_Proto.copy()
rack_1.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, -.2, 0])))
# insert test tube to the rack
rack_1.insert_tube(slot_id=np.array([0, 3]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([4, 1]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([3, 2]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([2, 3]), tube=TubeType.TUBE_TYPE_3)
print("The state after inserting the test tube rack: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)

# generate a new rack from the prototype
rack_2 = Rack_Hard_Proto.copy()
rack_2.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, 0, 0])))

# select the test tube to be removed
remove_ind = np.array([3, 2])
tube_type, tube_homomat_gl = rack_1.remove_slot(slot_id=remove_ind)
tube_move = TubeType.gen_tube_by_tubetype(tube_type)
tube_move.set_homomat(tube_homomat_gl)
tube_cm_mdl = tube_move.gen_collision_model()
tube_cm_mdl.attach_to(base)
# get rotational symmetry homogeneous matrix for the test tube
symmetry_homomat_init = tube_move.get_symmetry_homomat(discrete_factor=10)

# select the slot to be insert in the new rack
insert_ind = np.array([0, 1])
tgt_tube_pose = rack_2.get_slot_homomat(slot_id=insert_ind)
symmetry_homomat_tgt = tube_move.get_symmetry_homomat(discrete_factor=10, rot_mat=tgt_tube_pose)

# load the grasp for the tube
tube_move_grasps_list_info = fs.load_json(tube_move.grasps_path)
# generate the collision model in the environment
obs_rack_1, obs_tube_collection = rack_1.gen_collision_model(gen_tube=True)
obs_rack_2 = rack_2.gen_collision_model()

# the list to save the pick-and-place path
path = []

# search the grasp that can move the object to target pose
for ind, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(tube_move_grasps_list_info):
    print(f"--------------------- grasp pose index: {ind} ---------------------------")
    grasp_pose = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
    print(f"the homogenous matrix of the grasp pose is: {grasp_pose}")
    # find the possible ik set for the init pose
    rbt_ee_poses_init = np.dot(symmetry_homomat_init.reshape((-1, 4)), grasp_pose).reshape((-1, 4, 4))
    ik_sol_set_init = []
    for _rbt_ee_pose_init in rbt_ee_poses_init:
        ik_sol_init = ym_rbt.ik(component_name, tgt_pos=_rbt_ee_pose_init[:3, 3], tgt_rotmat=_rbt_ee_pose_init[:3, :3])
        if ik_sol_init is not None:
            ik_sol_set_init.append(ik_sol_init)
    # find the possible ik set for the goal pose
    rbt_ee_poses_tgt = np.dot(symmetry_homomat_tgt.reshape((-1, 4)), grasp_pose).reshape((-1, 4, 4))
    ik_sol_set_tgt = []
    for _rbt_ee_pose_tgt in rbt_ee_poses_tgt:
        ik_sol_tgt = ym_rbt.ik(component_name, tgt_pos=_rbt_ee_pose_tgt[:3, 3], tgt_rotmat=_rbt_ee_pose_tgt[:3, :3])
        if ik_sol_tgt is not None:
            ik_sol_set_tgt.append(ik_sol_tgt)
    if len(ik_sol_set_init) > 0 and len(ik_sol_set_tgt) > 0:
        # check collision
        for _ik_sol_init, _ik_sol_tgt in product(ik_sol_set_init, ik_sol_set_tgt):
            ym_rbt.fk(component_name, _ik_sol_init)
            is_collided_init = ym_rbt.is_collided([obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list])
            ym_rbt.fk(component_name, _ik_sol_tgt)
            is_collided_tgt = ym_rbt.is_collided([obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list])
            if is_collided_init or is_collided_tgt:  # check if is self-collided
                print(">>> The robot is collided")
                continue
            else:
                # generate the pick motion
                # move the arm to the init pose first
                ym_rbt.fk(component_name, _ik_sol_init)
                # get ee pose
                rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
                # grasp the object at init pose for collision detection only
                obj_mdl_grasped = tube_cm_mdl.copy()
                obj_mdl_grasped.set_homomat(symmetry_homomat_init[0])
                ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
                path_up = inik_svlr.gen_linear_motion(component_name,
                                                      start_tcp_pos=rbt_tcp_pos,
                                                      start_tcp_rotmat=rbt_tcp_rot,
                                                      goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                      goal_tcp_rotmat=rbt_tcp_rot,
                                                      obstacle_list=[*obs_tube_collection.cm_list, obs_rack_2],
                                                      granularity=0.01)
                ym_rbt.release(hnd_name="rgt_hnd", objcm=obj_mdl_grasped)

                # generate the place motion
                # move the arm to the target pose first
                ym_rbt.fk(component_name, _ik_sol_tgt)
                # get ee pose
                rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
                # grasp the object at init pose for collision detection only
                obj_mdl_grasped.set_homomat(symmetry_homomat_tgt[0])
                ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
                path_down_reverse = inik_svlr.gen_linear_motion(component_name,
                                                                start_tcp_pos=rbt_tcp_pos,
                                                                start_tcp_rotmat=rbt_tcp_rot,
                                                                goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                                goal_tcp_rotmat=rbt_tcp_rot,
                                                                obstacle_list=[*obs_tube_collection.cm_list,
                                                                               obs_rack_1],
                                                                granularity=0.01)
                if path_up is not None and path_down_reverse is not None:
                    path_down = path_down_reverse[:: -1]
                    rrt_path = rrtc_planner.plan(component_name=component_name,
                                                 start_conf=np.array(path_up[-1]),
                                                 goal_conf=np.array(path_down[0]),
                                                 obstacle_list=[obs_rack_1, obs_rack_2, *obs_tube_collection.cm_list],
                                                 ext_dist=.05,
                                                 max_time=300)
                    if rrt_path is not None:
                        # insert the tube
                        rack_2.insert_tube(slot_id=insert_ind, tube=tube_move)
                        # show the path
                        path = path_up + rrt_path + path_down
                        for jnts_s in path:
                            rack_1.gen_mesh_model(gen_tube=True).attach_to(base)
                            rack_2.gen_mesh_model(gen_tube=True).attach_to(base)
                            ym_rbt.fk(component_name, jnts_s)
                            ym_rbt.gen_meshmodel().attach_to(base)
                            obj_mdl_draw = obj_mdl_grasped.copy()
                            obj_mdl_draw.attach_to(base)
                        base.run()
                else:
                    print(">>> Cannot generate the path to move the object to the target pose by RRT")
                    ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
        else:
            print(">>> Cannot generate the path to raise the tube")
            ym_rbt.release("rgt_hnd", objcm=obj_mdl_grasped)
    else:
        print("No IK solution at init or target")
exit(-1)

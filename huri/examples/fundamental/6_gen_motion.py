"""
This is an example to
1. generate the linear motion for the simulation robot
2. RRT to move a object from one place to the other place
(Run the 4_define_grasp.py First)
"""
from huri.core.common_import import *
import motion.optimization_based.incremental_nik as inik
import motion.probabilistic.rrt_connect as rrtc

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
ym_rbt = ym.Yumi(enable_cc=True)
component_name = "rgt_arm"

# generate the tube to be grasped and define the initial and target pose for the object
obj_mdl = cm.CollisionModel(initor="../../models/tubebig.stl")
obj_mdl_init_pose = rm.homomat_from_posrot(np.array([0.2, -.07, 0.0]), np.eye(3))
obj_mdl_tgt_pose = rm.homomat_from_posrot(np.array([0.36, 0.07, 0.2]), np.eye(3))
obj_mdl.set_homomat(obj_mdl_init_pose)
obj_mdl.set_rgba(np.array([0, 0, 0, 1]))
obj_mdl.attach_to(base)

# initialize the module to generate the linear motion
inik_svlr = inik.IncrementalNIK(robot_s=ym_rbt)

# initialize the module for RRT
rrtc_planner = rrtc.RRTConnect(ym_rbt)

# load the grasp poses for the tube
grasps_list_info = fs.load_pickle("grasps.pkl")

# the list to save the path
path = []

# search the grasp that can move the object to target pose
for ind, (jaw_width, jaw_center_pos, jaw_center_rotmat, hnd_pos, hnd_rotmat) in enumerate(grasps_list_info):
    print(f"--------------------- grasp pose index: {ind} ---------------------------")
    # build a homogenous matrix (a 4x4 matrix with transition and orientation)
    grasp_pose = rm.homomat_from_posrot(jaw_center_pos, jaw_center_rotmat)
    print(f"the homogenous matrix of the grasp pose is: {grasp_pose}")
    # solve the IK for the initial pose of the tube
    rbt_ee_pose_init = np.dot(obj_mdl_init_pose, grasp_pose)
    ik_sol_init = ym_rbt.ik(component_name, tgt_pos=rbt_ee_pose_init[:3, 3], tgt_rotmat=rbt_ee_pose_init[:3, :3])
    # solve the IK for the target pose of the tube
    rbt_ee_pose_tgt = np.dot(obj_mdl_tgt_pose, grasp_pose)
    ik_sol_tgt = ym_rbt.ik(component_name, tgt_pos=rbt_ee_pose_tgt[:3, 3],
                           tgt_rotmat=rbt_ee_pose_tgt[:3, :3],
                           seed_jnt_values=ik_sol_init)
    ym_rbt.rgt_hnd.jaw_to(jaw_width)
    if ik_sol_init is not None and ik_sol_tgt is not None:  # check IK-feasible
        ym_rbt.fk(component_name, ik_sol_init)
        is_self_collided_init = ym_rbt.is_collided()
        ym_rbt.fk(component_name, ik_sol_tgt)
        is_self_collided_tgt = ym_rbt.is_collided()
        if is_self_collided_init or is_self_collided_tgt:  # check if is self-collided
            print(">>> The robot is self-collided")
            continue
        else:
            # generate the motion to raise the tube up by the robot
            ym_rbt.fk(component_name, ik_sol_init)
            rbt_tcp_pos, rbt_tcp_rot = ym_rbt.get_gl_tcp(component_name)
            obj_mdl_grasped = obj_mdl.copy()
            ym_rbt.hold(hnd_name="rgt_hnd", objcm=obj_mdl_grasped, jaw_width=jaw_width)
            path_up = inik_svlr.gen_linear_motion(component_name,
                                                  start_tcp_pos=rbt_tcp_pos,
                                                  start_tcp_rotmat=rbt_tcp_rot,
                                                  goal_tcp_pos=rbt_tcp_pos + np.array([0, 0, .15]),
                                                  goal_tcp_rotmat=rbt_tcp_rot,
                                                  granularity=0.01)
            if path_up is not None:
                # generate the motion to move the tube to the target pose
                rrt_path = rrtc_planner.plan(component_name=component_name,
                                             start_conf=np.array(path_up[-1]),
                                             goal_conf=np.array(ik_sol_tgt),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
                if rrt_path is not None:
                    # show the path
                    path = path_up + rrt_path
                    for jnts_s in path:
                        ym_rbt.fk(component_name, jnts_s)
                        ym_rbt.gen_meshmodel().attach_to(base)
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

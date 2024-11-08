from huri.core.common_import import wd, ym, np, fs, gm, rm, cm
from huri.components.data_annotaion._constants import *
from huri.core.constants import SENSOR_INFO
from huri.components.yumi_control.yumi_con import YumiController
import motion.probabilistic.rrt_connect as rrtc
from huri.paper_draw.tase2022.gen_obs_poses.rayhit import rayhit_check_2
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline

if __name__ == "__main__":
    base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
    yumi_s = ym.Yumi(enable_cc=True)
    init_jnt = np.array([0.93280159, -1.46939375, -1.46532976, 0.17108738, -0.26086498,
                         1.35395677, -0.87212437])

    yumi_s.fk("rgt_arm", init_jnt)
    pos_i, rot_i = yumi_s.get_gl_tcp("rgt_arm")

    # rots_candidate = np.array(rm.gen_icorotmats(icolevel=2,
    #                                             rotation_interval=np.radians(60),
    #                                             crop_normal=np.array([0, 0, 1]),
    #                                             crop_angle=np.pi / 6,
    #                                             toggleflat=True))
    #
    # for rot in rots_candidate:
    #     rot_t = np.dot(rot, rot_i)
    #     iks = yumi_s.ik("rgt_arm", pos_i, rot_t, seed_jnt_values=init_jnt)
    #     if iks is not None:
    #         yumi_s.fk("rgt_arm", iks)
    #         yumi_s.gen_meshmodel()
    #         rayhit_check_2(yumi_s.gen_meshmodel(), np.array([0.31255359, -0.15903892, 0.94915224]))
    #         print(f"np.{repr(iks)}")

    angles_x = [-20, -10, 0, 10, 20]
    angles_y = [-20, -10, 10, 20]
    jnts_list = []
    for ax in angles_x:
        ax = np.radians(ax)
        rot_t = np.dot(rm.rotmat_from_axangle(rot_i[:3, 0], ax), rot_i)
        iks = yumi_s.ik("rgt_arm", pos_i, rot_t, seed_jnt_values=init_jnt)
        if iks is not None:
            yumi_s.fk("rgt_arm", iks)
            if not yumi_s.is_collided():
                jnts_list.append(iks)
                # yumi_s.gen_meshmodel().attach_to(base)

    for ax in angles_y:
        ax = np.radians(ax)
        rot_t = np.dot(rm.rotmat_from_axangle(rot_i[:3, 2], ax), rot_i)
        iks = yumi_s.ik("rgt_arm", pos_i, rot_t, seed_jnt_values=init_jnt)
        if iks is not None:
            yumi_s.fk("rgt_arm", iks)
            if not yumi_s.is_collided():
                # yumi_s.gen_meshmodel().attach_to(base)
                jnts_list.append(iks)
    print(jnts_list)
    # initialize the robot controller
    yumi_x = YumiController()

    # the armname
    armname = "rgt_arm"  # "lft_arm"
    # generate the trajectory by RRT
    # initialize the module for RRT
    rrtc_planner = rrtc.RRTConnect(yumi_s)
    sensor_streamer = SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG)
    cnt = 0

    reco_data = []

    traj = rrtc_planner.plan(component_name=armname,
                             start_conf=yumi_x.get_jnt_values(armname),
                             goal_conf=init_jnt,
                             obstacle_list=[],
                             ext_dist=.05,
                             max_time=300)
    if traj is not None:
        # move the robot to the indicated joint angle
        yumi_x.move_jntspace_path(component_name=armname, path=traj, speed_n=300)

    input("Press any key to continue")

    for jnts in jnts_list:
        cnt += 1
        # the start joint angle of the robot
        jnt_val_start = yumi_x.get_jnt_values(armname)

        # the goal joint angles. An 1x7 np.array
        jnt_val_goal = jnts

        # the left and right arm go initial pose
        traj = rrtc_planner.plan(component_name=armname,
                                 start_conf=jnt_val_start,
                                 goal_conf=jnt_val_goal,
                                 obstacle_list=[],
                                 ext_dist=.05,
                                 max_time=300)
        if traj is not None:
            # move the robot to the indicated joint angle
            yumi_x.move_jntspace_path(component_name=armname, path=traj, speed_n=300)
            rbt_pos, rbt_rot = yumi_x.get_pose("rgt_arm")
            jnt_val = yumi_x.get_jnt_values("rgt_arm")
            pcd, texture, depth_img, rgb_texture, extcam_img = vision_pipeline(sensor_streamer,
                                                                               dump_path=None,
                                                                               rgb_texture=False)
            fs.dump_pickle([pcd, texture, jnt_val, rbt_pos, rbt_rot], f"data_3/pcd_{cnt}.pkl", reminder=False)
    # stop the connection of the real robot

    yumi_x.stop()

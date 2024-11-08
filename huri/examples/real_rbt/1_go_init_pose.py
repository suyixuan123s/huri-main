"""
Example to move the real robot to the initial pose defined in the simulation
"""
from huri.core._logging import color_logger, logging
import motion.probabilistic.rrt_connect as rrtc
from huri.core.common_import import *
from huri.components.yumi_control.yumi_con import YumiController
from motion.trajectory.piecewisepoly import PiecewisePoly


def go_init_pose(yumi_s: ym.Yumi, yumi_x: YumiController, component_name="rgt_arm", method="RRT", speed_n=300,
                 logger=logging.getLogger(__name__)):
    logger.setLevel(logging.INFO)
    logger.info(f'Start to move the robot back to the initial pose. Use RRT: {method.lower() == "rrt"}')
    if method.lower() == "rrt":
        # initialize the module for RRT
        rrtc_planner = rrtc.RRTConnect(yumi_s)
        # the left and right arm go initial pose
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            rrt_path_rgt = rrtc_planner.plan(component_name="rgt_arm",
                                             start_conf=np.array(jnt_vals_rgt),
                                             goal_conf=np.array(yumi_s.rgt_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            exe_r = yumi_x.move_jntspace_path(component_name="rgt_arm", path=rrt_path_rgt
                                              , speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            rrt_path_lft = rrtc_planner.plan(component_name="lft_arm",
                                             start_conf=np.array(jnt_vals_lft),
                                             goal_conf=np.array(yumi_s.lft_arm.homeconf),
                                             obstacle_list=[],
                                             ext_dist=.05,
                                             max_time=300)
            yumi_x.move_jntspace_path(component_name="lft_arm", path=rrt_path_lft, speed_n=speed_n)
    else:
        if component_name in ["rgt_arm", "rgt_hnd", "both"]:
            yumi_x.move_jnts(component_name="rgt_arm", jnt_vals=yumi_s.rgt_arm.homeconf, speed_n=speed_n)
        if component_name in ["lft_arm", "lft_hnd", "both"]:
            yumi_x.move_jnts(component_name="lft_arm", jnt_vals=yumi_s.lft_arm.homeconf, speed_n=speed_n)


if __name__ == "__main__":
    # define the virtual environment and the simulation robot
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    # initialize the real robot controller
    yumi_x = YumiController(debug=False)

    # get the left and right arm joints
    jnt_vals_rgt = yumi_x.get_jnt_values(component_name="rgt_arm")
    jnt_vals_lft = yumi_x.get_jnt_values(component_name="lft_arm")
    jaww_rgt = yumi_x.get_gripper_width(component_name="rgt_arm")
    jaww_lft = yumi_x.get_gripper_width(component_name="lft_arm")
    # synchronize the simulation robot with the real robot
    yumi_s.fk(component_name="rgt_arm", jnt_values=jnt_vals_rgt)
    yumi_s.fk(component_name="lft_arm", jnt_values=jnt_vals_lft)
    yumi_s.rgt_hnd.jaw_to(jaww_rgt)
    yumi_s.lft_hnd.jaw_to(jaww_lft)

    # uncomment to debug the initial status of the robot
    # yumi_s.gen_meshmodel().attach_to(base)
    # yumi_s.show_cdprimit()
    # base.run()

    # using RRT to plan the motion to go to initial pose or not
    go_init_pose(yumi_s, yumi_x, component_name="both", method="rrt", speed_n=100)

    # stop the connection of the real robot
    yumi_x.stop()

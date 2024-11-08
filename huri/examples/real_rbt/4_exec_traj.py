"""
Example to execute a trajectory for the simulation robot
"""
if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController
    import motion.probabilistic.rrt_connect as rrtc

    # initialize the simulation robot
    yumi_s = ym.Yumi(enable_cc=True)

    # initialize the robot controller
    yumi_x = YumiController()

    # the armname
    armname = "rgt_arm"  # "lft_arm"

    # the start joint angle of the robot
    jnt_val_start = yumi_x.get_jnt_values(armname)

    # the goal joint angles. An 1x7 np.array
    jnt_val_goal = np.zeros(7)

    # generate the trajectory by RRT
    # initialize the module for RRT
    rrtc_planner = rrtc.RRTConnect(yumi_s)
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

    # stop the connection of the real robot
    yumi_x.stop()

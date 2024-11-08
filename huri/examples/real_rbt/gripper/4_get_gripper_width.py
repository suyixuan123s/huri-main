"""
Example to get gripper information from the Yumi robot
"""
if __name__ == "__main__":
    from huri.components.yumi_control.yumi_con import YumiController
    from huri.core.common_import import *

    # create simulation environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # initialize the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)

    # initialize the real robot controller
    yumi_x = YumiController(debug=False)

    # get the arm joint angles of the real robot
    rgt_armjnts = yumi_x.get_jnt_values(component_name="rgt_arm")
    lft_armjnts = yumi_x.get_jnt_values(component_name="lft_arm")

    # get the gripper width of the real robot
    rgt_jawwdith = yumi_x.get_gripper_width(component_name="rgt_arm")
    lft_jawwidth = yumi_x.get_gripper_width(component_name="lft_arm")

    # move the simulation robot to the real robot
    yumi_robot.fk(component_name="rgt_arm", jnt_values=rgt_armjnts)
    yumi_robot.fk(component_name="lft_arm", jnt_values=lft_armjnts)

    # move the gripper to the real gripper width
    yumi_robot.jaw_to(hnd_name="rgt_hnd", jaw_width=rgt_jawwdith)
    yumi_robot.jaw_to(hnd_name="lft_hnd", jaw_width=lft_jawwidth)

    # generate the mesh model for the yumi robot
    yumi_robot.gen_meshmodel().attach_to(base)

    # show animation
    base.run()

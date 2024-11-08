"""
Example to get the real robot joint angles
"""
import numpy as np

if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController, ys
    from robot_con.yumi.autolab_core import RigidTransform

    # define the virtual environment and the simulation robot
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    # initialize the real robot controller
    yumi_x = YumiController(debug=False)

    # get the left and right arm joints
    jnt_vals_rgt = yumi_x.get_jnt_values(component_name="rgt_arm")
    jnt_vals_lft = yumi_x.get_jnt_values(component_name="lft_arm")
    # get the gripper width of the real robot
    rgt_jawwdith = yumi_x.get_gripper_width(component_name="rgt_arm")
    lft_jawwidth = yumi_x.get_gripper_width(component_name="lft_arm")
    # synchronize the simulation robot with the real robot
    yumi_s.fk(component_name="rgt_arm", jnt_values=jnt_vals_rgt)
    yumi_s.fk(component_name="lft_arm", jnt_values=jnt_vals_lft)
    yumi_s.rgt_hnd.jaw_to(rgt_jawwdith)
    yumi_s.lft_hnd.jaw_to(lft_jawwidth)

    pos, rot = yumi_s.get_gl_tcp("rgt_arm")
    # pos_t = pos
    pos_t = pos + np.array([.3, 0, 0])
    ikv = yumi_s.ik("rgt_arm", pos_t, rot, seed_jnt_values=jnt_vals_rgt)
    print(np.rad2deg(jnt_vals_rgt))
    print(yumi_s.get_gl_tcp("rgt_arm"))
    # print(ikv)
    if ikv is None:
        print("EEE")
        exit(-1)
    print(np.rad2deg(ikv))
    yumi_s.fk("rgt_arm",np.array(ikv))
    # yumi_x.rgt_arm_hnd.movetstate_sgl(state=ys.YuMiState(np.rad2deg(ikv)), )
    if yumi_x.rgt_arm_hnd.contactL(ys.YuMiState(np.rad2deg(ikv)),.05):
        print("GOOD")
    else:
        print("BAD")
    # print(pos_t)
    # print(rot)
    # print(f"the rgt arm joints are: {repr(jnt_vals_rgt)}")
    # print(f"the lft arm joints are: {repr(jnt_vals_lft)}")

    jnt_vals_rgt = yumi_x.get_jnt_values(component_name="rgt_arm")
    yumi_s.fk("rgt_arm",jnt_vals_rgt)
    print(yumi_s.get_gl_tcp("rgt_arm"))
    print(np.rad2deg(jnt_vals_rgt))
    yumi_x.stop()

    # generate the animation  for the simulation robot
    yumi_s.gen_meshmodel().attach_to(base)

    # stop the connection of the real robot
    base.run()

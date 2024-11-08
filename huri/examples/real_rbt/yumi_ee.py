"""
Example
"""
if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    # Define the virtual environment and the simulation robot
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    yumi_x = YumiController(debug=False)

    # get the left and right arm joints
    jnt_vals_rgt = yumi_x.get_jnt_values(component_name="rgt_arm")
    jnt_vals_lft = yumi_x.get_jnt_values(component_name="lft_arm")
    jaww_rgt = yumi_x.get_gripper_width(component_name="rgt_arm")
    jaww_lft = yumi_x.get_gripper_width(component_name="lft_arm")
    pose_rgt_raw = yumi_x.rgt_arm_hnd.get_pose()
    pose_r_rgt = rm.homomat_from_posrot(pose_rgt_raw.translation, pose_rgt_raw.rotation)
    pose_lft_raw = yumi_x.lft_arm_hnd.get_pose()
    pose_r_lft = rm.homomat_from_posrot(pose_lft_raw.translation, pose_lft_raw.rotation)
    # synchronize the simulation robot with the real robot
    yumi_s.fk(component_name="rgt_arm", jnt_values=jnt_vals_rgt)
    yumi_s.fk(component_name="lft_arm", jnt_values=jnt_vals_lft)
    yumi_s.rgt_hnd.jaw_to(jaww_rgt)
    yumi_s.lft_hnd.jaw_to(jaww_lft)

    print(f"the rgt arm joints are: {repr(jnt_vals_rgt)}")
    print(f"the lft arm joints are: {repr(jnt_vals_lft)}")
    yumi_x.stop()

    pose_s_rgt = rm.homomat_from_posrot(*yumi_s.get_gl_tcp("rgt_arm"))
    pose_s_lft = rm.homomat_from_posrot(*yumi_s.get_gl_tcp("lft_arm"))

    print(f"the rgt_arm pose in simulation is:{pose_s_rgt}")
    print(f"the rgt_arm pose in robot is:{pose_r_rgt}")

    print(
        f"the rgt arm pose different between real and sim robot is: {np.linalg.norm(pose_s_rgt[:3, 3] - pose_r_rgt[:3, 3])}")
    print(
        f"the lft arm pose different between real and sim robot is: {np.linalg.norm(pose_s_lft[:3, 3] - pose_r_lft[:3, 3])}")
    # print(f"the rgt arm pose different bewteen real and sim robot is:" )

    base.run()

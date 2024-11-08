if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    # Define the virtual environment and the simulation robot
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # sim robot
    yumi_s = ym.Yumi(enable_cc=True)
    armname = "rgt_arm"
    # real robot
    yumi_x = YumiController(debug=False)

    # get the arm joints
    jnt_vals = yumi_x.get_jnt_values(component_name=armname)
    # synchronize the simulation robot with the real robot
    yumi_s.fk(component_name=armname, jnt_values=jnt_vals)
    pose_s = rm.homomat_from_posrot(*yumi_s.get_gl_tcp(armname))
    pose_s[:3, :3] = np.round(pose_s[:3, :3], 0)/np.linalg.norm(np.round(pose_s[:3, :3], 0),axis=0)
    align_jnts = yumi_s.ik(armname, tgt_pos=pose_s[:3, 3], tgt_rotmat=pose_s[:3, :3], seed_jnt_values=jnt_vals)
    if align_jnts is not None:
        yumi_x.move_jnts(component_name=armname,
                         jnt_vals=align_jnts)
    yumi_x.stop()

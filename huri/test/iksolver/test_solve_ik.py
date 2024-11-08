import numpy as np

PI = np.pi



if __name__ == "__main__":
    from huri.core.common_import import *

    # init
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)
    yumi_s.gen_meshmodel().attach_to(base)
    #
    # move_armname = "rgt_arm"  # "lft_arm"
    # for i in range(1, 20):
    #     jnt_values = [0.93960737, -1.32485144, -0.85573201, 0.91508354, -3.3595108, -0.67104261, 0.5 * i]
    #
    #     yumi_s.fk(component_name="rgt_arm",
    #               jnt_values=np.array(jnt_values))
    #     pos, rot = yumi_s.get_gl_tcp("rgt_arm")
    #     print(rot)
    #     # print(np.rad2deg(np.round(rm.rotmat_to_euler(rot), 3)))
    #     print("\n")
    #     yumi_s.gen_meshmodel(toggle_tcpcs=True).attach_to(base)
    # base.run()

    data = fs.load_pickle("reachability_db_0.pkl")
    info = data["rgt_arm"]
    pm = info["pose_mat"]
    mv = info["manipuability_vec"]

    poses = pm[:,:3]
    gm.gen_pointcloud(poses).attach_to(base)
    base.run()


    # base.run()

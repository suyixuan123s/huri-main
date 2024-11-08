if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import *

    from huri.test.iksolver.ik_offline_utils import create_ik_db_js

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    # 38683008
    idx = 3
    fname = "jntspace_samples"
    fname_idx = f"{fname}_{idx}"

    # load sample
    ang_samples = fs.load_pickle(f"{fname_idx}.pkl")

    create_ik_db_js(yumi_s,
                    jnt_ang_set=ang_samples,
                    arb_jnt_id=7,
                    u_axis="z",
                    save_fname=f"reachability_db_{idx}",
                    num_worker=12)

    # for id, j in enumerate(ang_samples[0:5]):
    #     gen_mesh_model_at_jnts(yumi_s, np.append(j, 0)).attach_to(base)
    #     gm.gen_sphere(pos=pm[id][:3], radius=.008).attach_to(base)
    #     gm.gen_arrow(spos=pm[id][:3], epos=pm[id][:3] + pm[id][3:] * 0.05, ).attach_to(base)
    base.run()

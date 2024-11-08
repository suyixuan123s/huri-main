if __name__ == "__main__":
    import numpy as np
    from huri.core.common_import import *
    from huri.test.iksolver.ik_offline_utils import create_ik_db

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    yumi_s = ym.Yumi(enable_cc=True)

    # 38683008
    idx = 0
    fname = "space_samples"
    fname_idx = f"{fname}_{idx}"

    # load sample
    pos_sample, ang_sample = fs.load_pickle(f"{fname_idx}.pkl")

    # load previous file len
    past_slice_len = 0
    for i in range(idx):
        past_pos_sample_i, _ = fs.load_pickle(f"{fname}_{i}.pkl")
        past_slice_len += len(past_pos_sample_i)
        del past_pos_sample_i

    print(f"The file going to be loaded is {fname_idx}.pkl. "
          f"Past file len is {past_slice_len}")

    create_ik_db(yumi_s, pos_sample, ang_sample, past_slice_len=past_slice_len, num_worker=6)

    base.run()

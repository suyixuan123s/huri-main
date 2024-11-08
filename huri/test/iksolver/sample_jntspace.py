from huri.test.iksolver.ik_offline_utils import save_jntspacesamples, sample_jnt_space
import numpy as np
from typing import Union
import huri.core.file_sys as fs

PI = np.pi


def sample_rbt_jnt_space(jnt_ranges,
                         ang_sample_dense: Union[tuple, list, np.ndarray, float] = PI / 12):
    # only sample [-pi, pi)
    jnt_ranges_mapped = []
    for jnt_range in jnt_ranges:
        # TODO some robots's joint limit may lose set
        tmp_jnt_range = (max(-PI, jnt_range[0]), min(PI, jnt_range[1]))
        jnt_ranges_mapped.append(tmp_jnt_range)
    ang_sample = sample_jnt_space(jnt_space_range=jnt_ranges_mapped, ang_sample_dense=ang_sample_dense)
    return ang_sample


if __name__ == "__main__":
    from huri.core.common_import import *

    yumi_s = ym.Yumi(enable_cc=True)

    armname = "rgt_arm"

    jnt_ranges = yumi_s.manipulator_dict[armname].get_jnt_ranges()
    # for yumi only sample the 6 jnts (last joint is rotation, not affect tcp pos and z direction)
    jnt_ranges_6 = jnt_ranges[:-1]
    ang_samples = sample_rbt_jnt_space(jnt_ranges=jnt_ranges_6,
                                      ang_sample_dense=PI / 12)
    save_jntspacesamples(ang_samples, slice_num=4)

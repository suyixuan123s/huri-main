import numpy as np
from numpy import array
import huri.core.file_sys as fs


data = array([[ 2.82079054e-03, -1.00400178e+00, -5.74846621e-04,
         3.12553590e-01],
       [-9.82727430e-01, -7.97055000e-03,  1.97950550e-01,
        -1.58038920e-01],
       [-2.02360828e-01,  5.46017392e-03, -9.68000060e-01,
         9.49152240e-01],
       [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
         1.00000000e+00]])
path = fs.workdir / "data" / "calibration" / "qaqqq4.json"
fs.dump_json(data={'affine_mat': data.tolist()}, path=path)
# _,rotmat = yumi_s.get_gl_tcp("rgt_arm")
#     affine_mat[:3,3] = affine_mat[:3,3] - rotmat[:3,2]*10/1000
#     print(repr(affine_mat))

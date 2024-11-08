"""
This is the example to sample the data using the movable platform
"""

# import the necessary library
import numpy as np
from numpy import array
import random
from huri.core.common_import import *
from robot_sim.xyz_platform.xyz_platform import XYZPlatform
from robot_con.xyz_platform.xyz_platform import XYZPlatformController
from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
from time import strftime

AFFINE_MAT_PATH = fs.workdir / "data" / "calibration" / "qaqqq.json"

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
ym_rbt = ym.Yumi(enable_cc=True)
# generate the model of the yumi robot and attach to the environment
ym_rbt_mdl = ym_rbt.gen_meshmodel()
ym_rbt_mdl.attach_to(base)

# generate the xyz platform
xyz_platform = XYZPlatform()
pos_r = array([0.379     , 0.059     ,(4.08407045)-np.radians(65)])

xyz_platform.fk("all", pos_r)
xyz_platform.gen_meshmodel().attach_to(base)
# get pcd
DEBUG = True
affine_mat = np.asarray(
    fs.load_json(AFFINE_MAT_PATH)['affine_mat'])
if DEBUG:
    filename = fs.workdir / "data" / "vision_exp" / "exp_20211122-081626.pkl"
    pcd, img = fs.load_pickle(filename)
else:
    pcd, img = vision_pipeline(SensorMarkerHandler(),
                               fs.workdir / "data" / "vision_exp" / f"exp_{strftime('%Y%m%d-%H%M%S')}.pkl")
pcd = rm.homomat_transform_points(affine_mat, points=pcd)
gm.gen_pointcloud(pcd, [[0, 0, 0, .3]]).attach_to(base)
base.run()
# generate the mesh model
# # TODO install the zero position sensor to the movable platform
xyz_con = XYZPlatformController()
# !!! currently, zero position is manually set up. Pay attention when setting the zero position
# xyz_con.calibrate()
# for i in range(5):
#     pos_r = np.array(
#         [random.randrange(300, 450) / 1000, random.randrange(50, 350) / 1000, np.radians(random.randrange(0, 330))])
#     print(repr(pos_r))
#     xyz_con.set_pos("yxz", pos_r)
#     input("aaaa")
#     pcd, img = vision_pipeline(SensorMarkerHandler(),
#                                fs.workdir / "data" / "vision_exp" / f"exp_{strftime('%Y%m%d-%H%M%S')}.pkl")

exit(0)

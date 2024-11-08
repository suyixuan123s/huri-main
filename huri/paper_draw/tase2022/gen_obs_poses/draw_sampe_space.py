import numpy as np

from huri.core.common_import import wd, cm, fs, ym, gm, rm
from huri.core.base_boost import zoombase, boost_base
from huri.paper_draw.utils import phoxi
from huri.definitions.tube_def import TubeType
from huri.core.constants import SENSOR_INFO

base = wd.World(cam_pos=[3, 0, 2], lookat_pos=[.3, 0, .5])
robot = ym.Yumi()
# robot.fk("rgt_arm", np.array([0.99291781, -1.52541777, -1.52925749, 0.3122394, -0.33946654,
#                               1.15313904, -0.8545132]))


# phoxi.set_pos(np.array([0.27, 0, 1.05]))
# phoxi.set_rpy(np.radians(90), 0, np.radians(90))
#

affine_mat = np.asarray(
    fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)['affine_mat'])

gm.gen_sphere(affine_mat[:3, 3], rgba=[0, 1, 0, 1], radius=.01).attach_to(base)

phoxi.set_pos(affine_mat[:3, 3] + np.dot(affine_mat[:3, :3], phoxi.origin))
raw, pitch, yell = rm.rotmat_to_euler(affine_mat[:3, :3])
phoxi.set_rpy(np.radians(90), 0, yell)
gm.gen_sphere(phoxi.get_origin(), radius=.01).attach_to(base)
phoxi.attach_to(base)
print(affine_mat[:3, 3])
print(phoxi.origin)
# pos, rot = robot.get_gl_tcp("rgt_arm")
#
# purple_tube = TubeType.gen_tube_by_name("purple cap")
# purple_tube_cm = purple_tube.gen_collision_model()
# purple_tube_cm.set_pos(pos - rot[:3,2]*0.03 + rot[:3,1]*0.05)
# # purple_tube_cm.set_rotmat(rot)
# purple_tube_cm.attach_to(base)
#
# robot.jaw_to("rgt_hnd", 0.019)
robot.gen_meshmodel().attach_to(base)

# for pos in np.array(np.meshgrid(np.linspace(.2, .56, num=7),
#                                 np.linspace(-.3, .3, num=20),
#                                 np.array([.15]))).T.reshape(-1, 3):
#     sampled_pos = cm.CollisionModel(gm.gen_sphere(pos, radius=.01, ))
#     sampled_pos.attach_to(base)

for pos in np.array(np.meshgrid(np.linspace(.2, .5, num=5),
                                np.linspace(-.25, .25, num=10),
                                np.array([.2]))).T.reshape(-1, 3):
    sampled_pos = cm.CollisionModel(gm.gen_sphere(pos, radius=.01, ))
    sampled_pos.attach_to(base)

# for pos in np.array(np.meshgrid(np.linspace(.2, .57, num=7),
#                                     np.linspace(-.3, .3, num=20),
#                                     np.array([.1]))).T.reshape(-1, 3):
#     sampled_pos = cm.CollisionModel(gm.gen_sphere(pos, radius=.01,))
#     sampled_pos.attach_to(base)


# # phoxi.objpdnp.place()
base = boost_base(base)
# base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
base.boost.add_task(lambda task: base.boost.screen_shot("robot_sample"), timestep=0.4)
# base.boost.add_task(lambda task: exit(), timestep=0.6)

base.run()

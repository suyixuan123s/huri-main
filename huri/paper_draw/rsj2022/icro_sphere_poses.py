import numpy as np

from basis.trimesh.creation import icosphere
from huri.core.common_import import cm, wd, gm, rm
from huri.definitions.tube_def import TubeType
from huri.core.base_boost import zoombase, boost_base
from robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper import YumiGripper

img_name = "ico"
base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])

rots_candidate = np.array(rm.gen_icorotmats(icolevel=3,
                                                    rotation_interval=np.radians(60),
                                                    crop_normal=np.array([0, 0, -1]),
                                                    crop_angle=np.pi / 6,
                                                    toggleflat=True))

# rots_candidate2 = np.array(rm.gen_icorotmats(icolevel=3,
#                                                     rotation_interval=np.radians(60),
#                                                     crop_normal=np.array([0, 0, 1]),
#                                                     # crop_angle=np.pi / 6,
#                                                     toggleflat=True))

grasp = YumiGripper()
# grasp.jaw_to(.03)
# grasp.gen_meshmodel().attach_to(base)
alpha = 1
# for i in rots_candidate2:
#     grasp_tmp = grasp.copy()
#     grasp_tmp.grip_at_with_jcpose(np.zeros((3)),i, 0.02)
#     mesh_m = grasp_tmp.gen_meshmodel().cm_list
#     for j in mesh_m:
#         rgba = j.get_rgba()
#         j.set_rgba([rgba[0], rgba[1], rgba[2], alpha])
#         j.attach_to(base)
#     alpha = .1
    # tube.set_homomat(rm.homomat_from_posrot(rot=i))
    # tube.gen_mesh_model().attach_to(base)


tube = TubeType.gen_tube_by_name("purple cap")

for i in rots_candidate:
    # grasp_tmp = grasp.copy()
    # grasp_tmp.grip_at_with_jcpose(np.zeros((3)),i, 0.02)
    # grasp_tmp.gen_meshmodel().attach_to(base)
    tube.set_homomat(rm.homomat_from_posrot(rot=i))
    tube_mesh = tube.gen_collision_model()
    tube_mesh.attach_to(base)




base = boost_base(base)

base.boost.add_task(zoombase, args=[base, np.array([1, 0, 1])], timestep=0.2)
base.boost.add_task(lambda task: base.boost.screen_shot(img_name), timestep=0.4)
base.boost.add_task(lambda task: exit(), timestep=0.6)
base.run()
from huri.paper_draw.rsj2022.conv2eps import remove_white_space
remove_white_space(f"{img_name}.jpg")
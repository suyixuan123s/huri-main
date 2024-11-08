import numpy as np

from basis.trimesh.creation import icosphere
from huri.core.common_import import cm, wd, gm, rm

from huri.core.base_boost import zoombase, boost_base

img_name = "ico"
base = wd.World(cam_pos=[3.4, 0, .6], lookat_pos=[.3, 0, .5])
tm = icosphere(subdivisions=3)
# for i in tm.vertices:
#     gm.gen_sphere(i, radius=.01).attach_to(base)

for i, f_id in enumerate(tm.faces):
    center = np.average(tm.vertices[f_id],axis=0)
    center_normal = tm.face_normals[i]
    rgba = [0,1,0,1]
    if rm.angle_between_vectors(center_normal, np.array([0,0,1])) > np.pi/6:
        rgba = [1,0,0,1]

    random_color = rm.random_rgba()
    gm.gen_arrow(spos=center,epos=center+center_normal*0.3,rgba=rgba, thickness=.03).attach_to(base)
    gm.gen_sphere(center, radius=.01).attach_to(base)
icosphere_cm = cm.CollisionModel(tm)
icosphere_cm.set_rgba([1,1,1,1])
# icosphere_cm.objpdnp.setBin("fixed",10)

icosphere_cm.attach_to(base)
# icosphere_cm.show_cdmesh()

vertices, vertex_normals, faces = icosphere_cm.extract_rotated_vvnf()
objwm = gm.WireFrameModel(cm.da.trm.Trimesh(vertices=vertices, vertex_normals=vertex_normals, faces=faces))
objwm.attach_to(base)
icosphere_cm.set_scale([.995,.995,.995])
base = boost_base(base)

base.boost.add_task(zoombase, args=[base, np.array([0.8, 0, 0.8])], timestep=0.2)
base.boost.add_task(lambda task: base.boost.screen_shot(img_name), timestep=0.4)
base.boost.add_task(lambda task: exit(), timestep=0.6)
base.run()
from huri.paper_draw.rsj2022.conv2eps import remove_white_space
remove_white_space(f"{img_name}.jpg")
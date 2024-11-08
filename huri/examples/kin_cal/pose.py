from huri.components.yumi_control.yumi_con import YumiController
from huri.core.common_import import *
import numpy as np
if __name__ == "__main__":
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

    yumix = YumiController()

    component_name = "rgt_arm"

    pos_x, rot_x = yumix.get_pose(component_name=component_name)
    jnts = yumix.get_jnt_values(component_name=component_name)

    yumi_s = ym.Yumi()
    yumi_s.fk(component_name=component_name, jnt_values=jnts)
    pos_s, rot_s = yumi_s.get_gl_tcp(manipulator_name=component_name)

    yumi_s.gen_meshmodel().attach_to(base)

    gm.gen_frame(pos_s,rot_s,alpha=.7).attach_to(base)
    gm.gen_frame(pos_x,rot_x,alpha=.7).attach_to(base)
    print(np.linalg.norm(pos_s-pos_x))
    base.run()
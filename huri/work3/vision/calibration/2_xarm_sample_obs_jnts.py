from robot_con.xarm_lite6.xarm_lite6_x import XArmLite6X
from robot_sim.robots.xarm_lite6_wrs.xarm_lite6_wrs import XArmLite6WRSGripper
import visualization.panda.world as wd
import modeling.geometric_model as gm

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
rbtx = XArmLite6X()
rbt_pose = rbtx.get_pose()
rbt_jnts = rbtx.get_jnt_values()
print(f"np.{repr(rbt_jnts)}")
del rbtx

rbt = XArmLite6WRSGripper()
rbt.fk("arm", rbt_jnts)
rbt.gen_meshmodel().attach_to(base)
gm.gen_frame(rbt_pose[0], rbt_pose[1]).attach_to(base)
print(rbt_pose[0], rbt_pose[1])
print(rbt.get_gl_tcp("arm"))
base.run()

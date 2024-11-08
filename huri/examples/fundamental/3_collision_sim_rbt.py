"""
This is an example to
1. Show the collision model of the simulation robot
2. Check the collision between the robot and objects
"""

# import the necessary library
from huri.core.common_import import *

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
ym_rbt = ym.Yumi(enable_cc=True)
armname = "rgt_arm"
jnts = np.array([0.93960737, -1.32485144, -0.85573201, 0.91508354, -3.3595108, -0.67104261, 1.43773474])
ym_rbt.fk(armname, jnts)

# generate the model of the yumi robot and attach to the environment
ym_rbt_mdl = ym_rbt.gen_meshmodel()
ym_rbt_mdl.attach_to(base)

# show the primitive collision model of the robot
ym_rbt_mdl.show_cdprimit()

# generate a black and white tube
obj_mdl_prototype = cm.CollisionModel(initor="../../models/tubebig.stl")
obj_mdl_black = obj_mdl_prototype.copy()
obj_mdl_black.set_pos(np.array([0.36, -.07, 0.3]))
obj_mdl_black.set_rgba(np.array([0, 0, 0, 1]))
obj_mdl_black.attach_to(base)

obj_mdl_white = obj_mdl_prototype.copy()
obj_mdl_white.set_pos(np.array([0.36, .07, 0.3]))
obj_mdl_white.set_rgba(np.array([1, 1, 1, 1]))
obj_mdl_white.attach_to(base)

# check collision between the robot and black and white tubes
print(f"Does the robot collide with black tube: {ym_rbt.is_collided([obj_mdl_black])}")
print(f"Does the robot collide with white tube: {ym_rbt.is_collided([obj_mdl_white])}")

base.run()

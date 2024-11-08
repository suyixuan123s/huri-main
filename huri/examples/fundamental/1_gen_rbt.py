"""
This is an example to
1. Generate the simulation robot
2. Set the joints of the simulation robot
3. Import the model into the scene
4. Set the position of the imported model
"""

# import the necessary library
from huri.core.common_import import *

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot
ym_rbt = ym.Yumi(enable_cc=True)

# move the joints (forward kinematics) of the yumi robot
ym_rbt.fk("rgt_arm", np.array([30, 0, 60, 0, -30, 0, 0]))
ym_rbt.fk("lft_arm", np.array([-30, 0, 60, 0, -30, 0, 0]))

# generate the model of the yumi robot and attach to the environment
ym_rbt_mdl = ym_rbt.gen_meshmodel()
ym_rbt_mdl.attach_to(base)

# import the model and attach to virtual environment
obj_mdl = gm.GeometricModel(initor="../../models/tubebig.stl")
obj_mdl.attach_to(base)

# set color and the location of the object
obj_mdl.set_rgba(np.array([0, 1, 0, 1]))
obj_mdl.set_pos(np.array([.4, 0, 0]))

base.run()

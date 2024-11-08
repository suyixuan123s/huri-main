"""
This is an example to
1. Show the collision model of the object
2. Check collision between objects
"""

# import the necessary library
from huri.core.common_import import *

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# import the model using Collision Model
obj_mdl_prototype = cm.CollisionModel(initor="../../models/tubebig.stl")

# Show the collision model of the object
# There are two types of collision model: 1. Primitive Collision Model 2. Mesh Collision Model
# 1. The primitive collision model
obj_mdl_cdprimitive = obj_mdl_prototype.copy()
obj_mdl_cdprimitive.set_rgba(np.array([1, 0, 0, 1]))  # Shown in red
obj_mdl_cdprimitive.set_pos(np.array([0, -.2, 0]))
obj_mdl_cdprimitive.show_cdprimit()
obj_mdl_cdprimitive.attach_to(base)

# 2. The mesh collision model (It takes longer time to generate the collision model)
obj_mdl_cdmesh = obj_mdl_prototype.copy()
obj_mdl_cdmesh.set_pos(np.array([0, .2, 0]))
obj_mdl_cdmesh.set_rgba(np.array([0, 1, 0, 1]))  # Show in green
obj_mdl_cdmesh.show_cdmesh()
obj_mdl_cdmesh.attach_to(base)

# Generate a black and a white tube
obj_mdl_black = obj_mdl_prototype.copy()
obj_mdl_black.set_pos(np.array([0.2, .01, 0]))
obj_mdl_black.set_rgba(np.array([0, 0, 0, 1]))
obj_mdl_black.attach_to(base)

obj_mdl_white = obj_mdl_prototype.copy()
obj_mdl_white.set_pos(np.array([0.2, -.01, 0]))
obj_mdl_white.set_rgba(np.array([1, 1, 1, 1]))
obj_mdl_white.attach_to(base)

# check collision between black and white objects
print(f"Does the black tube collide with white tube "
      f"by using the primitive collision model: {obj_mdl_black.is_pcdwith(obj_mdl_white)}")

print(f"Does the black tube collide with white tube "
      f"by using the mesh collision model: {obj_mdl_black.is_mcdwith(obj_mdl_white)}")

base.run()

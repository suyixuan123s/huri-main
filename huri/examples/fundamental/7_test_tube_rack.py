"""
This is a example to
1. read status from the test tube rack
2. insert a test tube into a test tube rack
3. remove a test tube from a slot
"""
from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
from huri.core.common_import import *
from huri.core.print_tool import text_pd

# create the virtual environment
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# generate the yumi robot and its mesh model
ym_rbt = ym.Yumi(enable_cc=True)
ym_rbt.gen_meshmodel().attach_to(base)

# generate a rack from a prototype
rack_1 = Rack_Hard_Proto.copy()
# set the pos for the rack
rack_1.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, -.2, 0])))
# print the status of the test tube using the panda format
print("The current state of the test tube rack: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)

# insert test tube to the rack
rack_1.insert_tube(slot_id=np.array([0, 0]), tube=TubeType.TUBE_TYPE_1)
rack_1.insert_tube(slot_id=np.array([0, 1]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([0, 2]), tube=TubeType.TUBE_TYPE_4)
rack_1.insert_tube(slot_id=np.array([0, 3]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([4, 1]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([3, 2]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([4, 6]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([3, 3]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([2, 3]), tube=TubeType.TUBE_TYPE_2)
print("The state after inserting the test tube rack: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)

# remove a test tube from the rack
remove_ind = np.array([3, 3])
tube_type, tube_homomat_gl = rack_1.remove_slot(slot_id=remove_ind)
print(f"The state after removing a test tube at {remove_ind} from rack: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)
# attach the tube to be removed to the virtual environment. The tube is in red color
tube_r = TubeType.gen_tube_by_tubetype(tube_type)
tube_mdl = tube_r.gen_mesh_model()
tube_mdl.set_rgba([1, 0, 0, 1])
tube_mdl.set_homomat(tube_homomat_gl)
tube_mdl.attach_to(base)

# generate the mesh model for the rack and tubes
rack_1_mdl = rack_1.gen_mesh_model(gen_tube=True)
rack_1_mdl.attach_to(base)

# generate a new rack from prototype
rack_2 = Rack_Hard_Proto.copy()
rack_2.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, 0, 0])))
# insert the removed tube to the new rack
rack_2.insert_tube(slot_id=np.array([4, 4]), tube=tube_r)
# generate the mesh model for the new rack
rack_2_mdl = rack_2.gen_mesh_model(gen_tube=True)
rack_2_mdl.attach_to(base)

base.run()

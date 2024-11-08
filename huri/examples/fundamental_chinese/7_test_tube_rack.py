"""
这是一个示例，用于：
1. 读取试管架的状态
2. 将试管插入试管架
3. 从指定位置移除试管
"""
from huri.definitions.rack_def import Rack_Hard_Proto, TubeType
from huri.core.common_import import *
from huri.core.print_tool import text_pd

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 生成 Yumi 机器人及其模型
ym_rbt = ym.Yumi(enable_cc=True)
ym_rbt.gen_meshmodel().attach_to(base)

# 从原型生成试管架
rack_1 = Rack_Hard_Proto.copy()
# 设置试管架的位置
rack_1.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, -.2, 0])))
# 使用 pandas 格式打印当前试管架的状态
print("试管架的当前状态: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)

# 向试管架中插入试管
rack_1.insert_tube(slot_id=np.array([0, 0]), tube=TubeType.TUBE_TYPE_1)
rack_1.insert_tube(slot_id=np.array([0, 1]), tube=TubeType.TUBE_TYPE_3)
rack_1.insert_tube(slot_id=np.array([0, 2]), tube=TubeType.TUBE_TYPE_4)
rack_1.insert_tube(slot_id=np.array([0, 3]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([4, 1]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([3, 2]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([4, 6]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([3, 3]), tube=TubeType.TUBE_TYPE_2)
rack_1.insert_tube(slot_id=np.array([2, 3]), tube=TubeType.TUBE_TYPE_2)
print("插入试管后的试管架状态: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)

# 从试管架中移除试管
remove_ind = np.array([3, 3])
tube_type, tube_homomat_gl = rack_1.remove_slot(slot_id=remove_ind)
print(f"移除试管后在位置 {remove_ind} 的试管架状态: ")
print(text_pd(rack_1.rack_status))
print("-" * 30)
# 将移除的试管添加到虚拟环境中，试管为红色
tube_r = TubeType.gen_tube_by_tubetype(tube_type)
tube_mdl = tube_r.gen_mesh_model()
tube_mdl.set_rgba([1, 0, 0, 1])
tube_mdl.set_homomat(tube_homomat_gl)
tube_mdl.attach_to(base)

# 为试管架和试管生成网格模型
rack_1_mdl = rack_1.gen_mesh_model(gen_tube=True)
rack_1_mdl.attach_to(base)

# 从原型生成新的试管架
rack_2 = Rack_Hard_Proto.copy()
rack_2.set_homomat(rm.homomat_from_posrot(pos=np.array([0.4, 0, 0])))
# 将移除的试管插入到新的试管架中
rack_2.insert_tube(slot_id=np.array([4, 4]), tube=tube_r)
# 为新的试管架生成网格模型
rack_2_mdl = rack_2.gen_mesh_model(gen_tube=True)
rack_2_mdl.attach_to(base)

base.run()  # 运行虚拟环境

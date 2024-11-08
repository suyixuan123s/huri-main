import os
from huri.core.common_import import *

# 获取文件的完整路径
file_path = "../../models/20220610_blue_tube.stl"
full_path = os.path.abspath(file_path)

print(f"Full path of the file: {full_path}")

# 使用碰撞模型导入该文件
obj_mdl_prototype = cm.CollisionModel(initor=full_path)

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 显示对象的碰撞模型
obj_mdl_prototype.set_rgba(np.array([0, 0, 1, 1]))  # 设为蓝色显示
obj_mdl_prototype.attach_to(base)

# 启动虚拟环境
base.run()

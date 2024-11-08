"""
这是一个示例，用于
1. 显示物体的碰撞模型
2. 检查物体之间的碰撞
"""

# 导入必要的库
from huri.core.common_import import *

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 使用碰撞模型导入模型
obj_mdl_prototype = cm.CollisionModel(initor="../../models/tubebig.stl")

# 显示物体的碰撞模型
# 碰撞模型有两种类型：1. 原始碰撞模型 2. 网格碰撞模型
# 1. 原始碰撞模型
obj_mdl_cdprimitive = obj_mdl_prototype.copy()
obj_mdl_cdprimitive.set_rgba(np.array([1, 0, 0, 1]))  # 显示为红色
obj_mdl_cdprimitive.set_pos(np.array([0, -.2, 0]))
obj_mdl_cdprimitive.show_cdprimit()
obj_mdl_cdprimitive.attach_to(base)

# 2. 网格碰撞模型（生成碰撞模型需要更长的时间）
obj_mdl_cdmesh = obj_mdl_prototype.copy()
obj_mdl_cdmesh.set_pos(np.array([0, .2, 0]))
obj_mdl_cdmesh.set_rgba(np.array([0, 1, 0, 1]))  # 显示为绿色
obj_mdl_cdmesh.show_cdmesh()
obj_mdl_cdmesh.attach_to(base)

# 生成一个黑色和一个白色的管状物体
obj_mdl_black = obj_mdl_prototype.copy()
obj_mdl_black.set_pos(np.array([0.2, .01, 0]))
obj_mdl_black.set_rgba(np.array([0, 0, 0, 1]))
obj_mdl_black.attach_to(base)

obj_mdl_white = obj_mdl_prototype.copy()
obj_mdl_white.set_pos(np.array([0.2, -.01, 0]))
obj_mdl_white.set_rgba(np.array([1, 1, 1, 1]))
obj_mdl_white.attach_to(base)

# 检查黑色物体和白色物体之间的碰撞
print(f"黑色管道是否与白色管道发生碰撞（使用原始碰撞模型）："
      f"{obj_mdl_black.is_pcdwith(obj_mdl_white)}")

print(f"黑色管道是否与白色管道发生碰撞（使用网格碰撞模型）："
      f"{obj_mdl_black.is_mcdwith(obj_mdl_white)}")

# 运行虚拟环境
base.run()



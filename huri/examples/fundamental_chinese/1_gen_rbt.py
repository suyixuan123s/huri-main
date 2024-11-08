"""
这是一个示例，用于
1. 生成仿真机器人
2. 设置仿真机器人的关节
3. 将模型导入场景
4. 设置导入模型的位置
"""

# 导入必要的库
from huri.core.common_import import *

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 生成 Yumi 机器人
ym_rbt = ym.Yumi(enable_cc=True)

# 移动 Yumi 机器人的关节（正向运动学）
ym_rbt.fk("rgt_arm", np.array([30, 0, 60, 0, -30, 0, 0]))  # 设置右臂的关节角度
ym_rbt.fk("lft_arm", np.array([-30, 0, 60, 0, -30, 0, 0]))  # 设置左臂的关节角度

# 生成 Yumi 机器人的模型并附加到环境中
ym_rbt_mdl = ym_rbt.gen_meshmodel()
ym_rbt_mdl.attach_to(base)

# 导入模型并附加到虚拟环境
obj_mdl = gm.GeometricModel(initor="../../models/tubebig.stl")
obj_mdl.attach_to(base)

# 设置物体的颜色和位置
obj_mdl.set_rgba(np.array([0, 1, 0, 1]))  # 设置物体颜色为绿色
obj_mdl.set_pos(np.array([.4, 0, 0]))  # 设置物体的位置

# 运行虚拟环境
base.run()


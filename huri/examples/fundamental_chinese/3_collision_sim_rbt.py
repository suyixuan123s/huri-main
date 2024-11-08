"""
这是一个示例，用于
1. 显示仿真机器人的碰撞模型
2. 检查机器人和物体之间的碰撞
"""

# 导入必要的库
from huri.core.common_import import *

# 创建虚拟环境
base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])

# 生成 Yumi 机器人
ym_rbt = ym.Yumi(enable_cc=True)
armname = "rgt_arm"  # 设置要操作的手臂为右臂
jnts = np.array([0.93960737, -1.32485144, -0.85573201, 0.91508354, -3.3595108, -0.67104261, 1.43773474])
ym_rbt.fk(armname, jnts)  # 应用关节角度，进行正向运动学求解


# 生成 Yumi 机器人的模型并附加到环境中
ym_rbt_mdl = ym_rbt.gen_meshmodel()
ym_rbt_mdl.attach_to(base)


# 显示机器人的简化碰撞检测模型
ym_rbt_mdl.show_cdprimit()

# 创建一个黑白颜色的管状物体
obj_mdl_prototype = cm.CollisionModel(initor="../../models/tubebig.stl")  # 从文件加载模型
obj_mdl_black = obj_mdl_prototype.copy()  # 复制模型生成黑色管道
obj_mdl_black.set_pos(np.array([0.36, -.07, 0.3]))  # 设置黑色管道的位置
obj_mdl_black.set_rgba(np.array([0, 0, 0, 1]))  # 设置黑色管道的颜色
obj_mdl_black.attach_to(base)  # 将黑色管道添加到环境中

obj_mdl_white = obj_mdl_prototype.copy()  # 复制模型生成白色管道
obj_mdl_white.set_pos(np.array([0.36, .07, 0.3]))  # 设置白色管道的位置
obj_mdl_white.set_rgba(np.array([1, 1, 1, 1]))  # 设置白色管道的颜色
obj_mdl_white.attach_to(base)  # 将白色管道添加到环境中

# 检查机器人与黑白管道之间的碰撞
print(f"机器人是否与黑色管道碰撞: {ym_rbt.is_collided([obj_mdl_black])}")
print(f"机器人是否与白色管道碰撞: {ym_rbt.is_collided([obj_mdl_white])}")

# 运行仿真环境
base.run()




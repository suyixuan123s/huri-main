import numpy as np

from huri.components.yumi_control.yumi_con import YumiController

if __name__ == "__main__":
    yumi_x = YumiController(debug=False)
    # yumi_x.go_zero_pose()

    rgt_jnt = np.array([-0.34906585, -1.57079633, -2.0943951, 0.52359878, 0.,
                        0.6981317, 0.])
    lft_jnt = np.array([0.34906585, -1.57079633, 2.0943951, 0.52359878, 0.,
                        0.6981317, 0.])
    yumi_x.move_jnts("rgt_arm", rgt_jnt, speed_n=200)
    yumi_x.move_jnts("lft_arm", lft_jnt, speed_n=200)
    yumi_x.lft_arm_hnd.calibrate_gripper()
    yumi_x.stop()

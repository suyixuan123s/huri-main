"""
Example to move the real robot to an indicate joint angles
"""
if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    # initialize the robot controller
    yumi_x = YumiController()

    # the armname
    armname = "rgt_arm"  # "lft_arm"

    # the goal joint angles. An 1x7 np.array
    jnt_values = np.zeros(7)

    # move the robot to the goal joint angle
    yumi_x.move_jnts(component_name=armname, jnt_vals=jnt_values, speed_n=300)
    # speed_n: speed number. If speed_n = 100, then speed will be set to the corresponding v100
    # specified in RAPID. Loosely, n is translational speed in milimeters per second
    # Please refer to page 1186 of
    # https://library.e.abb.com/public/688894b98123f87bc1257cc50044e809/Technical%20reference%20manual_RAPID_3HAC16581-1_revJ_en.pdf

    # stop the connection of the real robot
    yumi_x.stop()

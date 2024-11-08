"""
Example to open the Yumi's grippers
"""
if __name__ == "__main__":
    from huri.components.yumi_control.yumi_con import YumiController

    # initialize the real robot controller
    yumi_x = YumiController(debug=False)

    # open the gripper of the yumi
    yumi_x.open_gripper(component_name="rgt_hnd")
    yumi_x.open_gripper(component_name="lft_hnd")
    yumi_x.stop()

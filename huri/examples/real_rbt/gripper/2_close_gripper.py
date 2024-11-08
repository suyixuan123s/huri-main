"""
Example to close the Yumi's gripper
"""
if __name__ == "__main__":
    from huri.components.yumi_control.yumi_con import YumiController

    # initialize the real robot controller
    yumi_x = YumiController(debug=False)

    # close the gripper
    yumi_x.close_gripper(component_name="rgt_hnd")
    yumi_x.close_gripper(component_name="lft_hnd")
    yumi_x.stop()

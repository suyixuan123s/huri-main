"""
Example to calibrate the gripper
If you want to use the Yumi's gripper, calibrate it first!
"""
if __name__ == "__main__":
    from huri.components.yumi_control.yumi_con import YumiController

    # initialize the real robot controller
    yumi_x = YumiController(debug=False)

    # calibrate the gripper of yumi first when using the gripper
    yumi_x.calibrate_gripper()
    yumi_x.stop()

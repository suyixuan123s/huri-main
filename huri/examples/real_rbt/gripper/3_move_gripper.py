"""
Example to move Yumi's gripper to a indicated width. The range of the gripper is 0-50 mm
(Yumi's gripper may not open to exactly 50 mm due to calibration deviation.
If you want to fully open the gripper, use opengripper (see 1_open_gripper.py))
"""
if __name__ == "__main__":
    from huri.components.yumi_control.yumi_con import YumiController
    from huri.math.units import Mm

    # initialize the real robot controller
    yumi_x = YumiController(debug=False)

    # move gripper to 40 mm (range is [0, 50 mm)
    yumi_x.set_gripper_width(component_name="rgt_hnd", width=Mm(40))
    yumi_x.set_gripper_width(component_name="lft_hnd", width=Mm(40))
    yumi_x.stop()

import unittest

import numpy as np

from huri.components.yumi_control.yumi_con import YumiController

class TestYumiControl(unittest.TestCase):
    def setUp(self) -> None:
        self.yumi_con = YumiController()

    def test_get_pose(self):
        rgt_arm_jnts = self.yumi_con.get_pose("rgt_arm")
        lft_arm_jnts = self.yumi_con.get_pose("lft_arm")
        self.assertIs(rgt_arm_jnts, np.ndarray)
        self.assertIs(lft_arm_jnts, np.ndarray)
#

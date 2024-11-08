import unittest
from huri.learning.env.rack_v3.env import GoalRackStateScheduler2, RackStatePlot, GoalRackStateScheduler3
import numpy as np
import cv2


class SchedulerTest(unittest.TestCase):
    def test_scheduler2(self):
        num_class = 3
        rack_sz = (5, 10)

        scheduler = GoalRackStateScheduler2(num_classes=num_class, rack_size=rack_sz)
        # for i in range(50):
        for i in range(25):
            scheduler.state_level = i
            goal = scheduler.gen_goal()
            # print(goal)
            # print("*"*10)
            # print(scheduler.gen_state(goal))
            state = scheduler.gen_state(goal)
            rp = RackStatePlot(goal_pattern=goal)
            img = rp.plot_states(rack_states=[state], row=1).get_img()
            cv2.imshow("img", img)
            cv2.waitKey(0)
        print("Success Finished")

    def test_scheduler3(self):
        num_class = 3
        rack_sz = (5, 10)

        scheduler = GoalRackStateScheduler3(num_classes=num_class, rack_size=rack_sz)
        # for i in range(50):
        for i in range(25):
            scheduler.set_training_level(i)
            for _ in range(10):
                goal = scheduler.gen_goal()
                # print(goal)
                # print("*"*10)
                # print(scheduler.gen_state(goal))
                state = scheduler.gen_state(goal)
                rp = RackStatePlot(goal_pattern=goal)
                img = rp.plot_states(rack_states=[state], row=1).get_img()
                cv2.imshow("img", img)
                cv2.waitKey(0)
        print("Success Finished")

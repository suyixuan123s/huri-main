""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231031osaka

"""

if __name__ == '__main__':
    from huri.learning.env.rack_v3.env import to_action, RackState, RackStatePlot, RackArrangementEnv
    import huri.core.file_sys as fs
    import cv2
    from huri.learning.method.APEX_DQN.distributed_fixed_goal.test.test_sequence_reward import main

    failed_path = fs.load_pickle(r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\run\data\debug_failed_path.pkl')
    # failed_path = fs.load_pickle(
    #     r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\run\data\debug_failed_path.pkl')

    # failed_path = fs.load_pickle(
        # r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\debug_data\debug_failed_path4.pkl')


    print(len(failed_path))
    for p, g in failed_path:
        # print('s:\n', f'np.{repr(s.state)}')
        # print('g:\n', f'np.{repr(g.state)}')
        # rp = RackStatePlot(goal_pattern=g)
        # img = rp.plot_states(rack_states=p, row=8, img_scale=1.8).get_img()
        # cv2.imshow(f"plot_1{0}", img)
        # cv2.waitKey(0)
        main(p, g)

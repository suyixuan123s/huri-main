""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231031osaka

"""

if __name__ == '__main__':
    from huri.learning.env.rack_v3.env import to_action, RackState, RackStatePlot, RackArrangementEnv
    import huri.core.file_sys as fs
    import cv2

    failed_path = fs.load_pickle(r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed\run\data\debug_failed_data.pkl')
    print(len(failed_path))
    for s, g in failed_path:
        print('s:\n', f'np.{repr(s.state)}')
        print('g:\n', f'np.{repr(g.state)}')
        rp = RackStatePlot(goal_pattern=g)
        img = rp.plot_states(rack_states=[s], row=2).get_img()
        cv2.imshow(f"plot_1{0}", img)
        cv2.waitKey(0)


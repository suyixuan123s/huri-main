""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20231129osaka

"""
import numpy as np
from huri.learning.method.APEX_DQN.distributed_fixed_goal.env import create_fixe_env

def get_env(goal_pattern):
    rack_size = goal_pattern.shape
    num_class = len(np.unique(goal_pattern[goal_pattern > 0]))
    env = create_fixe_env(rack_sz=rack_size,
                          goal_pattern=goal_pattern,
                          num_tube_class=num_class,
                          num_history=1,
                          seed=np.random.randint(0, 1000000),
                          toggle_curriculum=True)
    return env


if __name__ == '__main__':
    from huri.components.gui.pattern_swap import PatternSelectGui

    import visualization.panda.world as wd
    import huri.core.file_sys as fs



    seq = fs.load_pickle(
        r'E:\huri_shared\huri\learning\method\APEX_DQN\distributed_fixed_goal\test\debug_data\debug_failed_path3.pkl')

    base = wd.World(cam_pos=(1, -0.6, 0.7), lookat_pos=[.2, 0, .1], w=0,
                    h=0, )

    seq_start_id = 0
    s, g = seq[seq_start_id]
    env = get_env(g)

    psg = PatternSelectGui(base, init_arrangement=s[0], goal_pattern=g, env=env)

    base.run()

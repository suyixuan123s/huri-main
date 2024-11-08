"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20231128osaka

"""

if __name__ == '__main__':
    from huri.learning.method.APEX_DQN.distributed_fixed_goal.pipeline import EpsilonScheduler

    a = EpsilonScheduler(1, 0.15, decay_rate=1e-3)
    for i in range(1000):
        print(a.step())
    a.half_reset()
    print(a.step())

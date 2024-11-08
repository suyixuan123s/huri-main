import time

from huri.learning.env.rack_v3.env import RackState
import numpy as np

for _ in range(5000):
    t = np.random.randint(0, 3, (5, 10), dtype=int)
    c = RackState(t).feasible_action_set

print(len(RackState._cache))
t = np.random.randint(0, 3, (5, 10), dtype=int)
s_1 = time.time()
c = RackState(t).feasible_action_set
s_2 = time.time()
d = RackState(t).feasible_action_set
s_3 = time.time()

print(np.all(c == d))
print(s_2 - s_1, s_3 - s_2)

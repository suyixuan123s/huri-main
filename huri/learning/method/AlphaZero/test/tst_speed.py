""" 

Author: Hao Chen (chen960216@gmail.com)
Created: 20230704osaka

"""
from typing import List


def calc_acc_rets_no_numba(returns: List[float], gamma: float) -> float:
    acc_ret = 0
    for i, ret in enumerate(returns):
        acc_ret += ret * (gamma ** i)
    return acc_ret


if __name__ == '__main__':
    from huri.learning.method.AlphaZero.pipeline import calc_acc_rets
    import time

    r = [1, 2, 3, 4, 456, 6, 54, 65, 7, 56, 76, 8, 76, 8, 2, 34, 23, 423, 4, 543, 56, 54, 767, 56, 78, 8,
         234,45,4,654,6,567,6,8,7,9,8,0,345,43,5,345,34,5,34,53,45,34,5,2]
    gamma = .99

    s = time.time()
    for i in range(1000000):
        calc_acc_rets(
            r,
            gamma
        )
    e = time.time()
    print(f"Time consumption with numba {e - s}")

    s = time.time()
    for i in range(1000000):
        calc_acc_rets_no_numba(
            r,
            gamma
        )
    e = time.time()
    print(f"Time consumption {e - s}")
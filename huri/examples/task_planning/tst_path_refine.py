import copy

from numpy import array
import numpy as np

p = [array([[1, 0, 0, 1, 0, 0, 0, 0, 3, 0],
            [0, 2, 2, 0, 0, 2, 0, 2, 0, 2],
            [2, 0, 0, 3, 0, 2, 3, 0, 3, 0],
            [3, 1, 0, 0, 3, 0, 3, 2, 1, 1],
            [0, 3, 1, 3, 3, 2, 0, 1, 0, 0]]), array([[1, 0, 0, 1, 0, 2, 0, 0, 3, 0],
                                                     [0, 0, 2, 0, 0, 2, 0, 2, 0, 2],
                                                     [2, 0, 0, 3, 0, 2, 3, 0, 3, 0],
                                                     [3, 1, 0, 0, 3, 0, 3, 2, 1, 1],
                                                     [0, 3, 1, 3, 3, 2, 0, 1, 0, 0]]),
     array([[1, 1, 0, 0, 0, 2, 0, 0, 3, 0],
            [0, 0, 2, 0, 0, 2, 0, 2, 0, 2],
            [2, 0, 0, 3, 0, 2, 3, 0, 3, 0],
            [3, 1, 0, 0, 3, 0, 3, 2, 1, 1],
            [0, 3, 1, 3, 3, 2, 0, 1, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 0, 0, 3, 0],
                                                     [0, 0, 2, 0, 0, 2, 0, 2, 0, 2],
                                                     [2, 0, 0, 3, 0, 2, 3, 0, 3, 0],
                                                     [3, 1, 0, 0, 3, 0, 3, 2, 1, 0],
                                                     [0, 3, 1, 3, 3, 2, 0, 1, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 0, 0, 3, 0],
            [1, 0, 2, 0, 0, 2, 0, 2, 0, 2],
            [2, 0, 0, 3, 0, 2, 3, 0, 3, 0],
            [3, 1, 0, 0, 3, 0, 3, 2, 1, 0],
            [0, 3, 1, 3, 3, 2, 0, 0, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 0],
                                                     [1, 0, 0, 0, 0, 2, 0, 2, 0, 2],
                                                     [2, 0, 0, 3, 0, 2, 3, 0, 3, 0],
                                                     [3, 1, 0, 0, 3, 0, 3, 2, 1, 0],
                                                     [0, 3, 1, 3, 3, 2, 0, 0, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 0],
            [1, 0, 0, 0, 0, 2, 0, 2, 0, 2],
            [2, 0, 1, 3, 0, 2, 3, 0, 3, 0],
            [3, 1, 0, 0, 3, 0, 3, 2, 0, 0],
            [0, 3, 1, 3, 3, 2, 0, 0, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 0],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 2],
                                                     [2, 0, 1, 3, 0, 2, 3, 0, 3, 0],
                                                     [3, 1, 0, 0, 3, 0, 3, 2, 0, 0],
                                                     [2, 3, 1, 3, 3, 2, 0, 0, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 2],
            [2, 0, 1, 0, 0, 2, 3, 0, 3, 0],
            [3, 1, 0, 0, 3, 0, 3, 2, 0, 0],
            [2, 3, 1, 3, 3, 2, 0, 0, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 0],
                                                     [2, 0, 1, 2, 0, 2, 3, 0, 3, 0],
                                                     [3, 1, 0, 0, 3, 0, 3, 2, 0, 0],
                                                     [2, 3, 1, 3, 3, 2, 0, 0, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 3, 0, 0],
            [2, 0, 1, 2, 0, 2, 3, 0, 0, 0],
            [3, 1, 0, 0, 3, 0, 3, 2, 0, 0],
            [2, 3, 1, 3, 3, 2, 0, 0, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
                                                     [2, 0, 1, 2, 0, 2, 3, 0, 0, 0],
                                                     [3, 1, 0, 0, 0, 0, 3, 2, 0, 0],
                                                     [2, 3, 1, 3, 3, 2, 0, 0, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
            [2, 0, 1, 2, 0, 2, 3, 0, 0, 3],
            [3, 1, 0, 0, 0, 0, 3, 2, 0, 0],
            [2, 3, 1, 3, 0, 2, 0, 0, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
                                                     [2, 0, 1, 2, 0, 2, 3, 0, 0, 3],
                                                     [3, 1, 0, 0, 0, 0, 3, 0, 0, 0],
                                                     [2, 3, 1, 3, 2, 2, 0, 0, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
            [0, 0, 1, 2, 0, 2, 3, 0, 0, 3],
            [3, 1, 0, 0, 0, 0, 3, 0, 0, 0],
            [2, 3, 1, 3, 2, 2, 2, 0, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
                                                     [0, 0, 1, 2, 0, 2, 3, 0, 0, 3],
                                                     [0, 1, 0, 0, 0, 0, 3, 0, 0, 3],
                                                     [2, 3, 1, 3, 2, 2, 2, 0, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
            [1, 0, 1, 2, 0, 2, 3, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 3, 0, 0, 3],
            [2, 3, 1, 3, 2, 2, 2, 0, 0, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
                                                     [1, 0, 1, 2, 0, 2, 3, 0, 0, 3],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [2, 3, 1, 3, 2, 2, 2, 3, 0, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
            [1, 0, 1, 2, 0, 2, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [2, 3, 1, 3, 2, 2, 2, 3, 3, 0]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [2, 3, 1, 3, 2, 2, 2, 3, 3, 0]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [2, 0, 1, 3, 2, 2, 2, 3, 3, 3]]), array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 3, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [2, 0, 1, 3, 2, 2, 2, 3, 3, 3]]),
     array([[1, 1, 1, 0, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 3],
            [2, 0, 1, 3, 2, 2, 2, 3, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [0, 3, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 3, 2, 2, 2, 3, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 3, 1, 3, 2, 2, 2, 3, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 3, 1, 3, 2, 2, 2, 3, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 3, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 3, 2, 2, 2, 3, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 3, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 3, 2, 2, 2, 0, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
            [3, 3, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 3, 2, 2, 2, 0, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 3, 2, 2, 2, 3, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 3, 1, 2, 0, 0, 2, 0, 0, 3],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 3, 2, 2, 2, 0, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 3, 1, 2, 0, 0, 2, 0, 3, 3],
                                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 0, 2, 2, 2, 0, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 3, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 3, 3],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 0, 2, 2, 2, 0, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 3, 3],
                                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 3, 2, 2, 2, 0, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 0, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
            [3, 0, 0, 3, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 3, 2, 2, 2, 0, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 3, 2, 2, 2, 0, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
            [1, 0, 0, 3, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 3, 2, 2, 2, 0, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 0],
                                                     [3, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 3, 1, 3, 2, 2, 2, 0, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 3, 1, 3, 2, 2, 2, 3, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
                                                     [1, 0, 0, 3, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 3, 2, 2, 2, 3, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 3, 2, 2, 2, 3, 3, 3]]), array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
                                                     [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
                                                     [1, 0, 1, 2, 0, 0, 2, 0, 0, 3],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
                                                     [0, 0, 1, 3, 2, 2, 2, 3, 3, 3]]),
     array([[1, 1, 1, 2, 0, 2, 2, 3, 3, 3],
            [1, 0, 0, 0, 0, 2, 0, 0, 0, 3],
            [1, 0, 1, 2, 0, 0, 2, 0, 3, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 1, 0, 2, 2, 2, 3, 3, 3]])]

p = [array([[1, 0, 0, 3, 0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
            [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
            [0, 0, 1, 2, 0, 2, 0, 0, 3, 1]]), array([[1, 0, 0, 3, 0, 2, 0, 0, 2, 0],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                     [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
                                                     [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
                                                     [1, 0, 1, 2, 0, 2, 0, 0, 3, 0]]),
     array([[1, 0, 0, 0, 0, 2, 0, 0, 2, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
            [0, 1, 0, 0, 0, 0, 0, 2, 0, 0],
            [1, 0, 1, 2, 0, 2, 0, 0, 3, 0]]), array([[1, 0, 0, 2, 0, 2, 0, 0, 2, 3],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                     [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
                                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                     [1, 0, 1, 2, 0, 2, 0, 0, 3, 0]]),
     array([[1, 0, 0, 2, 0, 2, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 3, 0, 2, 2, 0, 0, 3, 3],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 2, 0, 2, 2, 0, 3, 0]]), array([[1, 0, 0, 2, 0, 2, 0, 3, 0, 3],
                                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                     [2, 0, 0, 0, 2, 2, 0, 0, 3, 3],
                                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                     [1, 0, 1, 2, 0, 2, 2, 0, 3, 0]]),
     array([[1, 0, 0, 2, 0, 2, 0, 3, 0, 3],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 2, 0, 0, 3, 3],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 2, 2, 2, 2, 0, 3, 0]])]

p = [array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
     array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
     array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0, 0, 0, 0, 0]])]
goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                         [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])


def get_suspicious_action_id(path, goal_pattern):
    path_len = len(path)
    suspicious_id = []
    for state_id in range(0, path_len - 1):
        s_i, s_ii = path[state_id], path[state_id + 1]
        move_map = s_ii - s_i
        move_to_idx = np.where(move_map > 0)
        move_from_idx = np.where(move_map < 0)
        is_move_to_pattern = goal_pattern[move_to_idx] == s_ii[move_to_idx]
        is_move_in_pattern = goal_pattern[move_to_idx] == goal_pattern[move_from_idx] and goal_pattern[move_to_idx] == \
                             s_ii[move_to_idx]
        if (not is_move_to_pattern) or is_move_in_pattern:
            suspicious_id.append(state_id + 1)
    return suspicious_id


def refine_immediately_redundant_action(path, goal_pattern):
    is_continue_iter = True
    path_r = copy.deepcopy(path)
    while is_continue_iter:
        is_continue_iter = False
        check_list = get_suspicious_action_id(path_r, goal_pattern)
        while len(check_list) > 0:
            check_id = check_list[0]
            if len(path_r) < 3:
                break
            s_0, s_1, s_2 = path_r[check_id - 1], path_r[check_id], path_r[check_id + 1]
            move_map = s_2 - s_0
            if len(np.where(move_map != 0)[0]) <= 2:
                is_continue_iter = True
                # remove the redundant move
                path_r.pop(check_id)
                # update the suspicious action id
                for _ in range(1, len(check_list)):
                    check_list[_] = check_list[_] - 1
            check_list.pop(0)
    return path_r


sa = get_suspicious_action_id(p, goal_pattern)
print(sa, len(sa))

path_r = refine_immediately_redundant_action(p, goal_pattern)

# from huri.learning.env.arrangement_planning_rack.env import RackArrangementEnv, RackStateVisualizer
# drawer = RackStateVisualizer(goal_pattern)
# drawer.plot_states(p, row=22)
# drawer.plot_states(path_r, row=22)
# print("a")
# sa = get_suspicious_action_id(path_r, goal_pattern)
# print(sa, len(sa))
print(len(p))
print(len(path_r))
print(len(get_suspicious_action_id(path_r, goal_pattern)))


def to_action(pick_id, place_id, rack_size):
    pick_id_int = pick_id[0] * rack_size[1] + pick_id[1]
    place_id_int = place_id[0] * rack_size[1] + place_id[1]
    return pick_id_int * np.prod(rack_size) + place_id_int


check_list = get_suspicious_action_id(path_r, goal_pattern)
is_continue_iter = True
from huri.examples.task_planning.a_star import TubePuzzle

while len(check_list) > 0:
    check_id = check_list[0] - 1
    end_id = len(path_r) - 1
    s_i, s_i_end = path_r[check_id], path_r[end_id]
    tp = TubePuzzle(s_i.copy())
    tp.goalpattern = s_i_end.copy()
    is_finished, refined_tmp_path = tp.atarSearch(max_iter_cnt=50)
    if is_finished and len(refined_tmp_path) < end_id - check_id:
        path_r = path_r[:check_id] + [_.grid for _ in refined_tmp_path]
        break
    check_list.pop(0)

# def test():
#     cnt = 1
#     while is_continue_iter:
#         is_continue_iter = False
#         end = None
#         # check_list = get_suspicious_action_id(path_r, goal_pattern)
#         while len(check_list) > 0:
#             check_id = check_list[-1]
#             s_i, s_ii = path_r[check_id - 1], path_r[check_id]
#             move_map = s_ii - s_i
#             move_to_idx = np.where(move_map > 0)
#             move_from_idx = np.where(move_map < 0)
#             if end is None:
#                 path_r_tmp = path_r[check_id + 1:].copy()
#                 end_part = []
#             else:
#                 path_r_tmp = path_r[check_id + 1:end].copy()
#                 end_part = path_r[end:]
#
#             path_r_valid = []
#             s_old = s_i
#             no_update = False
#             print(f"-------{check_id}")
#             for _id, _ in enumerate(path_r_tmp):
#                 print(_[move_from_idx], s_i[move_from_idx])
#                 # if _[move_to_idx]
#                 if _[move_from_idx] != 0:
#                     state_move = _ - s_old
#                     num_moves = len(np.where(state_move != 0)[0])
#                     if num_moves == 2:
#                         # if len(path_r_tmp)-1:
#                         pass
#                     elif num_moves == 0:
#                         continue
#                     else:
#                         print("EROR")
#                 else:
#                     _[move_to_idx], _[move_from_idx] = s_i[move_to_idx], s_i[move_from_idx]
#                 _move_map = _ - s_old
#                 _move_to_idx = np.where(_move_map > 0)
#                 _move_from_idx = np.where(_move_map < 0)
#                 if to_action(_move_to_idx, _move_from_idx, rack_size=_.shape) in RackState(s_old).feasible_action_set:
#                     s_old = _
#                     path_r_valid.append(_)
#                 else:
#                     end = check_id + 1 + _id
#                     no_update = True
#                     break
#
#             # print(path_r_valid)
#             if not no_update:
#                 path_r[check_id:] = path_r_valid + end_part
#             if cnt == 2:
#                 break
#             cnt += 1
#             print(cnt)
#             most_similar_id = check_id + 3 + np.argmin([len(_[_ != 0]) for _ in (path_r[check_id + 3:] - s_i)])
#             # print([len(_[_ != 0]) for _ in (path_r[check_id + 3:] - s_i)])
#             # s_i_end = path_r[most_similar_id]
#             # tp = TubePuzzle(s_i.copy())
#             # tp.goalpattern = s_i_end.copy()
#             # is_finished, refined_tmp_path = tp.atarSearch(max_iter_cnt=50)
#             # if is_finished and len(refined_tmp_path) < most_similar_id - check_id:
#             #     path_r = path_r[:check_id] + [_.grid for _ in refined_tmp_path] + path_r[most_similar_id + 1:]
#             #     is_continue_iter = True
#             check_list.pop(-1)


print("---")
print(len(get_suspicious_action_id(path_r, goal_pattern)))
print(len(path_r))
from huri.learning.env.arrangement_planning_rack.env import RackStatePlot

print(path_r)

drawer = RackStatePlot(goal_pattern)
# drawer.plot_states(p, row=22)
drawer.plot_states(path_r, row=22)
# #

from huri.components.task_planning.tube_puzzle_learning_solver import refine_redundant_action

drawer.plot_states(refine_redundant_action(p, goal_pattern), row=22)

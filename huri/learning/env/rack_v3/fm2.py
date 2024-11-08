import numpy as np
import basis.robot_math as rm

a = np.array([[0, 1, 0],
              [0, 0, 1],
              [0, 0, 1], ])

b = np.array([[1, 0, 0],
              [0, 1, 0],
              [1, 0, 0]])

b_1 = np.array([[0, 0, 1],
                [0, 0, 1],
                [0, 0, 1]])

c = np.array([[1, 0, 0],
              [1, 0, 0],
              [1, 0, 0]])

a_v = a.ravel()
b_v = b.ravel()
b_1_v = b_1.ravel()
c_v = c.ravel()

print(a_v, b_v, b_1_v, c_v)

angle_1 = rm.angle_between_vectors(a_v, c_v)
angle_2 = rm.angle_between_vectors(b_v, c_v)
angle_2_1 = rm.angle_between_vectors(b_1_v, c_v)
angle_3 = rm.angle_between_vectors(b_v, c_v)

print(angle_1, angle_2, angle_2_1)

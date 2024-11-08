import numpy as np
import basis.robot_math as rm
from numbers import Number
from typing import List


def distance_to_line(n, p, a=np.array([0, 0, 0])):
    """
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_vector_formulation
    p: points
    n: unit vector: direction of the line
    a: a point on the line
    """
    return np.linalg.norm(np.cross(a - p, n), axis=1) / np.linalg.norm(n)


def distance_to_line2(n, p, a=np.array([0, 0, 0])):
    """
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Another_vector_formulation
    p: points
    n: unit vector: direction of the line
    a: a point on the line
    """
    return np.linalg.norm((p - a) - np.outer(((p - a).dot(n)), n), axis=1)


def gen_line(alpha=0, beta=0):
    rot = rm.rotmat_from_axangle([1, 0, 0], alpha)
    rot = np.dot(rm.rotmat_from_axangle(rot[:, 1], beta), rot)
    vec = np.dot(rot, np.array([0, 0, 1]))
    return vec, rot


def combination(c_list: List[np.ndarray]) -> np.ndarray:
    c_list_len = len(c_list)
    comb = np.array(np.meshgrid(*c_list)).T.reshape(-1, c_list_len)
    return comb


def perpendicular_vector(v: np.ndarray):
    r""" Finds an arbitrary perpendicular vector to *v*."""
    # for two vectors (x, y, z) and (a, b, c) to be perpendicular,
    # the following equation has to be fulfilled
    #     0 = ax + by + cz

    # x = y = z = 0 is not an acceptable solution
    if v[0] == v[1] == v[2] == 0:
        raise ValueError('zero-vector')

    # If one dimension is zero, this can be solved by setting that to
    # non-zero and the others to zero. Example: (4, 2, 0) lies in the
    # x-y-Plane, so (0, 0, 1) is orthogonal to the plane.
    if v[0] == 0:
        return np.array(1, 0, 0)
    if v[1] == 0:
        return np.array(0, 1, 0)
    if v[2] == 0:
        return np.array(0, 0, 1)

    # arbitrarily set a = b = 1
    # then the equation simplifies to
    #     c = -(x + y)/z
    return rm.unit_vector(np.array([1, 1, -1.0 * (v[0] + v[1]) / v[2]]))

if __name__ == "__main__":
    line_vec, _ = gen_line(np.radians(45))

    points = np.array([[0, 1, 0],
                       [0, 2, 0],
                       [0, 3, 0],
                       [0, 0, 1],
                       [0, -1, 0]
                       ])
    print(line_vec)

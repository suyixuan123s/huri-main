"""

Author: Hao Chen (chen960216@gmail.com)
Created: 20220811osaka

"""
from pathlib import Path
from typing import Literal

import numpy as np
from . import pytracik_bindings as pytracik_bindings


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.

    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4,))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / np.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                      [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                      [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                      [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


class TracIK(object):
    def __init__(self, base_link_name: str,
                 tip_link_name: str,
                 urdf_path: str,
                 timeout: float = .005,
                 epsilon: float = 1e-5,
                 solver_type: Literal['Speed', 'Distance', 'Manip1', 'Manip2'] = "Speed"):
        """
        Create a TRAC_IK instance and keep track of it.
        This file is made changes from the original trac_ik.py file in the https://bitbucket.org/traclabs/trac_ik/src/master/trac_ik_python/src/trac_ik_python/trac_ik.py

        :param str base_link: Starting link of the chain.
        :param str tip_link: Last link of the chain.
        :param float timeout: Timeout in seconds for the IK calls.
        :param float epsilon: Error epsilon.
        :param solve_type str: Type of solver, can be:
            Speed (default), Distance, Manipulation1, Manipulation2
        :param urdf_string str: Optional arg, if not given URDF is taken from
            the param server at /robot_description.
        """
        if solver_type == "Speed":
            _solve_type = pytracik_bindings.SolveType.Speed
        elif solver_type == "Distance":
            _solve_type = pytracik_bindings.SolveType.Distance
        elif solver_type == "Manip1":
            _solve_type = pytracik_bindings.SolveType.Manip1
        elif solver_type == "Manip2":
            _solve_type = pytracik_bindings.SolveType.Manip2
        else:
            raise ValueError(f"Unsupported solver type: {solver_type}")

        urdf_path = Path(urdf_path)
        if urdf_path.exists():
            urdf_string = urdf_path.read_text()
        else:
            raise ValueError(f"{urdf_path} is not exist")
        self._urdf_string = urdf_string
        self._timeout = timeout
        self._epsilon = epsilon
        self._solve_type = _solve_type
        self.base_link_name = base_link_name
        self.tip_link_name = tip_link_name
        self._ik_solver = pytracik_bindings.TRAC_IK(self.base_link_name,
                                                    self.tip_link_name,
                                                    self._urdf_string,
                                                    self._timeout,
                                                    self._epsilon,
                                                    self._solve_type)

    @property
    def dof(self) -> int:
        """
        Get the number of joints in the chain.
        :return:  Number of joints in the chain.
        """
        return pytracik_bindings.get_num_joints(self._ik_solver, self.base_link_name, self.tip_link_name)

    def ik(self,
           tgt_pos: np.ndarray,
           tgt_rot: np.ndarray,
           seed_jnt_values: np.ndarray) -> None or np.ndarray:
        """
        Solve the IK.
        :param tgt_pos: 1x3 target position
        :param tgt_rot: 3x3 target rotation matrix
        :param seed_jnt_values: 1xN seed joint values
        :return: None if no solution is found, otherwise 1xN joint values
        """

        qw, qx, qy, qz = quaternion_from_matrix(tgt_rot)
        r = pytracik_bindings.ik(self._ik_solver, seed_jnt_values.astype(float), float(tgt_pos[0]),
                                 float(tgt_pos[1]), float(tgt_pos[2]), float(qx), float(qy), float(qz), float(qw))
        # r = pytracik_bindings.ik(self._ik_solver, seed_jnt_values[0], seed_jnt_values[1], seed_jnt_values[2],
        #                          seed_jnt_values[3], seed_jnt_values[4], seed_jnt_values[5], seed_jnt_values[6],
        #                          float(tgt_pos[0]),
        #                          float(tgt_pos[1]), float(tgt_pos[2]), float(qx), float(qy), float(qz), float(qw))
        # r = np.asarray(r)
        # print(r)
        succ = r[0] >= 0
        if succ:
            # if not np.allclose(self.fk(r[1:])[0], np.array([tgt_pos]), atol=1e-5):
            #     print("IK 异常", np.linalg.norm(self.fk(r[1:])[0] - np.array([tgt_pos])), self.fk(r[1:])[0],
            #           np.array([tgt_pos]))
            #     print("r", r)
            #     print(repr(tgt_pos), repr(tgt_rot), seed_jnt_values)
            #     return None
            # else:
            return r[1:]
        else:
            return None

    def fk(self, q: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Forward kinematics.
        :param q: 1xN joint values
        :return: position and rotation matrix of the tip link
        """
        assert isinstance(q, np.ndarray), f"q must be a numpy array, not {type(q)}"
        assert q.shape == (self.dof,), f"q must be a 1x{self.dof} array, not {q.shape}"
        homomat = pytracik_bindings.fk(self._ik_solver, q)
        return homomat[:3, 3], homomat[:3, :3]

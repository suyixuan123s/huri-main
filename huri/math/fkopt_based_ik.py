from motion.optimization_based.fkopt_based_ik import FKOptBasedIK
import basis.robot_math as rm
import time
from scipy.optimize import minimize


class FKOptBasedIK_C(FKOptBasedIK):
    def _constraint_yangle(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rotmat = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        delta_angle = rm.angle_between_vectors(gl_tcp_rotmat[:, 1], self.tgt_rotmat[:, 0])
        self.xangle_err.append(delta_angle)
        return self._xangle_limit - delta_angle

    def solve(self, tgt_pos, tgt_rotmat, seed_jnt_values, method='SLSQP'):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param method:
        :return:
        """
        self.seed_jnt_values = seed_jnt_values
        self.tgt_pos = tgt_pos
        self.tgt_rotmat = tgt_rotmat
        self.cons = []
        if tgt_rotmat is not None:
            self.add_constraint(self._constraint_xangle, type="ineq")
            # self.add_constraint(self._constraint_yangle, type="ineq")
            # self.add_constraint(self._constraint_zangle, type="ineq")
        self.add_constraint(self._constraint_x, type="ineq")
        self.add_constraint(self._constraint_y, type="ineq")
        self.add_constraint(self._constraint_z, type="ineq")
        self.add_constraint(self._constraint_collision, type="ineq")
        time_start = time.time()
        sol = minimize(self.optimization_goal,
                       seed_jnt_values,
                       method=method,
                       bounds=self.bnds,
                       constraints=self.cons)

        print("time cost", time.time() - time_start)
        if self.toggle_debug:
            print(sol)
            self._debug_plot()
        if sol.success:
            return sol.x, sol.fun
        else:
            return None, None

    def _constraint_x_2(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        x_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
        self.x_err.append(x_err)
        return (.1 - x_err)

    # def _constraint_x_3(self, jnt_values):
    #     self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
    #     gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
    #     x_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
    #     self.x_err.append(x_err)
    #     return (x_err - .02)

    # def _constraint_y_2(self, jnt_values):
    #     self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
    #     gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
    #     x_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
    #     self.x_err.append(x_err)
    #     return 1e-2 - x_err

    def _constraint_y_2(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        y_err = abs(self.tgt_pos[1] - gl_tcp_pos[1])
        self.y_err.append(y_err)
        return (.1 - y_err)

    # def _constraint_y_3(self, jnt_values):
    #     self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
    #     gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
    #     y_err = abs(self.tgt_pos[1] - gl_tcp_pos[1])
    #     self.y_err.append(y_err)
    #     return (y_err - .02)

    def _constraint_z_2(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        z_err = abs(.035 - gl_tcp_pos[2])
        self.z_err.append(z_err)
        return (1e-4 - z_err)

    # def _c_x(self, jnt_values):
    #     self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
    #     gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
    #     return ( gl_tcp_pos[0]-self.tgt_pos[0] + 0.03) / 1e8

    # def _constraint_v(self, jnt_values):
    #     self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
    #     gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
    #     v = gl_tcp_pos
    #     v[2] = .04
    #     ii = self.rbt.ik(self.jlc_name, tgt_pos=v, tgt_rotmat=gl_tcp_rot, seed_jnt_values=jnt_values)
    #     if ii is None:
    #         return -1
    #     else:
    #         self.rbt.fk(jnt_values=ii, component_name=self.jlc_name)
    #         if self.rbt.is_collided(obstacle_list=self.obstacle_list):
    #             return -1
    #         else:
    #             return 1

    def solve2(self, tgt_pos, seed_jnt_values, large_cons=False, method='SLSQP'):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param method:
        :return:
        """
        self.seed_jnt_values = seed_jnt_values
        self.tgt_pos = tgt_pos
        # if True:
        #     self.add_constraint(self._c_x, type="ineq")
        #     print("?")
        self.add_constraint(self._constraint_x_2, type="ineq")
        self.add_constraint(self._constraint_y_2, type="ineq")
        # self.add_constraint(self._constraint_x_3, type="ineq")
        # self.add_constraint(self._constraint_y_3, type="ineq")
        self.add_constraint(self._constraint_z_2, type="ineq")
        self.add_constraint(self._constraint_collision, type="ineq")
        # self.add_constraint(self._constraint_v, type="ineq")
        time_start = time.time()
        sol = minimize(self.optimization_goal,
                       seed_jnt_values,
                       method=method,
                       bounds=self.bnds,
                       constraints=self.cons,
                       options={"maxiter": 400})
        print("time cost", time.time() - time_start)

        print(sol)
        # self._debug_plot()
        if sol.success:
            return sol.x, sol.fun
        else:
            return None, None

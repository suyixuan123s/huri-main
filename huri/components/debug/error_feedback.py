from huri.math.fkopt_based_ik import FKOptBasedIK
import time
from scipy.optimize import minimize


class RecaptureFeedbackMoveSim(FKOptBasedIK):
    def __init__(self, robot_s, component_name, obstacle_list=[], toggle_debug=False):
        super(RecaptureFeedbackMoveSim, self).__init__(robot_s, component_name, obstacle_list, toggle_debug)

    def fk_s(self, component_name, jnt_values):
        self.rbt.fk(component_name, jnt_values)

    def set_obs_list(self, obs_list):
        self.obstacle_list = obs_list

    def _constraint_x(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        x_err = abs(self.tgt_pos[0] - gl_tcp_pos[0])
        self.x_err.append(x_err)
        return self._x_limit - x_err

    def _constraint_y(self, jnt_values):
        self.rbt.fk(jnt_values=jnt_values, component_name=self.jlc_name)
        gl_tcp_pos, gl_tcp_rot = self.rbt.get_gl_tcp(manipulator_name=self.jlc_name)
        y_err = abs(self.tgt_pos[1] - gl_tcp_pos[1])
        self.y_err.append(y_err)
        return self._y_limit - y_err

    def solve(self, tgt_pos, tgt_rotmat, seed_jnt_values=None, method='SLSQP'):
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
        # self.add_constraint(self._constraint_xangle, type="ineq")
        # self.add_constraint(self._constraint_zangle, type="ineq")
        self.add_constraint(self._constraint_x, type="ineq")
        self.add_constraint(self._constraint_y, type="ineq")
        # self.add_constraint(self._constraint_z, type="ineq")
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

if __name__ == "__main__":
    RecaptureFeedbackSim
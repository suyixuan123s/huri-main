import numpy as np

from huri.core.common_import import ym, rm, gm
from modeling.model_collection import ModelCollection
from huri.learning.env.env_meta import Gym_Proto, spaces
from huri.test.iksolver.ik_offline_utils import sample_workspace


class YuMiEnv(Gym_Proto):
    def __init__(self,
                 arm_name="rgt_arm",
                 action_seed=777,
                 is_render=False):
        super(YuMiEnv, self).__init__()

        # rbt
        self._yumi = ym.Yumi(enable_cc=True)
        self._yumi_arm_name = arm_name
        if arm_name == "rgt_arm":
            self._yumi_arm = self._yumi.rgt_arm
        elif arm_name == "lft_arm":
            self._yumi_arm = self._yumi.lft_arm
        else:
            raise

        self.ndof = self._yumi_arm.ndof

        arm_jnt_limit_low = self._yumi_arm.jlc.jnt_ranges[:, 0]
        arm_jnt_limit_high = self._yumi_arm.jlc.jnt_ranges[:, 1]
        pos_space_low = np.array([0.15, -0.4, 0.03])
        pos_space_high = np.array([.65, 0.4, 0.4])
        ang_space_low = np.array([-np.pi, -np.pi, -np.pi])
        ang_space_high = np.array([np.pi, np.pi, np.pi])

        #
        self.observation_space = spaces.Box(np.concatenate((arm_jnt_limit_low, pos_space_low, ang_space_low)),
                                            np.concatenate((arm_jnt_limit_high, pos_space_high, ang_space_high)))
        self.action_space = spaces.Box(low=arm_jnt_limit_low, high=arm_jnt_limit_high, dtype=np.float32)

        # sampled workspace
        self.pos_sample, self.ang_sample = self.sample_workspace(
            pos_space_range=[[pos_space_low[0], pos_space_high[0]], [pos_space_low[1], pos_space_high[1]],
                             [pos_space_low[2], pos_space_high[2]]],
            rot_space_range=(
                (ang_space_low[0], ang_space_high[0]), (ang_space_low[1], ang_space_high[1]),
                (ang_space_low[2], ang_space_high[2])),
            pos_sample_dense=.01,
            rot_sample_dense=np.pi / 18)

        self.state = None

        # render
        self.is_render = is_render
        self.rbt_mdl: ModelCollection = None
        self.tgt_pt_mdl: gm.GeometricModel = None

    def sample_workspace(self,
                         pos_space_range,
                         rot_space_range,
                         pos_sample_dense=(.1, .1, .1),
                         rot_sample_dense=(np.pi / 36, np.pi / 36, np.pi / 36)):

        if isinstance(pos_sample_dense, float):
            pos_sample_dense = (pos_sample_dense, pos_sample_dense, pos_sample_dense)

        if isinstance(rot_sample_dense, float):
            rot_sample_dense = (rot_sample_dense, rot_sample_dense, rot_sample_dense)

        pos_space_range[0][1] += pos_sample_dense[0]
        pos_space_range[1][1] += pos_sample_dense[1]
        pos_space_range[2][1] += pos_sample_dense[2]

        x = np.arange(*pos_space_range[0], pos_sample_dense[0])
        y = np.arange(*pos_space_range[1], pos_sample_dense[1])
        z = np.arange(*pos_space_range[2], pos_sample_dense[2])

        pos_sampled = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

        pitch = np.arange(*rot_space_range[0], rot_sample_dense[0])
        roll = np.arange(*rot_space_range[1], rot_sample_dense[1])
        yaw = np.arange(*rot_space_range[2], rot_sample_dense[2])

        ang_sampled = np.array(np.meshgrid(roll, yaw, pitch)).T.reshape(-1, 3)

        return pos_sampled, ang_sampled

    def reset(self):
        arm_jnts = self.rbt_rnd_conf()
        goal_pos = self.random_goal_pos()
        self.state = np.concatenate((arm_jnts, goal_pos))
        if self.is_render:
            if self.rbt_mdl is not None:
                self.rbt_mdl.remove()
            if self.tgt_pt_mdl is not None:
                self.tgt_pt_mdl.remove()
        return self.state

    def sample(self):
        return self.action_space.sample()

    def step(self, action):
        if self.state is None:
            raise

        tgt_pos = self.state[self.ndof:]
        nxt_state = np.concatenate((action, tgt_pos)).reshape(-1)

        self.state = nxt_state

        if self._yumi.is_collided():
            reward, done = -20, True
        else:
            reward, done = self.get_reward(nxt_state)
        info = {}
        return nxt_state, reward, done, info

    def get_reward(self, state):
        arm_jnts = state[:self.ndof]
        tgt_loc = state[self.ndof:]
        tgt_pos = tgt_loc[:3]
        tgt_rpy = tgt_loc[3:]

        self._yumi.fk(self._yumi_arm_name, arm_jnts)
        tcp_pos, tcp_rot = self._yumi.get_gl_tcp(self._yumi_arm_name)

        err = np.zeros(6)
        err[0:3] = (tgt_pos - tcp_pos)
        err[3:6] = rm.deltaw_between_rotmat(rm.rotmat_from_euler(tgt_rpy[0], tgt_rpy[1], tgt_rpy[2]), tcp_rot.T)
        errnorm = err.T.dot(err)
        accuracy = 1e-6
        if errnorm <= 1:
            return 50 * (1 / errnorm * accuracy), False
        else:
            return -errnorm, False

    def rbt_rnd_conf(self):
        arm_name = self._yumi_arm_name
        while True:
            rnd_conf = self._yumi.rand_conf(arm_name)
            self._yumi.fk(arm_name, rnd_conf)
            if not self._yumi.is_collided():
                break
        arm_jnts = rnd_conf.reshape(-1)
        return arm_jnts

    def random_goal_pos(self):
        while True:
            return np.concatenate((self.pos_sample[np.random.randint(len(self.pos_sample))],
                                   self.ang_sample[np.random.randint(len(self.ang_sample))])).reshape(-1)

    def obs_2_tgt_homomat(self, obs=None):
        if obs is None:
            obs = self.state
        tgt_pos = obs[self.ndof:]
        result = np.eye(4)
        result[:3, :3] = rm.rotmat_from_euler(tgt_pos[3], tgt_pos[4], tgt_pos[5])
        result[:3, 3] = tgt_pos[:3]
        return result

    def render_current_state(self):
        if self.state is None:
            raise
        arm_jnts = self.state[:self.ndof]
        tgt_pos = self.state[self.ndof:]

        self._yumi.fk(self._yumi_arm_name, arm_jnts)

        return self._yumi.gen_meshmodel(), \
               gm.gen_frame(tgt_pos[:3], rm.rotmat_from_euler(tgt_pos[3], tgt_pos[4], tgt_pos[5]))

    def render(self, mode, **kwargs):
        if self.rbt_mdl is not None:
            self.rbt_mdl.remove()
        if self.tgt_pt_mdl is not None:
            self.tgt_pt_mdl.remove()
        self.rbt_mdl, self.tgt_pt_mdl = self.render_current_state()
        self.rbt_mdl.attach_to(base)
        self.tgt_pt_mdl.attach_to(base)
        base.graphicsEngine.renderFrame()
        base.graphicsEngine.renderFrame()


if __name__ == "__main__":
    from huri.core.common_import import wd, gm

    env = YuMiEnv()
    state = env.reset()
    rbt_mdl, tgt_pt_mdl = env.render_current_state()

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_pointcloud(env.pos_sample).attach_to(base)

    rbt_mdl.attach_to(base)
    tgt_pt_mdl.attach_to(base)
    print(state)
    action = (env.sample())
    print(action)
    print(env.step(action=action))
    print(env.state)
    base.run()

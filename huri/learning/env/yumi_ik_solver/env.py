import numpy as np

from huri.core.common_import import ym, rm
from huri.learning.env.env_meta import Gym_Proto, spaces
from huri.test.iksolver.ik_offline_utils import sample_workspace
import torch


def perm_gpu(pop_size, num_samples):
    """
    Use torch.randperm to generate indices on a GPU tensor.
        https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/16
    """

    return torch.randperm(pop_size, device='cuda')[:num_samples]


class YuMiEnv(Gym_Proto):
    def __init__(self,
                 arm_name="rgt_arm",
                 action_seed=777, ):
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
        #
        # # setup action space
        # self.action_space =
        #
        # # set the seed for the env, ! action space and observation space should set seed as well
        # self.action_sapce.seed(action_seed)
        # self.seed(seed)

        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def sample_workspace(self,
                         pos_space_range,
                         rot_space_range,
                         pos_sample_dense=(.1, .1, .1),
                         rot_sample_dense=(np.pi / 36, np.pi / 36, np.pi / 36)):
        device = self.device

        if isinstance(pos_sample_dense, float):
            pos_sample_dense = (pos_sample_dense, pos_sample_dense, pos_sample_dense)

        if isinstance(rot_sample_dense, float):
            rot_sample_dense = (rot_sample_dense, rot_sample_dense, rot_sample_dense)

        pos_space_range[0][1] += pos_sample_dense[0]
        pos_space_range[1][1] += pos_sample_dense[1]
        pos_space_range[2][1] += pos_sample_dense[2]
        x = torch.arange(*pos_space_range[0], pos_sample_dense[0], device=device)
        y = torch.arange(*pos_space_range[1], pos_sample_dense[1], device=device)
        z = torch.arange(*pos_space_range[2], pos_sample_dense[2], device=device)

        pos_sampled = torch.cartesian_prod(x, y, z)

        pitch = torch.arange(*rot_space_range[0], rot_sample_dense[0], device=device)
        roll = torch.arange(*rot_space_range[1], rot_sample_dense[1], device=device)
        yaw = torch.arange(*rot_space_range[2], rot_sample_dense[2], device=device)

        ang_sampled = torch.cartesian_prod(roll, yaw, pitch)

        return pos_sampled, ang_sampled

    def reset(self):
        arm_jnts = self.rbt_rnd_conf()
        goal_pos = self.random_goal_pos()
        self.state = torch.cat((arm_jnts, goal_pos))
        return self.state

    def sample(self, state=None):
        if state is None:
            state = self.state
        return self.state

    def rbt_rnd_conf(self):
        arm_name = self._yumi_arm_name
        while True:
            rnd_conf = self._yumi.rand_conf(arm_name)
            self._yumi.fk(arm_name, rnd_conf)
            if not self._yumi.is_collided():
                break
        arm_jnts = torch.from_numpy(rnd_conf).to(self.device).view(-1)
        return arm_jnts

    def random_goal_pos(self):

        return torch.cat((self.pos_sample[torch.randint(len(self.pos_sample), (1,))],
                          self.ang_sample[torch.randint(len(self.ang_sample), (1,))]), 1).view(-1)

    def render_current_state(self, ndof=7):
        if self.state is None:
            raise
        arm_jnts = self.state[:ndof].cpu().numpy()
        tgt_pos = self.state[ndof:].cpu().numpy()

        self._yumi.fk(self._yumi_arm_name, arm_jnts)

        return self._yumi.gen_meshmodel(), \
               gm.gen_frame(tgt_pos[:3], rm.rotmat_from_euler(arm_jnts[3], arm_jnts[4], arm_jnts[5]))


if __name__ == "__main__":
    from huri.core.common_import import wd, gm

    env = YuMiEnv()
    state = env.reset()
    rbt_mdl, tgt_pt_mdl = env.render_current_state()

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_pointcloud(env.pos_sample.cpu().numpy()).attach_to(base)

    rbt_mdl.attach_to(base)
    tgt_pt_mdl.attach_to(base)
    base.run()

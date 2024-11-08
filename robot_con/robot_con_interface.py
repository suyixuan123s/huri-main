class RobotConInterface(object):
    def move_j(self, jnt_val, speed, *args, **kwargs):
        raise NotImplementedError

    def move_p(self, pos, rot, speed, *args, **kwargs):
        raise NotImplementedError

    def move_jntspace_path(self, path, *args, **kwargs):
        raise NotImplementedError

    def get_pose(self, *args, **kwargs):
        raise NotImplementedError

    def get_jnt_values(self, *args, **kwargs):
        raise NotImplementedError
import copy
import numpy as np
import modeling.model_collection as mc
import modeling.collision_model as cm
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim._kinematics.collision_checker as cc
import os


class XYZPlatform(object):

    def __init__(self, pos=np.array([0.137 + .15 + .2, 0.0043, -.18]), rotmat=np.eye(3),
                 homeconf=np.array([0.3545, .2105, 0]), cdmesh_type='aabb',
                 name='xyz_platform'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        self.cdmesh_type = cdmesh_type  # aabb, convexhull, or triangles
        # joints
        pri_jnt_safemargin = .0
        rel_jnt_safemargin = np.pi / 18.0
        this_dir, this_filename = os.path.split(__file__)

        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        self.jlc.jnts[1]['loc_pos'] = np.array([0, -0.3905 + 0.028, 0.048])
        self.jlc.jnts[1]['motion_rng'] = [.0 + pri_jnt_safemargin, .709 - pri_jnt_safemargin]
        self.jlc.jnts[1]['type'] = "prismatic"
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[2]['loc_pos'] = np.array([-0.237, 0.055, 0.058])
        self.jlc.jnts[2]['type'] = "prismatic"
        self.jlc.jnts[2]['motion_rng'] = [.0 + pri_jnt_safemargin, .421 - pri_jnt_safemargin]
        self.jlc.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.jlc.jnts[3]['loc_pos'] = np.array([0.04, 0, 0.036])
        self.jlc.jnts[3]['motion_rng'] = [0, 2 * 3.1415926535 - rel_jnt_safemargin]

        # links
        self.jlc.lnks[0]['name'] = 'platform_base'
        self.jlc.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "platform_base.stl")
        self.jlc.lnks[0]['rgba'] = np.array([0.3372549, 0.45882353, 0.50980392, 1])
        self.jlc.lnks[1]['name'] = "y_platform"
        self.jlc.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "x_platform.stl")
        self.jlc.lnks[1]['rgba'] = np.array([0.2745098, 0.30196078, 0.36078431, 1])
        self.jlc.lnks[2]['name'] = "x_platform"
        self.jlc.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "z_platform.stl")
        self.jlc.lnks[2]['rgba'] = np.array([0.96862745, 0.86666667, 0.64313725, 1])
        self.jlc.lnks[3]['name'] = "z_platform"
        self.jlc.lnks[3]['meshfile'] = os.path.join(this_dir, "meshes", "rotate_platform.stl")
        self.jlc.lnks[3]['rgba'] = np.array([0.72156863, 0.56470588, 0.40784314, 1])

        self.jlc.reinitialize()
        # collision detection
        # if enable_cc:
        #     self.enable_cc()
        # collision detection
        self.cc = None
        # cd mesh collection for precise collision checking
        self.cdmesh_collection = mc.ModelCollection()
        self.platform_jnt_dict = {
            "x": self.jlc.jnts[2],
            "y": self.jlc.jnts[1],
            "z": self.jlc.jnts[3],
        }
        # the order is x, y
        self.platform_jnt_list = [self.jlc.jnts[2], self.jlc.jnts[1], self.jlc.jnts[3], ]

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        return_val = self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        return return_val

    def is_mesh_collided(self, objcm_list=[], toggle_debug=False):
        for i, cdelement in enumerate(self.all_cdelements):
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            self.cdmesh_collection.cm_list[i].set_pos(pos)
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
            iscollided, collided_points = self.cdmesh_collection.cm_list[i].is_mcdwith(objcm_list, True)
            if iscollided:
                if toggle_debug:
                    print(self.cdmesh_collection.cm_list[i].get_homomat())
                    self.cdmesh_collection.cm_list[i].show_cdmesh()
                    for objcm in objcm_list:
                        objcm.show_cdmesh()
                    for point in collided_points:
                        import modeling.geometric_model as gm
                        gm.gen_sphere(point, radius=.001).attach_to(base)
                    print("collided")
                return True
        return False

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def fk(self, component_name, motion_val=np.zeros(3)):
        assert component_name in ["x", "y", "z", "all"]
        if component_name != "all":
            jlc_jnts = self.platform_jnt_dict[component_name]
            if jlc_jnts['motion_rng'][0] <= motion_val <= jlc_jnts['motion_rng'][1]:
                jlc_jnts['motion_val'] = motion_val
                self.jlc.fk()
            else:
                raise ValueError("The motion_val parameter is out of range!")
        else:
            assert len(motion_val) == len(self.platform_jnt_list)
            for _ind, _ in enumerate(motion_val):
                jlc_jnts = self.platform_jnt_list[_ind]
                if jlc_jnts['motion_rng'][0] <= _ <= jlc_jnts['motion_rng'][1]:
                    jlc_jnts['motion_val'] = _
                    self.jlc.fk()
                else:
                    print(f"{jlc_jnts['motion_rng'][0]} <= {_} <= {jlc_jnts['motion_rng'][1]}")
                    raise ValueError("The motion_val parameter is out of range!")

    def get_jnt_values(self, component_name):
        assert component_name in ["x", "y", "z", "all"]
        if component_name == "all":
            jnts = self.jlc.get_jnt_values()
            return jnts
        else:
            return_val = self.platform_jnt_dict[component_name]["motion_val"]
            return return_val

    def show_cdprimit(self):
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def show_cdmesh(self):
        for i, cdelement in enumerate(self.cc.all_cdelements):
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            self.cdmesh_collection.cm_list[i].set_pos(pos)
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
        self.cdmesh_collection.show_cdmesh()

    def unshow_cdmesh(self):
        self.cdmesh_collection.unshow_cdmesh()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi_gripper_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xyz_platform'):
        return self.jlc._mt.gen_meshmodel(tcp_jntid=tcp_jntid,
                                          tcp_loc_pos=tcp_loc_pos,
                                          tcp_loc_rotmat=tcp_loc_rotmat,
                                          toggle_tcpcs=toggle_tcpcs,
                                          toggle_jntscs=toggle_jntscs,
                                          rgba=rgba,
                                          name=name)

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy

    def rand_conf(self, component_name, region=None):
        """
        random configuration for the xyz platform
        :param region: the region should be either None or a np.array
                        if the region is None, it does not specify the region of random configuration
                        if the region is specified:
                            1. the component_name is "x" or "y" or "z",
                               region has to be np.array([lowerbound, upperbound])
                            2. the component_name is "all", region has to be
                                np.array([lowerbound_x, upperbound_x],
                                         [lowerbound_y, upperbound_y],
                                         [lowerbound_z, upperbound_z])
        """

        def _gen_rand_conf(_jnt, _region):
            if _region is None:
                _region = _jnt['motion_rng']
            return np.random.uniform(_region[0], _region[1])

        if component_name in self.platform_jnt_dict:
            assert (isinstance(region, np.ndarray) and len(region) == 2) or region is None
            jnt = self.platform_jnt_dict[component_name]
            return _gen_rand_conf(jnt, region)
        elif component_name == "all":
            assert (isinstance(region, np.ndarray) and region.shape == (3, 2)) or region is None
            if region is None:
                rand_conf = self.jlc.rand_conf()
                return np.asarray([rand_conf[1], rand_conf[0], rand_conf[2]])
            else:
                return np.array([
                    _gen_rand_conf(self.platform_jnt_dict["x"], region[0]),
                    _gen_rand_conf(self.platform_jnt_dict["y"], region[1]),
                    _gen_rand_conf(self.platform_jnt_dict["z"], region[2])
                ])
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    a = XYZPlatform()
    a.fk(component_name="all", motion_val=np.array([0, 0, 0]))
    a.gen_meshmodel(toggle_jntscs=False, toggle_tcpcs=False).attach_to(base)
    base.run()

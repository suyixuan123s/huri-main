import modeling.collision_model as cm
import modeling.geometric_model as gm
from shapely.geometry.point import Point
from basis.trimesh.creation import extrude_polygon
from huri.core.common_import import fs
import basis.robot_math as rm
import numpy as np
import huri.vision.pnt_utils as ptu
import vision.depth_camera.util_functions as dcuf
from huri.components.vision.extract import oriented_box_icp


def circle_pcd(radius=.1):
    p = Point(0, 0)
    circle = p.buffer(radius)
    calibration_tool = cm.CollisionModel(extrude_polygon(circle, .1 / 1000))
    pcd = calibration_tool.sample_surface(1 / 1000, nsample=20000)[0]
    return pcd


c2z = 60.00175 / 1000


class CalibrationTool(object):

    def __init__(self, rbt, rbtx, template='resources/calibration_tool_pcd.pkl', toggle_debug=False):
        self._pcd_temp = fs.load_pickle(fs.workdir.joinpath('work2', template))
        self._rbt = rbt
        self._rbtx = rbtx
        self._toggle_debug = toggle_debug
        if not self._toggle_debug:
            self._rbtx.move_p(pos=np.array([.25, 0, .22]), rot=np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T)

    def locate_tooltip(self, pcd, toggle_debug=False):
        mat = self.sync()
        if mat is None:
            return None
        pos, rot = mat[:3, 3], mat[:3, :3]
        ind = ptu.extract_pcd_by_range(pcd,
                                       x_range=[-60 / 1000, 60 / 1000],
                                       y_range=[-60 / 1000, 60 / 1000],
                                       z_range=[c2z - 10 / 1000, c2z + 10 / 1000],
                                       origin_pos=pos,
                                       origin_rot=rot, )
        pcd_tool = pcd[ind]
        if toggle_debug:
            gm.gen_pointcloud(pcd_tool, rgbas=[[1, 0, 0, 1]], pntsize=10).attach_to(base)
        downsampling_voxelsize = .004
        pcd_inl, _ = dcuf.remove_outlier(src_nparray=pcd_tool,
                                         downsampling_voxelsize=downsampling_voxelsize,
                                         nb_points=10,
                                         radius=downsampling_voxelsize * 1.5)
        init_homo = np.eye(4)
        init_homo[:3, :3] = rot
        init_homo[:3, 3] = pos + init_homo[:3, 2] * c2z
        transform = oriented_box_icp(pcd=pcd_inl,
                                     pcd_template=self._pcd_temp,
                                     init_homo=init_homo,
                                     downsampling_voxelsize=.002,
                                     toggle_debug=False,
                                     toggle_remove_pcd_statistical=False,
                                     maximum_distance=.03)
        tooltip_real = transform.copy()
        tooltip_real[:3, 3] - tooltip_real[:3, 2] * c2z
        if toggle_debug:
            gm.gen_pointcloud(pcd_inl, rgbas=[[1, 1, 0, 1]], pntsize=10).attach_to(base)
            gm.gen_frame(init_homo[:3, 3], init_homo[:3, :3]).attach_to(base)
            gm.gen_frame(transform[:3, 3], transform[:3, :3]).attach_to(base)
            gm.gen_frame(tooltip_real[:3, 3], tooltip_real[:3, :3]).attach_to(base)
        return tooltip_real

    def sync(self):
        if not self._toggle_debug:
            jnt_val = self._rbtx.get_jnt_values()
            self._rbt.fk("arm", jnt_val)
            pos, rot = self._rbt.get_gl_tcp('arm')
            return rm.homomat_from_posrot(pos, rot)
        return None


def generate_pcd_template(base=None):
    radius = 25 / 1000
    pcd = circle_pcd(radius=radius)
    fs.dump_pickle(pcd, fs.workdir.joinpath('work2', "resources/calibration_tool_pcd.pkl"))
    if base is not None:
        gm.gen_pointcloud(pcd).attach_to(base)


def detect_tooltip():
    from robot_sim.robots.cobotta.cobotta_ripps import CobottaRIPPS
    import robot_con.cobotta.cobotta_x as cbtx
    from huri.work2.dual_d405_manager import RealSenseD405Dual
    from huri.work2.obs_rack import Rack_Locator

    robot_s = CobottaRIPPS()
    robot_x = cbtx.CobottaX()
    robot_x.defult_gripper()
    sensor_hdl = RealSenseD405Dual()
    rl = Rack_Locator(robot_x, robot_s, sensor_hdl, )
    ct = CalibrationTool(rbt=robot_s, rbtx=robot_x, )

    # calibration
    pcd_rgba1, pcd_rgba2 = rl.get_pcd_im2()
    tooltip_mat = ct.locate_tooltip(pcd=pcd_rgba1, toggle_debug=True)
    if tooltip_mat is not None:
        sim_mat = ct.sync()
        real_mat = rm.homomat_from_posrot(robot_x.get_pose())
        gm.gen_frame(real_mat[:3, 3], real_mat[:3, :3]).attach_to(base)
        tcp2tooltip_mat = np.linalg.inv(real_mat).dot(tooltip_mat)
        print("TCP2TOOLTIP MAT")
        print(repr(tcp2tooltip_mat))


def pos2tcp(target_pos, target_rot, rbtx, tcp2tooltip_mat=np.eye(4)):
    tcp = np.linalg.inv(tcp2tooltip_mat).dot(rm.homomat_from_posrot(target_pos, target_rot))
    rbtx.move_p_nullspace(tcp[:3, 3], tcp[:3, :3], k=5)
    # pose = rbtx.wrshomomat2cobbotapos(wrshomomat=rm.homomat_from_posrot(target_pos, target_rot), k=5)
    # cobbota_pos, times = rbtx.null_space_search(pose)
    # print(cobbota_pos)
    # wrs_pos, wrs_rot = rbtx.cobottapos2wrspos(cobbota_pos)
    # rm.homomat_from_posrot(wrs_pos, wrs_rot).dot(tooltip_mat)


if __name__ == '__main__':
    import visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, 0])
    # generate_pcd_template(base)
    detect_tooltip()
    base.run()

import numpy as np
import basis.robot_math as rm
import scipy.optimize as sopt
import motion.optimization_based.incremental_nik as inik
import pickle
import logging, sys
import modeling.geometric_model as gm
from huri.math.ran_sphere import Sphere
import huri.vision.pnt_utils as pntu
from huri.components.vision.extract import oriented_box_icp_general
import basis.trimesh as tm
import huri.core.file_sys as fs
import modeling.collision_model as cm


def get_board_pos(robot_tcp_pos, robot_tcp_rotmat, affine_mat, pcd, offset):
    # Transform and Plot Point Clouds
    pcd = rm.homomat_transform_points(affine_mat, points=pcd)
    # gm.gen_pointcloud(pcd, [[1, 0, 0, .3]]).attach_to(base)

    pcd_in_hand = rm.homomat_transform_points(
        rm.homomat_inverse(rm.homomat_from_posrot(robot_tcp_pos, robot_tcp_rotmat)),
        pcd)
    idx = np.where((pcd_in_hand[:, 2] > -0.03) & (pcd_in_hand[:, 2] < 0.12)
                   & (pcd_in_hand[:, 0] > -0.01) & (pcd_in_hand[:, 0] < 0.01))
    pcd = pcd[idx] + offset

    board_pcd, outlier = pntu.segment_table_plane(pcd, distance_threshold=0.005, ransac_n=3)
    # gm.gen_pointcloud(pcd, [[1, 1, 0, .3]]).attach_to(base)
    # gm.gen_pointcloud(board_pcd, [[0, 1, 0, .3]]).attach_to(base)
    board_mdl = cm.CollisionModel(tm.primitives.Box(box_extents=[0.1, 0.15, 0.001]))
    board_mdl_pnts = board_mdl.sample_surface(radius=0.003)[0]
    transform = oriented_box_icp_general(board_pcd, board_mdl_pnts)

    board_mdl_pnts_trans = rm.homomat_transform_points(transform, points=board_mdl_pnts)
    gm.gen_pointcloud(board_mdl_pnts_trans, [[1, 1, 1, .3]]).attach_to(base)
    gm.gen_sphere(transform[:3, 3], radius=0.001).attach_to(base)
    return transform[:3, 3]


def load_calibration_data(file="./depth_sensor_calib_mat.pkl",
                          has_sensor_and_real_points=False):
    """
    :param file:
    :param has_sensor_and_real_points:
    :return:
    author: weiwei
    date: 20210519
    """
    if has_sensor_and_real_points:
        affine_mat, pos_in_real_array, pos_in_sensor_array = pickle.load(open(file, "rb"))
    else:
        affine_mat = pickle.load(open(file, "rb"))
        pos_in_real_array = None
        pos_in_sensor_array = None
    return affine_mat, pos_in_real_array, pos_in_sensor_array


class DepthCaliberator(object):

    def __init__(self, robot_x, robot_s, affine_mat_path):
        self.robot_x = robot_x
        self.robot_s = robot_s
        self.sphere = Sphere()
        self.affine_mat = np.asarray(
            fs.load_json(affine_mat_path)['affine_mat'])

    def _find_tcp_in_sensor(self, component_name, action_pos, action_rotmat, sensor_marker_handler, std_out=None):
        """
        find the robot_s tcp's pos and rotmat in the sensor coordinate system
        :param component_name:
        :param action_center_pos, action_rotmat:
        :param marker_callback:
        :return: [estiamted tcp center in sensor, radius of the sphere formed by markers]
        author: weiwei
        date: 20210408
        """

        def _fit_sphere(p, coords):
            x0, y0, z0, radius = p
            x, y, z = coords.T
            return np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2)

        _err_fit_sphere = lambda p, x: _fit_sphere(p, x) - p[3]

        marker_pos_in_sensor_list = []
        rot_range_x = [np.array([1, 0, 0]), [np.radians(angle)
                                             for angle in [-60, -15, -30, 0, 15, 30]]]
        rot_range_xy = [np.array([1, 1, 0]), [np.radians(angle)
                                              for angle in [-150, -120, -60, -30, 30, 60, 120, 150]]]
        rot_range_xz = [np.array([1, 0, 1]), [np.radians(angle)
                                              for angle in [-60, -30, 30]]]
        rot_range_y = [np.array([0, 1, 0]), [np.radians(angle)
                                             for angle in [-150, -120, -60, -30, 30, 60, 120, 150]]]
        # rot_range_yz = [np.array([0, 1, 1]), [ -np.pi / 3, -np.pi / 6, -np.pi / 13, np.pi / 13, np.pi / 6, np.pi/3]]
        rot_range_z = [np.array([0, 0, 1]), [-np.pi / 2 + i * np.pi / 6 for i in range(0, 12)]]
        range_axes = [rot_range_x, rot_range_xy, rot_range_xz, rot_range_z]
        # range_axes = [rot_range_x,rot_range_y]
        last_jnt_values = self.robot_x.get_jnt_values(component_name=component_name)
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        for axisid in range(len(range_axes)):
            axis = range_axes[axisid][0]
            for angle in range_axes[axisid][1]:
                print(f"-----------------{axisid}----------{np.rad2deg(angle)}")
                goal_pos = action_pos
                goal_rotmat = np.dot(rm.rotmat_from_axangle(axis, angle), action_rotmat)
                jnt_values = self.robot_s.ik(component_name=component_name,
                                             tgt_pos=goal_pos,
                                             tgt_rotmat=goal_rotmat,
                                             seed_jnt_values=last_jnt_values,
                                             max_niter=1000)
                if jnt_values is not None:
                    self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
                    if not self.robot_s.is_collided():
                        last_jnt_values = jnt_values
                        self.robot_x.move_jnts(component_name, jnt_values)

                        pcd = sensor_marker_handler.get_pcd()
                        jnts = self.robot_x.get_jnt_values(component_name)
                        print(jnts)
                        self.robot_s.fk(component_name=component_name, jnt_values=jnts)
                        robot_pos, robot_rot = self.robot_s.get_gl_tcp(component_name)
                        offset = action_pos - robot_pos
                        print(f"The real pos and action pos offset is: {offset}")
                        board_pos = get_board_pos(robot_tcp_pos=robot_pos, robot_tcp_rotmat=robot_rot,
                                                  affine_mat=self.affine_mat, pcd=pcd, offset=offset)

                        marker_pos_in_sensor = board_pos
                        print(marker_pos_in_sensor)

                        if marker_pos_in_sensor is not None:
                            marker_pos_in_sensor_list.append(marker_pos_in_sensor)
                            if std_out is not None:
                                std_out.attach(gm.gen_sphere(pos=marker_pos_in_sensor, radius=0.001))
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        if len(marker_pos_in_sensor_list) < 3:
            return [None, None]
        center_in_camera_coords_array = np.asarray(marker_pos_in_sensor_list)
        center_in_camera_coords_array = rm.homomat_transform_points(np.linalg.inv(self.affine_mat),
                                                                    center_in_camera_coords_array)
        for i in center_in_camera_coords_array:
            gm.gen_sphere(i, radius=0.001).attach_to(base)
        gm.gen_pointcloud(pcd, ).attach_to(base)

        # # try:
        # initial_guess = np.ones(4) * .001
        # initial_guess[:3] = np.mean(center_in_camera_coords_array, axis=0)
        # final_estimate, flag = sopt.leastsq(_err_fit_sphere, initial_guess, args=(center_in_camera_coords_array,))
        # print(final_estimate)
        # if len(final_estimate) == 0:
        #     return [None, None]
        # return np.array(final_estimate[:3]), final_estimate[3]
        center, radius, inliers = self.sphere.fit(center_in_camera_coords_array, thresh=0.001)
        # print(inliers)
        # for i in inliers:
        #     gm.gen_sphere(pos=np.array(i), radius=0.003).attach_to(base)
        # print(center)
        # print(radius)
        return np.array(center), np.array(radius)

    def find_board_center_in_hand(self,
                                  component_name,
                                  sensor_marker_handler,
                                  action_center_pos=np.array([.3, -.05, .2]),
                                  action_center_rotmat=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T,
                                  action_dist=.1,
                                  std_out=None):
        """
        :param component_name:
        :param action_center_pos:
        :param action_center_rotmat:
        :param action_dist:
        :return:
        author: weiwei
        date: 20210408, 20210519
        """
        tcp_in_sensor, radius_by_markers = self._find_tcp_in_sensor(component_name=component_name,
                                                                    action_pos=action_center_pos,
                                                                    action_rotmat=action_center_rotmat,
                                                                    sensor_marker_handler=sensor_marker_handler,
                                                                    std_out=std_out)
        print("estimated center is", tcp_in_sensor)
        print("estimated radius is", radius_by_markers)
        sphere = gm.gen_sphere(pos=tcp_in_sensor, radius=radius_by_markers, rgba=[1, 0, 0, .3])
        std_out.attach(node=sphere, name="estimated sphere")
        gm.gen_sphere(pos=tcp_in_sensor, radius=0.003, rgba=[1, 0, 0, .3]).attach_to(base)
        base.run()
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        # move to action pos, action rotmat
        last_jnt_values = self.robot_x.get_jnt_values(component_name=component_name)
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_pos,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=1000)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            # acquire marker_pos
            pcd = sensor_marker_handler.get_pcd()
            jnts = self.robot_x.get_jnt_values(component_name)
            print(jnts)
            self.robot_s.fk(component_name=component_name, jnt_values=jnts)
            robot_pos, robot_rot = self.robot_s.get_gl_tcp(component_name)
            offset = np.zeros(3)
            print(f"The real pos and action pos offset is: {offset}")
            board_pos = get_board_pos(robot_tcp_pos=robot_pos, robot_tcp_rotmat=robot_rot,
                                      affine_mat=self.affine_mat, pcd=pcd, offset=offset)
            board_pos = rm.homomat_transform_points(np.linalg.inv(self.affine_mat), board_pos)
            # acqurie finished
            marker_pos_in_sensor = board_pos

            rgt_jnt = self.robot_x.get_jnt_values(component_name=component_name)
            self.robot_s.fk(component_name=component_name, jnt_values=rgt_jnt)
        else:
            raise ValueError("The action center is not reachable. Try a different pos or robtmat!")
        # move to x+action_dist
        action_center_dist_x = action_center_pos + action_center_rotmat[:, 0] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_x,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=1000)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            # acquire marker_pos
            pcd = sensor_marker_handler.get_pcd()
            jnts = self.robot_x.get_jnt_values(component_name)
            print(jnts)
            self.robot_s.fk(component_name=component_name, jnt_values=jnts)
            robot_pos, robot_rot = self.robot_s.get_gl_tcp(component_name)
            offset = np.zeros(3)
            print(f"The real pos and action pos offset is: {offset}")
            board_pos = get_board_pos(robot_tcp_pos=robot_pos, robot_tcp_rotmat=robot_rot,
                                      affine_mat=self.affine_mat, pcd=pcd, offset=offset)
            board_pos = rm.homomat_transform_points(np.linalg.inv(self.affine_mat), board_pos)
            # acqurie finished
            marker_pos_xplus_in_sensor = board_pos

            rgt_jnt = self.robot_x.get_jnt_values(component_name=component_name)
            self.robot_s.fk(component_name=component_name, jnt_values=rgt_jnt)

        else:
            raise ValueError("The action center with xplus is not reachable. Try a different pos or robtmat!")
        # move to y+action_dist
        action_center_dist_y = action_center_pos + action_center_rotmat[:, 1] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_y,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=1000)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            # acquire marker_pos
            pcd = sensor_marker_handler.get_pcd()
            jnts = self.robot_x.get_jnt_values(component_name)
            print(jnts)
            self.robot_s.fk(component_name=component_name, jnt_values=jnts)
            robot_pos, robot_rot = self.robot_s.get_gl_tcp(component_name)
            offset = np.zeros(3)
            print(f"The real pos and action pos offset is: {offset}")
            board_pos = get_board_pos(robot_tcp_pos=robot_pos, robot_tcp_rotmat=robot_rot,
                                      affine_mat=self.affine_mat, pcd=pcd, offset=offset)
            board_pos = rm.homomat_transform_points(np.linalg.inv(self.affine_mat), board_pos)
            # acqurie finished
            marker_pos_yplus_in_sensor = board_pos

            rgt_jnt = self.robot_x.get_jnt_values(component_name=component_name)
            self.robot_s.fk(component_name=component_name, jnt_values=rgt_jnt)
            marker_pos_yplus_in_real, _ = self.robot_s.get_gl_tcp(manipulator_name=component_name)
        else:
            raise ValueError("The action center with yplus is not reachable. Try a different pos or robtmat!")
        # move to z+action_dist
        action_center_dist_z = action_center_pos + action_center_rotmat[:, 2] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_z,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=1000)
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            self.robot_x.move_jnts(component_name, jnt_values)
            # acquire marker_pos
            pcd = sensor_marker_handler.get_pcd()
            jnts = self.robot_x.get_jnt_values(component_name)
            print(jnts)
            self.robot_s.fk(component_name=component_name, jnt_values=jnts)
            robot_pos, robot_rot = self.robot_s.get_gl_tcp(component_name)
            offset = np.zeros(3)
            print(f"The real pos and action pos offset is: {offset}")
            board_pos = get_board_pos(robot_tcp_pos=robot_pos, robot_tcp_rotmat=robot_rot,
                                      affine_mat=self.affine_mat, pcd=pcd, offset=offset)
            board_pos = rm.homomat_transform_points(np.linalg.inv(self.affine_mat), board_pos)
            # acqurie finished
            marker_pos_zplus_in_sensor = board_pos

            rgt_jnt = self.robot_x.get_jnt_values(component_name=component_name)
            self.robot_s.fk(component_name=component_name, jnt_values=rgt_jnt)

        else:
            raise ValueError("The action center with zplus is not reachable. Try a different pos or robtmat!")
        print(
            "Markerpos in sensor", marker_pos_in_sensor,
            "Markerpos +x in sensor", marker_pos_xplus_in_sensor,
            "Markerpos +y in sensor", marker_pos_yplus_in_sensor,
            "Markerpos +z in sensor", marker_pos_zplus_in_sensor
        )
        unnormalized_marker_mat_in_sensor = np.array([marker_pos_xplus_in_sensor - marker_pos_in_sensor,
                                                      marker_pos_yplus_in_sensor - marker_pos_in_sensor,
                                                      marker_pos_zplus_in_sensor - marker_pos_in_sensor]).T
        marker_rotmat_in_sensor, r = np.linalg.qr(unnormalized_marker_mat_in_sensor)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_xplus_in_sensor, rgba=[1, 0, 0, 1],
                     thickness=0.001).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_yplus_in_sensor, rgba=[0, 1, 0, 1],
                     thickness=0.001).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_zplus_in_sensor, rgba=[0, 0, 1, 1],
                     thickness=0.001).attach_to(base)
        # gm.gen_frame(pos=marker_pos_in_sensor, rotmat=marker_rotmat_in_sensor).attach_to(base)
        gm.gen_arrow(spos=tcp_in_sensor, epos=marker_pos_in_sensor, thickness=0.001).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=2 * marker_pos_in_sensor - tcp_in_sensor,
                     thickness=0.001).attach_to(base)
        print(marker_pos_in_sensor - tcp_in_sensor)
        result = marker_pos_in_sensor - tcp_in_sensor
        unnormalized_marker_mat_in_sensor2 = np.array(
            [rm.unit_vector(marker_pos_xplus_in_sensor - marker_pos_in_sensor),
             rm.unit_vector(marker_pos_yplus_in_sensor - marker_pos_in_sensor),
             rm.unit_vector(marker_pos_zplus_in_sensor - marker_pos_in_sensor)]).T

        _result = np.dot(unnormalized_marker_mat_in_sensor2, result)
        marker_pos_in_hnd = np.dot(marker_rotmat_in_sensor.T, marker_pos_in_sensor - tcp_in_sensor)
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        print("marker in hand", marker_pos_in_hnd)
        # return marker_pos_in_hnd
        return _result, unnormalized_marker_mat_in_sensor2

    def calibrate(self,
                  component_name,
                  sensor_marker_handler,
                  marker_pos_in_hnd=None,
                  action_pos_list=[np.array([.3, -.2, .9]), np.array([.3, .2, .9]),
                                   np.array([.4, -.2, .9]), np.array([.4, .2, .9]),
                                   np.array([.3, -.2, 1.1]), np.array([.3, .2, 1.1]),
                                   np.array([.4, -.2, 1.1]), np.array([.4, .2, 1.1])],
                  action_rotmat_list=[np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T] * 8,
                  save_calib_file='depth_sensor_calib_mat.pkl',
                  save_sensor_and_real_points=False):
        """
        :param marker_pos_in_hnd:
        :param aruco_dict:
        :return:
        author: weiwei
        date: 20191228
        """
        if marker_pos_in_hnd is None:
            raise Exception
            # marker_pos_in_hnd = self.find_board_center_in_hand(component_name=component_name,
            #                                                    sensor_marker_handler=sensor_marker_handler,
            #                                                    action_center_pos=,
            #                                                    action_center_rotmat=,
            #                                                    action_dist=)
        pos_in_real_list = []
        pos_in_sensor_list = []
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        last_jnt_values = self.robot_x.get_jnt_values(component_name=component_name)
        for i, action_pos in enumerate(action_pos_list):
            jnt_values = self.robot_s.ik(component_name=component_name,
                                         tgt_pos=action_pos,
                                         tgt_rotmat=action_rotmat_list[i],
                                         seed_jnt_values=last_jnt_values, max_niter=1000)
            if jnt_values is not None:
                self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
                last_jnt_values = jnt_values
                if not self.robot_s.is_collided():
                    self.robot_x.move_jnts(component_name, jnt_values)
                    # acquire marker_pos
                    pcd = sensor_marker_handler.get_pcd()
                    jnts = self.robot_x.get_jnt_values(component_name)
                    print(jnts)
                    self.robot_s.fk(component_name=component_name, jnt_values=jnts)
                    robot_pos, robot_rot = self.robot_s.get_gl_tcp(component_name)
                    offset = np.zeros(3)
                    print(f"The real pos and action pos offset is: {offset}")
                    board_pos = get_board_pos(robot_tcp_pos=robot_pos, robot_tcp_rotmat=robot_rot,
                                              affine_mat=self.affine_mat, pcd=pcd, offset=offset)
                    board_pos = rm.homomat_transform_points(np.linalg.inv(self.affine_mat), board_pos)
                    # acqurie finished
                    marker_pos_in_sensor = board_pos
                    if marker_pos_in_sensor is not None:
                        rgt_jnt = self.robot_x.get_jnt_values(component_name=component_name)
                        self.robot_s.fk(component_name=component_name, jnt_values=rgt_jnt)
                        real_pos, _ = self.robot_s.get_gl_tcp(manipulator_name=component_name)
                        pos_in_real_list.append(real_pos + np.dot(action_rotmat_list[i], marker_pos_in_hnd))
                        gm.gen_sphere(real_pos + np.dot(action_rotmat_list[i], marker_pos_in_hnd),
                                      radius=0.003).attach_to(base)
                        pos_in_sensor_list.append(marker_pos_in_sensor)
                else:
                    print(f"The {i}th action pose is collided!")
            else:
                print(f"The {i}th action pose is reachable!")
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        pos_in_real_array = np.array(pos_in_real_list)
        pos_in_sensor_array = np.array(pos_in_sensor_list)
        affine_mat = rm.affine_matrix_from_points(pos_in_sensor_array.T, pos_in_real_array.T)
        if save_sensor_and_real_points:
            data = [affine_mat, pos_in_real_array, pos_in_sensor_array]
        else:
            data = affine_mat
        pickle.dump(data, open('./' + save_calib_file, "wb"))
        return affine_mat

    def refine_with_template(self, affine_mat, template_file):
        """
        refine the affine_mat by matching it with a template
        :param affine_mat:
        :param template_file:
        :return:
        author: weiwei
        date: 20191228, 20210519
        """
        pass

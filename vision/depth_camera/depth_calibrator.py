import numpy as np
import basis.robot_math as rm
import scipy.optimize as sopt
import motion.optimization_based.incremental_nik as inik
import pickle
import logging, sys
import modeling.geometric_model as gm
from huri.math.ran_sphere import Sphere
from huri.vision.phoxi_capture import SensorMarkerHandler
from robot_sim.robots.robot_interface import RobotInterface


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

    def __init__(self, robot_x: 'RobotConInterface', robot_s: RobotInterface):
        self.robot_x = robot_x
        self.robot_s = robot_s
        self.sphere = Sphere()

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
        tcp_pos_real_list = []

        rot_range_x = [np.array([1, 0, 0]), [np.radians(angle)
                                             for angle in [-30, -15, 0, 15, 30]]]
        rot_range_y = [np.array([0, 1, 0]), [np.radians(angle)
                                             for angle in [-45, -15, 15, 45, ]]]
        rot_range_mxy = [np.array([-1, 1, 0]), [np.radians(angle)
                                                for angle in [-45, -15, 15, 45]]]
        rot_range_z = [np.array([0, 0, 1]), [np.radians(angle)
                                             for angle in [-120, -90, -60, -30, 30, 90, 60, 120]]]
        range_axes = [rot_range_x, rot_range_y, rot_range_mxy, rot_range_z]
        last_jnt_values = self.robot_x.get_jnt_values(component_name=component_name)
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        for axisid in range(len(range_axes)):
            axis = range_axes[axisid][0]
            for angle in range_axes[axisid][1]:
                print(f"-----------------{axisid}----------{angle}")
                goal_pos = action_pos
                goal_rotmat = np.dot(rm.rotmat_from_axangle(axis, angle), action_rotmat)
                jnt_values = self.robot_s.ik(component_name=component_name,
                                             tgt_pos=goal_pos,
                                             tgt_rotmat=goal_rotmat,
                                             seed_jnt_values=last_jnt_values,
                                             max_niter=1000)
                if jnt_values is not None:
                    print(jnt_values)
                    self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
                    if not self.robot_s.is_collided():
                        last_jnt_values = jnt_values
                        conf = self.robot_x.fk(component_name, jnt_values, return_conf=True)[2]
                        # self.robot_x.move_jnts(component_name, jnt_values)
                        try:
                            self.robot_x.move_p(component_name, goal_pos, goal_rotmat, conf, linear=False, )
                        except Exception as e:
                            print(e)
                            continue

                        marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                        if marker_pos_in_sensor is not None:
                            marker_pos_in_sensor_list.append(marker_pos_in_sensor)
                            print("POSE: ", repr(self.robot_x.get_pose(component_name, return_conf=True)))
                            if std_out is not None:
                                std_out.attach(gm.gen_sphere(pos=marker_pos_in_sensor, radius=0.001))
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        if len(marker_pos_in_sensor_list) < 3:
            return [None, None]
        center_in_camera_coords_array = np.asarray(marker_pos_in_sensor_list)
        print(marker_pos_in_sensor_list)
        # try:
        initial_guess = np.ones(4) * .001
        initial_guess[:3] = np.mean(center_in_camera_coords_array, axis=0)
        final_estimate, flag = sopt.leastsq(_err_fit_sphere, initial_guess, args=(center_in_camera_coords_array,))
        print(final_estimate)
        if len(final_estimate) == 0:
            return [None, None]
        pcd = sensor_marker_handler.get_pcd()
        gm.gen_pointcloud(pcd, ).attach_to(base)
        center, radius, _ = self.sphere.fit(center_in_camera_coords_array, thresh=0.001, maxIteration=1000)

        print(f"Number of inliner {len(_)}/{len(center_in_camera_coords_array)} ")
        print("Randsac:", center, radius)
        print("Normal:", np.array(final_estimate[:3]), final_estimate[3])

        tcp_in_sensor, radius_by_markers = np.array(final_estimate[:3]), radius
        gm.gen_sphere(pos=tcp_in_sensor, radius=radius_by_markers, rgba=[0, 1, 0, .3]).attach_to(base)
        gm.gen_sphere(pos=center, radius=radius, rgba=[0, 1, 1, .3]).attach_to(base)

        return center, radius
        # return np.array(final_estimate[:3]), final_estimate[3]

        # print(inliers)
        # for i in inliers:
        #     gm.gen_sphere(pos=np.array(i), radius=0.003).attach_to(base)
        # print(center)
        # print(radius)

        # gm.gen_pointcloud(pcd, ).attach_to(base)
        # return np.array(center), np.array(radius)

    def find_board_center_in_hand(self,
                                  component_name: str,
                                  sensor_marker_handler: SensorMarkerHandler,
                                  action_center_pos: np.ndarray = np.array([.3, -.05, .2]),
                                  action_center_rotmat: np.ndarray = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T,
                                  action_dist: float = .1,
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
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        tcp_in_sensor, radius_by_markers = self._find_tcp_in_sensor(component_name=component_name,
                                                                    action_pos=action_center_pos,
                                                                    action_rotmat=action_center_rotmat,
                                                                    sensor_marker_handler=sensor_marker_handler,
                                                                    std_out=std_out)
        # print("tcp center", tcp_in_sensor, "radius", radius_by_markers)
        # sphere = gm.gen_sphere(pos=tcp_in_sensor, radius=0.01, rgba=[1, 0, 0, .3])
        # gm.gen_sphere(pos=tcp_in_sensor, radius=0.003, rgba=[1, 0, 0, .3]).attach_to(base)
        # std_out.attach(node=sphere, name="estimated sphere")
        # base.run()
        # move to action pos, action rotmat
        self.robot_x.move_jnts(component_name, jnt_values_bk)
        action_center_pos = action_center_pos.copy()
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_pos,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=jnt_values_bk, max_niter=1500)
        if jnt_values is not None:
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            if not self.robot_s.is_collided():
                conf = self.robot_x.fk(component_name, jnt_values, return_conf=True)[2]
                # self.robot_x.move_jnts(component_name, jnt_values)
                self.robot_x.move_p(component_name, action_center_pos, action_center_rotmat, conf, linear=False, )

                marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                # rgt_jnt = self.robot_x.get_jnt_values(component_name=component_name)
                # self.robot_s.fk(component_name=component_name, jnt_values=rgt_jnt)
                # real_pos, _ = self.robot_s.get_gl_tcp(manipulator_name=component_name)
                # offset = action_center_pos - real_pos

                # print(marker_pos_in_sensor)
                # print("offset", offset)
        else:
            raise ValueError("The action center is not reachable. Try a different pos or robtmat!")
        # move to x+action_dist
        action_center_dist_x = action_center_pos + action_center_rotmat[:, 0] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_x,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=jnt_values_bk, max_niter=1500)
        if jnt_values is not None:
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            if not self.robot_s.is_collided():
                conf = self.robot_x.fk(component_name, jnt_values, return_conf=True)[2]
                self.robot_x.move_p(component_name, action_center_dist_x, action_center_rotmat, conf, linear=False, )
                marker_pos_xplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with xplus is not reachable. Try a different pos or robtmat!")
        # move to y+action_dist
        action_center_dist_y = action_center_pos + action_center_rotmat[:, 1] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_y,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=jnt_values_bk, max_niter=1000)
        if jnt_values is not None:
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            if not self.robot_s.is_collided():
                conf = self.robot_x.fk(component_name, jnt_values, return_conf=True)[2]
                self.robot_x.move_p(component_name, action_center_dist_y, action_center_rotmat, conf)
            marker_pos_yplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with yplus is not reachable. Try a different pos or robtmat!")
        # move to z+action_dist
        action_center_dist_z = action_center_pos + action_center_rotmat[:, 2] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_z,
                                     tgt_rotmat=action_center_rotmat,
                                     seed_jnt_values=jnt_values_bk, max_niter=1000)
        if jnt_values is not None:
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            if not self.robot_s.is_collided():
                conf = self.robot_x.fk(component_name, jnt_values, return_conf=True)[2]
                self.robot_x.move_p(component_name, action_center_dist_z, action_center_rotmat, conf, linear=False, )
            marker_pos_zplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with zplus is not reachable. Try a different pos or robtmat!")
        print(
            "Markerpos in sensor", marker_pos_in_sensor,
            "Markerpos +x in sensor", marker_pos_xplus_in_sensor,
            "Markerpos +y in sensor", marker_pos_yplus_in_sensor,
            "Markerpos +z in sensor", marker_pos_zplus_in_sensor
        )
        unnormalized_marker_mat_in_sensor = np.array(
            [rm.unit_vector(marker_pos_xplus_in_sensor - marker_pos_in_sensor),
             rm.unit_vector(marker_pos_yplus_in_sensor - marker_pos_in_sensor),
             rm.unit_vector(marker_pos_zplus_in_sensor - marker_pos_in_sensor)]).T
        # marker_rotmat_in_sensor, r = np.linalg.qr(unnormalized_marker_mat_in_sensor)
        marker_pos_in_hnd = np.dot(unnormalized_marker_mat_in_sensor.T, marker_pos_in_sensor - tcp_in_sensor)

        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_xplus_in_sensor, rgba=[1, 0, 0, 1]).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_yplus_in_sensor, rgba=[0, 1, 0, 1]).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_zplus_in_sensor, thickness=.001,
                     rgba=[0, 0, 1, 1]).attach_to(base)
        # gm.gen_frame(pos=marker_pos_in_sensor, rotmat=marker_rotmat_in_sensor).attach_to(base)
        gm.gen_arrow(spos=tcp_in_sensor, epos=marker_pos_in_sensor).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=2 * marker_pos_in_sensor - tcp_in_sensor).attach_to(base)

        return marker_pos_in_hnd, unnormalized_marker_mat_in_sensor

    def calibrate(self,
                  component_name: str,
                  sensor_marker_handler: SensorMarkerHandler,
                  marker_pos_in_hnd=None,
                  action_pos_list=(np.array([.3, -.2, .9]), np.array([.3, .2, .9]),
                                   np.array([.4, -.2, .9]), np.array([.4, .2, .9]),
                                   np.array([.3, -.2, 1.1]), np.array([.3, .2, 1.1]),
                                   np.array([.4, -.2, 1.1]), np.array([.4, .2, 1.1])),
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
                    marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                    if marker_pos_in_sensor is not None:
                        real_pos, real_rot = self.robot_x.get_pose(component_name)
                        pos_in_real_list.append(real_pos + np.dot(real_rot, marker_pos_in_hnd))
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

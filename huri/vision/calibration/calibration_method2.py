import numpy as np
import pickle
from huri.math.ran_sphere import Sphere
import motion.probabilistic.rrt_connect as rrtc


class DepthCaliberator(object):

    def __init__(self, robot_x, robot_s, component_name, init_rotmat):
        self.robot_x = robot_x
        self.robot_s = robot_s
        self._component_name = component_name
        self._init_rotmat = init_rotmat
        self.rrtc_planner = rrtc.RRTConnect(robot_s)
        self.ransac_sphere_fitter = Sphere()
        self.reference_pos_in_cam = None

    def robotx_move(self, jnt_values):
        component_name = self._component_name
        current_jnts = self.robot_x.get_jnt_values(component_name=component_name)
        motions = self.rrtc_planner.plan(component_name=component_name,
                                         start_conf=current_jnts,
                                         goal_conf=jnt_values,
                                         obstacle_list=[],
                                         ext_dist=.05,
                                         max_time=300)
        if motions is not None:
            self.robot_x.move_jntspace_path(path=motions, component_name=component_name)
            return True
        else:
            return False

    def robotx_to_init(self):
        component_name = self._component_name
        if component_name in ["lft_arm", "lft_hnd"]:
            arm = self.robot_s.lft_arm
        elif component_name in ["rgt_arm", "rgt_hnd"]:
            arm = self.robot_s.rgt_arm
        else:
            raise ValueError("Component_name must be in ['lft_arm/lft_hnd', 'rgt_arm/rgt_hnd']!")
        current_jnts = self.robot_x.get_jnt_values(component_name=component_name)
        motions = self.rrtc_planner.plan(component_name=component_name,
                                         start_conf=current_jnts,
                                         goal_conf=arm.homeconf,
                                         obstacle_list=[],
                                         ext_dist=.05,
                                         max_time=300)
        self.robot_x.move_jntspace_path(path=motions, component_name=component_name)

    def sphere_sampler(self, sphere_center: np.ndarray, ):
        component_name = self._component_name
        init_rotmat = self._init_rotmat
        angle_range_with_zero = [-np.pi / 2, -np.pi / 3, -np.pi / 6, -np.pi / 12, 0, np.pi / 12, np.pi / 6, np.pi / 3,
                                 np.pi / 2]
        range_axes = [np.array([1, 0, 0]), np.array([1, 1, 0]), np.array([1, 0, 1]), np.array([0, 1, 0]),
                      np.array([0, 1, 1]), np.array([0, 0, 1])]

        last_jnt_values = self.robot_x.get_jnt_values(component_name=component_name)
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        goal_pos_ik_list = []
        for axisid in range(len(range_axes)):
            axis = range_axes[axisid]
            for angle_id, angle in enumerate(angle_range_with_zero):
                # print(f"-----------------{axisid}----------{angle_id}")
                goal_pos = sphere_center
                goal_rotmat = np.dot(rm.rotmat_from_axangle(axis, angle), init_rotmat)
                jnt_values = self.robot_s.ik(component_name=component_name,
                                             tgt_pos=goal_pos,
                                             tgt_rotmat=goal_rotmat,
                                             seed_jnt_values=last_jnt_values, max_niter=3000, local_minima="end")
                if jnt_values is not None and not self.robot_s.is_collided():
                    self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
                    # print(f"the tcp pos is:{self.robot_s.get_gl_tcp(component_name)[0]}")
                    last_jnt_values = jnt_values
                    goal_pos_ik_list.append(jnt_values)
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        return goal_pos_ik_list

    def get_offset_pos_in_cam(self, center_pos, off_set_in_world, sensor_marker_handler):
        offset_pos_in_cam = None
        component_name = self._component_name
        init_rotmat = self._init_rotmat
        last_jnt_values = self.robot_x.get_jnt_values(component_name=component_name)
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=center_pos + off_set_in_world,
                                     tgt_rotmat=init_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=3000, local_minima="end")

        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            self.robotx_move(jnt_values=jnt_values)
            marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
            if marker_pos_in_sensor is not None:
                if np.linalg.norm(off_set_in_world) == 0:
                    self.reference_pos_in_cam = marker_pos_in_sensor
                    offset_pos_in_cam = 0
                else:
                    if self.reference_pos_in_cam is None:
                        raise Exception("Please indicate the reference pos in cam first")
                    offset_pos_in_cam = marker_pos_in_sensor - self.reference_pos_in_cam
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        return offset_pos_in_cam

    def _find_tcp_in_sensor(self, reference_pos, sensor_marker_handler, rotmat_w_c, std_out=None):
        """
        find the robot_s tcp's pos and rotmat in the sensor coordinate system
        :param component_name:
        :param action_center_pos, action_rotmat:
        :param marker_callback:
        :return: [estiamted tcp center in sensor, radius of the sphere formed by markers]
        author: weiwei
        date: 20210408
        """

        marker_pos_in_sensor_list = []
        pos_offset_list = [np.array([0, 0, 0]), np.array([0, 0, .15]), np.array([0, 0, -.15]),
                           np.array([0, .15, 0]), np.array([0, -.15, 0]),
                           np.array([.15, 0, 0]), np.array([-.15, 0, 0])]
        # pos_offset_list = [np.array([0, 0, 0])]
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        for pos_offset in pos_offset_list:
            action_center = reference_pos + pos_offset
            action_groups = self.sphere_sampler(sphere_center=action_center)
            offset_pos = self.get_offset_pos_in_cam(center_pos=reference_pos,
                                                    off_set_in_world=pos_offset,
                                                    sensor_marker_handler=sensor_marker_handler)
            if offset_pos is None:
                continue
            for jnt_values in action_groups:
                if self.robotx_move(jnt_values=jnt_values):
                    self.robot_s.fk(component_name=component_name,
                                    jnt_values=self.robot_x.get_jnt_values(component_name=component_name))
                    print(f"the tcp of the robot is {self.robot_s.get_gl_tcp(manipulator_name=component_name)}",)
                    marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                    if marker_pos_in_sensor is not None:
                        print(f"the action center is {action_center}"
                              f"marker pos in sensorr is {marker_pos_in_sensor},pos offset is {offset_pos}, the norm of the offset {np.linalg.norm(offset_pos)}")
                        marker_pos_in_sensor_list.append(marker_pos_in_sensor - offset_pos)
                        if std_out is not None:
                            std_out.attach(gm.gen_sphere(pos=marker_pos_in_sensor_list[-1], radius=0.001))
        print(f"There exit {len(marker_pos_in_sensor_list)} in the system")
        if len(marker_pos_in_sensor_list) < 3:
            return [None, None]
        center_in_camera_coords_array = np.asarray(marker_pos_in_sensor_list)

        center, radius, inliers = self.ransac_sphere_fitter.fit(center_in_camera_coords_array, thresh=0.002)
        print(f"The inliers are {inliers}")
        for i in inliers:
            gm.gen_sphere(pos=np.array(i), radius=0.003).attach_to(base)
        print(f"the center is {center}, the radius is {radius}")
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        return np.array(center), np.array(radius)

    def _get_rotmat_w_c(self, reference_pos, sensor_marker_handler, action_dist):
        component_name = self._component_name
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        reference_rotmat = self._init_rotmat
        # move to action pos, action rotmat
        last_jnt_values = self.robot_x.get_jnt_values(component_name=component_name)
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=reference_pos,
                                     tgt_rotmat=reference_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=3000, local_minima="accept")
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center is not reachable. Try a different pos or robtmat!")
        # move to x+action_dist
        action_center_dist_x = reference_pos + reference_rotmat[:, 0] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_x,
                                     tgt_rotmat=reference_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=3000 , local_minima="accept")
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_xplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with xplus is not reachable. Try a different pos or robtmat!")
        # move to y+action_dist
        action_center_dist_y = reference_pos + reference_rotmat[:, 1] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_y,
                                     tgt_rotmat=reference_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=3000 , local_minima="accept" )
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            last_jnt_values = jnt_values
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_yplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with yplus is not reachable. Try a different pos or robtmat!")
        # move to z+action_dist
        action_center_dist_z = reference_pos + reference_rotmat[:, 2] * action_dist
        jnt_values = self.robot_s.ik(component_name=component_name,
                                     tgt_pos=action_center_dist_z,
                                     tgt_rotmat=reference_rotmat,
                                     seed_jnt_values=last_jnt_values, max_niter=3000 , local_minima="accept")
        if jnt_values is not None and not self.robot_s.is_collided():
            self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
            self.robot_x.move_jnts(component_name, jnt_values)
            marker_pos_zplus_in_sensor = sensor_marker_handler.get_marker_center()
        else:
            raise ValueError("The action center with zplus is not reachable. Try a different pos or robtmat!")
        print(
            "Markerpos in sensor", marker_pos_in_sensor,
            "Markerpos +x in sensor", marker_pos_xplus_in_sensor,
            "Markerpos +y in sensor", marker_pos_yplus_in_sensor,
            "Markerpos +z in sensor", marker_pos_zplus_in_sensor
        )
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_xplus_in_sensor, rgba=[1, 0, 0, 1]).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_yplus_in_sensor, rgba=[0, 1, 0, 1]).attach_to(base)
        gm.gen_arrow(spos=marker_pos_in_sensor, epos=marker_pos_zplus_in_sensor, rgba=[0, 0, 1, 1]).attach_to(base)
        normalized_marker_mat_in_sensor = np.array([marker_pos_xplus_in_sensor - marker_pos_in_sensor,
                                                    marker_pos_yplus_in_sensor - marker_pos_in_sensor,
                                                    marker_pos_zplus_in_sensor - marker_pos_in_sensor]).T
        self.robot_s.fk(component_name=component_name, jnt_values=jnt_values_bk)
        return np.linalg.qr(normalized_marker_mat_in_sensor)[0].T

    def find_board_center_in_hand(self,
                                  sensor_marker_handler,
                                  reference_point=np.array([.3, -.05, .2]),
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

        rotmat_w_c = self._get_rotmat_w_c(reference_pos=reference_point,
                                          sensor_marker_handler=sensor_marker_handler,
                                          action_dist=action_dist)

        # print(rotmat_w_c)
        # base.run()
        tcp_in_sensor, radius_by_markers = self._find_tcp_in_sensor(reference_pos=reference_point,
                                                                    sensor_marker_handler=sensor_marker_handler,
                                                                    rotmat_w_c=rotmat_w_c,
                                                                    std_out=std_out)
        print(radius_by_markers)
        sphere = gm.gen_sphere(pos=tcp_in_sensor, radius=radius_by_markers, rgba=[1, 0, 0, .3])
        gm.gen_sphere(pos=tcp_in_sensor, radius=0.01, rgba=[1, 0, 0, .3]).attach_to(base)
        std_out.attach(node=sphere, name="estimated sphere")

        return

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
                                         seed_jnt_values=last_jnt_values, max_niter=3000,)
            if jnt_values is not None:
                self.robot_s.fk(component_name=component_name, jnt_values=jnt_values)
                last_jnt_values = jnt_values
                if not self.robot_s.is_collided():
                    if self.robotx_move(jnt_values=jnt_values):
                        marker_pos_in_sensor = sensor_marker_handler.get_marker_center()
                        if marker_pos_in_sensor is not None:
                            pos_in_real_list.append(action_pos + np.dot(action_rotmat_list[i], marker_pos_in_hnd))
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


if __name__ == "__main__":
    from huri.components.yumi_control.yumi_con import YumiController
    from huri.core.common_import import *
    from huri.components.pipeline.data_pipeline import RenderController
    import vision.depth_camera.sensor_marker_handler as dcsmh
    import basis.trimesh.primitives as btp

    '''initilization'''
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    std_out = RenderController(root=base.tkRoot, base=base)
    robot_s = ym.Yumi()  # simulation robot
    component_name = "rgt_hnd"
    save_margin = 1.3
    board_cm = cm.CollisionModel(btp.Box(box_extents=np.array([0.005, 0.15, 0.1]) * save_margin))
    robot_s.gen_meshmodel().attach_to(base)
    tooltip_pos, tooltip_rot = robot_s.get_gl_tcp(manipulator_name=component_name)
    board_cm.set_homomat(rm.homomat_from_posrot(tooltip_pos, tooltip_rot))
    board_cm.attach_to(base)
    board_cm.show_localframe()
    robot_s.hold(hnd_name=component_name, objcm=board_cm, jaw_width=0.005)

    robot_x = YumiController()  # robot control handler
    calibrator = DepthCaliberator(robot_x=robot_x,
                                  robot_s=robot_s,
                                  component_name=component_name,
                                  init_rotmat=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).T)  # init calibrator
    calibrator.robotx_to_init()
    rgt_arm_jnts = robot_s.get_jnt_values(component_name=component_name)
    sensor_handler = dcsmh.SensorMarkerHandler(cv_show=True, debug=False)
    calibrator.find_board_center_in_hand(
        sensor_marker_handler=sensor_handler,
        reference_point=np.array([.3, 0.05, .2]),
        action_dist=.1,
        std_out=std_out)
    base.run()

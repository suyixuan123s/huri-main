import argparse
from typing import List
from time import strftime

import numpy as np

import basis.trimesh as tm
import motion.probabilistic.rrt_connect as rrtc

from huri.core.common_import import cm
from _constants import *  # import constants and logging conf
from rgb_phoxi_calib.utils import project_pcd2extcamimg
from utils import SelectRegionBasePhoxiStreamer, img_to_n_channel, phoxi_map_pcd_2_poly_mask, highlight_mask
from huri.core.file_sys import workdir
from huri.core.constants import SENSOR_INFO, ANNOTATION
from huri.core._logging import colorstr
from huri.definitions.tube_def import TubeType
from huri.vision.pnt_utils import extc_fgnd_pcd
from huri.vision.phoxi_capture import depth2gray_map, enhance_gray_img, vision_pipeline

logger = logging.getLogger(__file__)

VERSION = ANNOTATION.VERSION
Label = ANNOTATION.LABEL
Bbox = ANNOTATION.BBOX_XYXY
Save_format = ANNOTATION.IN_HAND_ANNOTATION_SAVE_FORMAT


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tube_type', type=str, default='blue cap', help='Type of the test tube to be annotated')
    parser.add_argument('--debug', type=bool, default=False, help='Is debug?')
    parser.add_argument('--color_img', type=bool, default=False, help='0 for gray image, 1 for color image')
    opt = parser.parse_args()
    opt.img_type = ["gray", "color"][opt.color_img]
    logger.info(colorstr(opt))
    return opt


class RobotMover:
    """
    Move the robot to indicated waypoint
    Author: Chen Hao
    Email: chen960216@gmail.com
    Date: 20220313
    """

    def __init__(self, yumi_s, yumi_con=None, stop=400, obj_cm=None, logger=logger):
        """
        :param yumi_s: robot_sim.robots.yumi.Yumi
        :param yumi_con: huri.component.control.yumi_con.YumiController
        """
        # simulation robot, robot controller, rrt-connect planner
        self._yumi_s = yumi_s
        self._yumi_con = yumi_con
        self._rrtc_pl = rrtc.RRTConnect(yumi_s)

        # waypoints set for rgt and lft arm; waypoint counter
        self._wp_rgt = []
        self._wp_lft = []
        self._wp_ctr_rgt = 0
        self._wp_ctr_lft = 0
        self.stop = stop

        # cached waypoint path
        self._wp_rgt_cache_path = SEL_PARAM_PATH.joinpath("wp_rgt.cache")
        self._wp_lft_cache_path = SEL_PARAM_PATH.joinpath("wp_lft.cache")

        # object be hold in the hand
        self._obj_cm = obj_cm

        # logger
        self._logger = logger

        # setup the collision model for Phoxi sensor
        phoxi_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
        phoxi_obs.attach_to(base)
        phoxi_obs.set_pos(np.array([0.1, 0, 1]))
        self._obs_list = [phoxi_obs]

    def go_init(self, armname="both"):
        if "rgt" in armname or armname == "both":
            init_target_pos = np.array([.3, -.1, .15])
            init_target_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
            jnt_values = self._yumi_s.ik(component_name=armname,
                                         tgt_pos=init_target_pos,
                                         tgt_rotmat=init_target_rot,
                                         max_niter=100)
            rrt_path_rgt = self._rrtc_pl.plan(component_name=armname,
                                              start_conf=self._yumi_con.get_jnt_values(armname),
                                              goal_conf=np.array(jnt_values),
                                              obstacle_list=self._obs_list,
                                              ext_dist=.01,
                                              max_time=300)
            self._yumi_con.move_jnts(component_name="rgt_arm", jnt_vals=rrt_path_rgt)

        if "lft" in armname or armname == "both":
            init_target_pos = np.array([.3, .1, .15])
            init_target_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
            jnt_values = self._yumi_s.ik(component_name=armname,
                                         tgt_pos=init_target_pos,
                                         tgt_rotmat=init_target_rot,
                                         max_niter=100)
            rrt_path_lft = self._rrtc_pl.plan(component_name=armname,
                                              start_conf=self._yumi_con.get_jnt_values(armname),
                                              goal_conf=np.array(jnt_values),
                                              obstacle_list=self._obs_list,
                                              ext_dist=.01,
                                              max_time=300)
            self._yumi_con.move_jnts(component_name="lft_arm", jnt_vals=rrt_path_lft)

    def add_wp_homomats(self, homomats: List[np.ndarray], armname: str = "rgt_arm", load: bool = True):
        """
        Add waypoint in form of homogeneous matrix. Waypoints will be restored in self._wp_rgt/self._wp_lft
        :param homomats: waypoints in form of homogenous matrix list (np.array with 4x4 shape)
        :param armname: indicate the arm for waypoints
        :param load: True: load previous generated waypoints, if the previous data does not exit, it will generate new data
                     False: generate new data
        """
        assert armname in ["rgt_arm", "lft_arm"]
        if "rgt" in armname:
            wp_list = self._wp_rgt
            wp_cache_path = self._wp_rgt_cache_path
            init_target_pos = np.array([.3, -.1, .15])
            init_target_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
        else:
            wp_list = self._wp_lft
            wp_cache_path = self._wp_lft_cache_path
            init_target_pos = np.array([.3, .1, .15])
            init_target_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        if load and wp_cache_path.exists():
            wp_list.extend(fs.load_pickle(wp_cache_path))
        else:
            robot_s = self._yumi_s
            # generate the
            jnt_values_bk = robot_s.get_jnt_values(armname)
            last_jnt_values = jnt_values_bk
            init_jnt_values = robot_s.ik(component_name=armname,
                                         tgt_pos=init_target_pos,
                                         tgt_rotmat=init_target_rot,
                                         seed_jnt_values=last_jnt_values,
                                         max_niter=100)
            wp_list.append(init_jnt_values)

            for idx, homomat in enumerate(homomats):
                jnt_values = robot_s.ik(component_name=armname,
                                        tgt_pos=homomat[:3, 3],
                                        tgt_rotmat=homomat[:3, :3],
                                        seed_jnt_values=last_jnt_values,
                                        max_niter=100)
                if jnt_values is not None:
                    robot_s.fk(component_name=armname, jnt_values=jnt_values)
                    if not robot_s.is_collided():
                        # last_jnt_values = jnt_values
                        wp_list.append(jnt_values)
                self._logger.debug(colorstr(f"{idx}, jnt value: {jnt_values}"))
            fs.dump_pickle(wp_list, wp_cache_path)
            robot_s.fk(armname, jnt_values_bk)
        self._logger.info(colorstr(f"Feasible number of homogenous matrix is {len(wp_list)}"))

    def _set_wp_ctr(self, val: int, armname: str = "rgt_arm"):
        """
        Set counter of waypoints
        """
        assert armname in ["rgt_arm", "lft_arm"] and isinstance(val, int)
        if "rgt" in armname:
            self._wp_ctr_rgt = val
        else:
            self._wp_ctr_lft = val

    def _get_wp_ctr(self, armname: str = "rgt_arm") -> int:
        """
        Get counter of waypoints
        """
        assert armname in ["rgt_arm", "lft_arm"]
        if "rgt" in armname:
            return self._wp_ctr_rgt
        else:
            return self._wp_ctr_lft

    def goto_next_wp(self, armname: str = "rgt_arm"):
        """
        Move robot to the next waypoint
        :param armname: Arm name of the robot
        """
        if self._yumi_con is None:
            if "rgt" in armname:
                wp_list = self._wp_rgt
            else:
                wp_list = self._wp_lft
            wp_jnt_val = wp_list[self._get_wp_ctr(armname)]
            self._set_wp_ctr(self._get_wp_ctr(armname) + 1, armname)
            self._yumi_s.fk(armname, wp_jnt_val)
            return

        assert armname in ["rgt_arm", "lft_arm"]

        if "rgt" in armname:
            wp_list = self._wp_rgt
        else:
            wp_list = self._wp_lft

        if self._get_wp_ctr(armname) >= min(len(wp_list), self.stop):
            self._logger.debug(colorstr(len(wp_list)))
            raise StopIteration

        # sync simulation robot poses to real robot poses
        self._yumi_s.fk("rgt_arm", self._yumi_con.get_jnt_values("rgt_arm"))
        self._yumi_s.fk("lft_arm", self._yumi_con.get_jnt_values("lft_arm"))

        wp_jnt_val = wp_list[self._get_wp_ctr(armname)]
        # increase waypoint counter
        self._set_wp_ctr(self._get_wp_ctr(armname) + 1, armname)

        # plan motion to move to next waypoint
        if self._obj_cm is not None:
            pos, rot = self._yumi_s.cvt_loc_tcp_to_gl(armname,
                                                      np.array([0, -0.045, 0]),
                                                      np.array([
                                                          [1, 0, 0],
                                                          [0, 0, -1],
                                                          [0, 1, 0]
                                                      ]).T,
                                                      # np.array([0, 0, 0]),
                                                      )

            self._obj_cm.set_homomat(rm.homomat_from_posrot(pos, rot))
            self._yumi_s.hold(hnd_name=f"{armname[:-3]}hnd", objcm=self._obj_cm, )
            # self._obj_cm.attach_to(base)
        # rrt_path_rgt = self._rrtc_pl.plan(component_name=armname,
        #                                   start_conf=self._yumi_con.get_jnt_values(armname),
        #                                   goal_conf=np.array(wp_jnt_val),
        #                                   obstacle_list=self._obs_list,
        #                                   ext_dist=.01,
        #                                   max_time=300)
        if self._obj_cm is not None:
            self._yumi_s.release(hnd_name=f"{armname[:-3]}hnd", objcm=self._obj_cm, )
        # if rrt_path_rgt is None:
        #     return
        # if len(rrt_path_rgt) < 5:
        #     print("Move jnts")
        self._yumi_con.move_jnts(component_name=armname, jnt_vals=wp_jnt_val)
        # else:
        #     print("Move RRT")
        #     self._yumi_con.move_jntspace_path(component_name=armname, path=rrt_path_rgt
        #                                       , speed_n=300)


class SelectRegion(SelectRegionBasePhoxiStreamer):
    def __init__(self, showbase,
                 robot_mover: RobotMover,
                 tube_label="blue tube",
                 save_path=DATA_ANNOT_PATH,
                 pcd_affine_matrix=None,
                 streamer_ip=None,
                 work_armname="both",
                 debug_path=None,
                 toggle_debug=False,
                 img_type="gray"):

        assert work_armname in ["rgt_arm", "lft_arm", "both"]
        self._params_path = SEL_PARAM_PATH.joinpath("params_in_hnd.json")
        super(SelectRegion, self).__init__(showbase,
                                           param_path=self._params_path,
                                           vision_server_ip=streamer_ip,
                                           pcd_affine_matrix=pcd_affine_matrix,
                                           img_type=img_type,
                                           toggle_debug=toggle_debug,
                                           debug_path=debug_path
                                           )

        # is use both arm?
        if work_armname == "both":
            self._use_both_arm = True
        else:
            self._use_both_arm = False
        self._work_armname = work_armname  # work arm

        # init gui
        self._init_gui()

        # control sensor capture and robot movement
        self._yumi_con = robot_mover._yumi_con  # robot controller
        self._robot_mover = robot_mover  # robot mover
        self._rbt_tcp_pos = [None, None]  # record robot tcp pos
        self._rbt_tcp_rot = [None, None]  # record robot tcp rotmat
        self._rbt_jnts = [None, None]

        # label info and save path
        self._tube_label = tube_label
        self._save_path = fs.Path(save_path)

        # let the robot mover move to init pose
        if self._robot_mover is not None:
            self._move_rbt_to_next_wp()

        if toggle_debug:
            self._update(None)

        # backgound point cloud data
        self._bgnd_pcd = None

    def _init_gui(self):
        # init x direction in tcp coordinate
        self._x_bound = np.array([-.1, .1])
        self._y_bound = np.array([-.1, .1])
        self._z_bound = np.array([-.1, .1])
        _row = 1
        self._xrange = [
            self._panel.add_scale(f"x-",
                                  default_value=self._params['last_selection_val_x'][0],
                                  command=self._update,
                                  val_range=[self._x_bound[0], self._x_bound[1]],
                                  pos=(_row, 0)),
            self._panel.add_scale(f"x+",
                                  default_value=self._params['last_selection_val_x'][1],
                                  command=self._update,
                                  val_range=[self._x_bound[0], self._x_bound[1]],
                                  pos=(_row, 1))]
        # init y direction in tcp coordinate

        self._yrange = [
            self._panel.add_scale(f"y-",
                                  default_value=self._params['last_selection_val_y'][0],
                                  command=self._update,
                                  val_range=[self._y_bound[0], self._y_bound[1]],
                                  pos=(_row + 1, 0)),
            self._panel.add_scale(f"y+",
                                  default_value=self._params['last_selection_val_y'][1],
                                  command=self._update,
                                  val_range=[self._y_bound[0], self._y_bound[1]],
                                  pos=(_row + 1, 1))]
        # init z direction in tcp coordinate

        self._zrange = [
            self._panel.add_scale(f"z-",
                                  default_value=self._params['last_selection_val_z'][0],
                                  command=self._update,
                                  val_range=[self._z_bound[0], self._z_bound[1]],
                                  pos=(_row + 2, 0)),
            self._panel.add_scale(f"z+",
                                  default_value=self._params['last_selection_val_z'][1],
                                  command=self._update,
                                  val_range=[self._z_bound[0], self._z_bound[1]],
                                  pos=(_row + 2, 1))]
        _row += 3
        if self._use_both_arm:
            self._xrange += [self._panel.add_scale(f"x-",
                                                   default_value=self._params['last_selection_val_x'][2],
                                                   command=self._update,
                                                   val_range=[self._x_bound[0], self._x_bound[1]],
                                                   pos=(_row, 2)),
                             self._panel.add_scale(f"x+",
                                                   default_value=self._params['last_selection_val_x'][3],
                                                   command=self._update,
                                                   val_range=[self._x_bound[0], self._x_bound[1]],
                                                   pos=(_row, 3))]
            self._yrange += [self._panel.add_scale(f"y-",
                                                   default_value=self._params['last_selection_val_y'][2],
                                                   command=self._update,
                                                   val_range=[self._y_bound[0], self._y_bound[1]],
                                                   pos=(_row + 1, 2)),
                             self._panel.add_scale(f"y+",
                                                   default_value=self._params['last_selection_val_y'][3],
                                                   command=self._update,
                                                   val_range=[self._y_bound[0], self._y_bound[1]],
                                                   pos=(_row + 1, 3))]
            self._zrange += [self._panel.add_scale(f"z-",
                                                   default_value=self._params['last_selection_val_z'][2],
                                                   command=self._update,
                                                   val_range=[self._z_bound[0], self._z_bound[1]],
                                                   pos=(_row + 2, 2)),
                             self._panel.add_scale(f"z+",
                                                   default_value=self._params['last_selection_val_z'][3],
                                                   command=self._update,
                                                   val_range=[self._z_bound[0], self._z_bound[1]],
                                                   pos=(_row + 2, 3))]
            _row += 3

        # self._panel.add_button(text="Get background pcd", command=self._get_bgnd_pcd, pos=(_row, 0))
        self._panel.add_button(text="Get data and render and save", command=self._render_acquire_data_and_save,
                               pos=(_row, 0))
        self._panel.add_button(text="Get data and render", command=self._render_acquire_data, pos=(_row, 1))
        self._panel.add_button(text="Move robot to next position", command=self._move_rbt_to_next_wp,
                               pos=(_row + 1, 0))
        self._panel.add_button(text="Auto Collect Data", command=self._auto_collect, pos=(_row + 1, 1))
        self._panel.show()

    def _get_bgnd_pcd(self):
        self._render_acquire_data()
        if self._pcd is not None:
            self._bgnd_pcd = self._pcd.copy()
        else:
            logger.error(colorstr(f"Cannot acquire background point cloud, please check acquiring data"))

    def _move_rbt_to_next_wp(self):
        if self._use_both_arm:
            self._robot_mover.goto_next_wp("rgt_arm")
            self._robot_mover.goto_next_wp("lft_arm")
        else:
            self._robot_mover.goto_next_wp(self._work_armname)

    def _auto_collect(self):
        def task_auto_collect(t, task):
            try:
                t._move_rbt_to_next_wp()
                t._render_acquire_data_and_save()
            except StopIteration:
                print("Finished")
                return task.done
            except Exception as e:
                raise Exception(e)
                pass
            return task.again

        print("TASK again")
        taskMgr.doMethodLater(1, task_auto_collect, "autocollect", extraArgs=[self], appendTask=True, priority=999)

    def _render_acquire_data(self):
        self._acquire_data()
        if self._pcd is None or self._depth_img is None or self._img is None:
            return
        if self._pcd_affine_mat is not None:
            self._pcd_aligned = rm.homomat_transform_points(self._pcd_affine_mat, points=self._pcd)

        return self._update(None)

    def _render_acquire_data_and_save(self):
        annotaions = self._render_acquire_data()
        fs.dump_pickle(
            tuple(Save_format(version=VERSION,
                              pcd=self._pcd,
                              gray_img=self._texture,
                              extcam_img=self._extcam_img,
                              depth_img=self._depth_img,
                              rbt_tcp_pos=self._rbt_tcp_pos,
                              rbt_tcp_rot=self._rbt_tcp_rot,
                              rbt_joints=self._rbt_jnts,
                              annotations=annotaions)),
            self._save_path / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

    def _acquire_data(self):
        """
        1. Acquire the raw sensor data: 1)pcd 2)texture 3) depth image
        2. Acquire the robot tcp information
        """

        # acquire vision data
        if self._streamer is None:
            self._pcd, self._texture, self._depth_img, self._rgb_texture, self._extcam_img = None, None, None, None, None
            print("Cannot acquire data")
        else:
            self._pcd, self._texture, self._depth_img, \
                self._rgb_texture, self._extcam_img = vision_pipeline(self._streamer,
                                                                      rgb_texture=True if self._img_type == "color" else False,
                                                                      get_depth=True)
            self._assign_img()

        # acquire robot data
        if self._yumi_con is None:
            self._rbt_tcp_pos, self._rbt_tcp_rot, self._rbt_jnts = [None, None], [None, None], [None, None]
        else:
            if self._use_both_arm:
                self._rbt_tcp_pos[0], self._rbt_tcp_rot[0] = self._yumi_con.get_pose(component_name="rgt_arm")
                self._rbt_tcp_pos[1], self._rbt_tcp_rot[1] = self._yumi_con.get_pose(component_name="lft_arm")
                self._rbt_jnts[0], self._rbt_jnts[1] = self._yumi_con.get_jnt_values(
                    component_name="rgt_arm"), self._yumi_con.get_jnt_values(component_name="lft_arm")

            else:
                work_arm_id = ANNOTATION.WORK_ARM_ID.RIGHT  # 0 for right arm 1 for left arm
                if "lft" in self._work_armname:
                    work_arm_id = ANNOTATION.WORK_ARM_ID.LEFT
                self._rbt_tcp_pos[work_arm_id], self._rbt_tcp_rot[work_arm_id] = self._yumi_con.get_pose(
                    component_name=self._work_armname)
                self._rbt_jnts[work_arm_id] = self._yumi_con.get_jnt_values(component_name="lft_arm")

    def _is_prepared(self):
        """
        Make sure the necessary information about the robot is successfully acquired
        """
        return self._rbt_tcp_pos[0] is not None and self._rbt_tcp_rot[0] is not None and self._pcd is not None \
            and self._img is not None and self._depth_img is not None

    def _update(self, x):
        bad_img = False
        if not self._is_prepared():
            return

        x_range = np.array([_.get() for _ in self._xrange])
        y_range = np.array([_.get() for _ in self._yrange])
        z_range = np.array([_.get() for _ in self._zrange])

        # depth_img_labeled = img_to_n_channel(depth2gray_map(self._depth_img))
        if self._img_type == "gray":
            img_labeled = img_to_n_channel(enhance_gray_img(self._img))
        else:
            img_labeled = self._img.copy()

        # render pcd
        # if self._bgnd_pcd is not None:
        #     ind = np.where(np.linalg.norm((self._pcd - self._bgnd_pcd)) > .0)[0]
        #     print(ind)
        #     self._render_pcd(self._pcd_aligned[ind] if self._pcd_aligned is not None else self._pcd[ind])
        # else:
        self._render_pcd(self._pcd_aligned if self._pcd_aligned is not None else self._pcd)

        labels = []
        for indx in range(len(self._rbt_tcp_pos)):
            if self._rbt_tcp_pos[indx] is None:
                continue
            data = SelectRegion.extract_pixel_around_hand(self._rbt_tcp_pos[indx],
                                                          self._rbt_tcp_rot[indx],
                                                          self._pcd_affine_mat,
                                                          self._pcd,
                                                          img_shape=self._img.shape,
                                                          extract_area=(x_range[2 * indx:2 * indx + 2],
                                                                        y_range[2 * indx:2 * indx + 2],
                                                                        z_range[2 * indx:2 * indx + 2]))
            if data is None:
                bad_img = True
                continue
            (h1, w1, h2, w2), polygon_mask, extracted_pcd_idx = data
            if len(extracted_pcd_idx) < 400:
                continue

            label_info = tuple(Label(label_name=self._tube_label,
                                     version=VERSION,
                                     img_type="gray",
                                     bboxes=tuple(Bbox(w1=w1, h1=h1, w2=w2, h2=h2)),
                                     polygons=polygon_mask,
                                     extracted_pcd_idx=extracted_pcd_idx, ))
            labels.append(label_info)

            if self._img_type == "color":
                feasible_points, feasible_points_indx = project_pcd2extcamimg(pcd_raw=self._pcd[extracted_pcd_idx],
                                                                              phoxi2extcam_homo_mat=np.linalg.inv(
                                                                                  self._homo_mat),
                                                                              cam_mat=self._cam_mat,
                                                                              extcam_res=self._cam_res,
                                                                              dist_coef=self._dist_coef, )
                w1, h1 = feasible_points[:, 0].min(), feasible_points[:, 1].min()
                w2, h2 = feasible_points[:, 0].max(), feasible_points[:, 1].max()

                label_info = tuple(Label(label_name=self._tube_label,
                                         version=VERSION,
                                         img_type="color",
                                         bboxes=tuple(Bbox(w1=w1, h1=h1, w2=w2, h2=h2)),
                                         polygons=None,
                                         extracted_pcd_idx=extracted_pcd_idx, ))
                labels.append(label_info)

            img_labeled = cv2.rectangle(img_labeled, (w1, h1), (w2, h2),
                                        color=(255, 0, 0), thickness=3)
            img_labeled = highlight_mask(img_labeled, polygon_mask)

            gm.gen_pointcloud(
                self._pcd_aligned[extracted_pcd_idx] if self._pcd_aligned is not None else self._pcd[extracted_pcd_idx],
                rgbas=[[1, 0, 0, 1]]).attach_to(self._np_pcd)
        # self._render_depth_img(depth_img_labeled)
        self._render_img(img_labeled)
        if self._np_pcd is not None:
            trans = rm.homomat_from_posrot(self._rbt_tcp_pos[0], self._rbt_tcp_rot[0])
            trans[:3, 3] += ((x_range[1] - self._x_bound[1]) - (self._x_bound[0] - x_range[0])) * trans[:3, 0] / 2
            trans[:3, 3] += ((y_range[1] - self._y_bound[1]) - (self._y_bound[0] - y_range[0])) * trans[:3, 1] / 2
            trans[:3, 3] += ((z_range[1] - self._z_bound[1]) - (self._z_bound[0] - z_range[0])) * trans[:3, 2] / 2

            gm.gen_box([x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]],
                       homomat=trans, rgba=(1, 0, 1, .2)).attach_to(self._np_pcd)
            gm.gen_frame(self._rbt_tcp_pos[0], self._rbt_tcp_rot[0]).attach_to(self._np_pcd)
            if self._use_both_arm:
                trans = rm.homomat_from_posrot(self._rbt_tcp_pos[1], self._rbt_tcp_rot[1])
                trans[:3, 3] += ((x_range[3] - self._x_bound[1]) - (self._x_bound[0] - x_range[2])) * trans[:3, 0] / 2
                trans[:3, 3] += ((y_range[3] - self._y_bound[1]) - (self._y_bound[0] - y_range[2])) * trans[:3, 1] / 2
                trans[:3, 3] += ((z_range[3] - self._z_bound[1]) - (self._z_bound[0] - z_range[2])) * trans[:3, 2] / 2

                gm.gen_box([x_range[3] - x_range[2], y_range[3] - y_range[2], z_range[3] - z_range[2]],
                           homomat=trans, rgba=(0, 1, 1, .2)).attach_to(self._np_pcd)
                gm.gen_frame(self._rbt_tcp_pos[1], self._rbt_tcp_rot[1]).attach_to(self._np_pcd)
        self._save_params()
        if bad_img:
            return None
        return labels

    def _save_params(self):
        x_range = np.array([_.get() for _ in self._xrange])
        y_range = np.array([_.get() for _ in self._yrange])
        z_range = np.array([_.get() for _ in self._zrange])
        for i in range(len(x_range)):
            self._params["last_selection_val_x"][i] = x_range[i]
        for i in range(len(y_range)):
            self._params["last_selection_val_y"][i] = y_range[i]
        for i in range(len(z_range)):
            self._params["last_selection_val_z"][i] = z_range[i]
        fs.dump_json(self._params, self._params_path, reminder=False)

    @staticmethod
    def extract_pixel_around_hand(robot_tcp_pos,
                                  robot_tcp_rotmat,
                                  affine_mat,
                                  pcd,
                                  img_shape,
                                  extract_area=((-.03, .03), (.01, .05), (-.03, .03)), ):
        # Transform and Plot Point Clouds
        pcd = rm.homomat_transform_points(affine_mat, points=pcd)
        pcd_in_hand = rm.homomat_transform_points(
            rm.homomat_inverse(rm.homomat_from_posrot(robot_tcp_pos, robot_tcp_rotmat)),
            pcd)
        extracted_pcd_idx = np.where((pcd_in_hand[:, 0] > extract_area[0][0]) & (pcd_in_hand[:, 0] < extract_area[0][1])
                                     & (pcd_in_hand[:, 1] > extract_area[1][0]) & (
                                             pcd_in_hand[:, 1] < extract_area[1][1])
                                     & (pcd_in_hand[:, 2] > extract_area[2][0]) & (
                                             pcd_in_hand[:, 2] < extract_area[2][1]))[
            0]
        if len(extracted_pcd_idx) < 1:
            return None
        h, w = img_shape[0], img_shape[1]
        idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
        h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
        w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()

        ploygon_mask = phoxi_map_pcd_2_poly_mask(extracted_pcd_idx, img_shape, conv_area=True)
        return (h1, w1, h2, w2), ploygon_mask, extracted_pcd_idx


if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    # par
    opt = parse_opt()
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    yumi_s = ym.Yumi(enable_cc=True)
    work_arm = "rgt_arm"
    # # init the real robot and get joint angles
    if not opt.debug:
        yumi_con = YumiController(debug=False)
        jnt_vals_rgt = yumi_con.get_jnt_values(component_name="rgt_arm")
        jnt_vals_lft = yumi_con.get_jnt_values(component_name="lft_arm")
        # sync sim to real
        yumi_s.fk("rgt_arm", jnt_vals_rgt)
        yumi_s.fk("lft_arm", jnt_vals_lft)
    else:
        yumi_con = None

    # affine matrix
    affine_mat = np.asarray(
        fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)['affine_mat'])

    tube = TubeType.gen_tube_by_name(opt.tube_type)
    # setup robot mover
    rbt_mover = RobotMover(yumi_s=yumi_s, yumi_con=yumi_con, obj_cm=tube.gen_collision_model())

    # pos_list_rgt = []
    # for pos in np.array(np.meshgrid(np.linspace(.3, .4, num=2),
    #                                 np.linspace(-.20, -.1, num=2),
    #                                 np.array([.2]))).T.reshape(-1, 3):
    #     rots_candidate = np.array(rm.gen_icorotmats(icolevel=3,
    #                                                 rotation_interval=np.radians(60),
    #                                                 crop_normal=np.array([0, 0, 1]),
    #                                                 crop_angle=np.pi / 6,
    #                                                 toggleflat=True))
    #     rots_candidate[..., [2, 1]] = rots_candidate[..., [1, 2]]
    #     rots_candidate[..., 0] = -rots_candidate[..., 0]
    #     for rot in rots_candidate:
    #         pos_list_rgt.append(rm.homomat_from_posrot(pos, rot))
    # print(f"There are {len(pos_list_rgt)} rgt poses")
    # rbt_mover.add_wp_homomats(pos_list_rgt, armname="rgt_arm")

    # pos_list_lft = []
    # for pos in np.array(np.meshgrid(np.linspace(.3, .4, num=2),
    #                                 np.linspace(.20, .1, num=2),
    #                                 np.array([.2]))).T.reshape(-1, 3):
    #     rots_candidate = np.array(rm.gen_icorotmats(icolevel=3,
    #                                                 rotation_interval=np.radians(60),
    #                                                 crop_normal=np.array([0, 0, 1]),
    #                                                 crop_angle=np.pi / 6,
    #                                                 toggleflat=True))
    #     rots_candidate[..., [2, 1]] = rots_candidate[..., [1, 2]]
    #     rots_candidate[..., 1] = -rots_candidate[..., 1]
    #     for rot in rots_candidate:
    #         pos_list_lft.append(rm.homomat_from_posrot(pos, rot))
    # print(f"There are {len(pos_list_lft)} lft poses")
    # rbt_mover.add_wp_homomats(pos_list_lft, armname="lft_arm")

    # rbt_mover._wp_ctr_rgt = 820
    # rbt_mover._wp_ctr_lft = 820
    # rbt_mover.move_next_record_pos(work_arm="rgt_arm")
    # rbt_mover.move_next_record_pos(work_arm="lft_arm")
    # counter = 0
    # while True:
    #     print(counter)
    #     counter += 1
    #     try:
    #         rbt_mover.move_next_record_pos(work_arm="rgt_arm")
    #         rbt_mover.move_next_record_pos(work_arm="lft_arm")
    #     except:
    #         raise Exception("ST")
    # rbt_mover._set_wp_ctr(396, armname="lft_arm")
    # rbt_mover._set_wp_ctr(396, armname="rgt_arm")

    # opt.debug = True
    tube_type = "WHITE_CAP"
    rbt_mover.add_wp_homomats(None, armname="rgt_arm", load=True)
    rbt_mover.add_wp_homomats(None, armname="lft_arm", load=True)
    al = SelectRegion(showbase=base,
                      robot_mover=rbt_mover,
                      # tube_label=opt.tube_type,
                      tube_label=tube_type,
                      pcd_affine_matrix=affine_mat,
                      work_armname="both",
                      streamer_ip=SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG,
                      debug_path=workdir.joinpath("data", "data_annotation", "EXP", "BLUE_DRAW", "20221004-160811.pkl"),
                      save_path=DATA_ANNOT_PATH.joinpath("EXP", tube_type),
                      toggle_debug=opt.debug,
                      img_type="gray",
                      # img_type=opt.img_type
                      )

    base.startTk()
    base.tkRoot.withdraw()
    base.run()

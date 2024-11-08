import matplotlib
from time import strftime

import basis.trimesh as tm
from huri.core.file_sys import workdir
from huri.core.base_boost import spawn_window, gen_img_texture_on_render, set_img_texture
from huri.vision.phoxi_capture import (vision_pipeline,
                                       SensorMarkerHandler,
                                       depth2gray_map,
                                       enhance_gray_img, )
from huri.components.utils.panda3d_utils import img_to_n_channel
from huri.components.gui.tk_gui.base import GuiFrame
import motion.probabilistic.rrt_connect as rrtc

matplotlib.use('TkAgg')

IP_ADR = "192.168.125.100:18300"
SAVE_PATH = workdir / "examples" / "vision" / "paper" / "data" / "dataset_bluecap_2"
AFFINE_MAT_PATH = workdir / "data/calibration/qaqqq.json"
DEBUG = False
TUBE_TYPE = "blue cap"


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
                                 & (pcd_in_hand[:, 1] > extract_area[1][0]) & (pcd_in_hand[:, 1] < extract_area[1][1])
                                 & (pcd_in_hand[:, 2] > extract_area[2][0]) & (pcd_in_hand[:, 2] < extract_area[2][1]))[
        0]
    if len(extracted_pcd_idx) < 1:
        return None
    h, w = img_shape[0], img_shape[1]
    idx_in_pixel = np.unravel_index(extracted_pcd_idx, (h, w))
    h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
    w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
    return h1, w1, h2, w2, extracted_pcd_idx


class RobotMover:
    def __init__(self, yumi_s, yumi_con=None, ):
        self._yumi_s = yumi_s
        self.yumi_con = yumi_con

        self._rrtc_planner = rrtc.RRTConnect(yumi_s)

        self._record_loc_rgt = []
        self._record_loc_lft = []
        self._counter_rgt = 0
        self._counter_lft = 0

        # setup the collision model for phoxi camera
        camera_obs = cm.CollisionModel(tm.primitives.Box(box_extents=[1, 1, .2]))
        camera_obs.attach_to(base)
        camera_obs.set_pos(np.array([0.1, 0, 1]))
        self._obslist = [camera_obs]

    # x: .2 -- .45
    # y: -.25 -- .25
    # z: .15 -- .3
    # def add_pos(self, action_pos=np.array(np.meshgrid(np.linspace(.2, .45, num=3),
    #                                                   np.linspace(-.25, .25, num=5),
    #                                                   np.linspace(.15, .25, num=2))).T.reshape(-1, 3),
    #             action_center_rotmat=np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T,
    #             rot_range_x=(np.array([1, 0, 0]), [np.radians(angle)
    #                                                for angle in [-25, -15, 0, 15, 25]]),
    #             rot_range_y=(np.array([0, 1, 0]), [np.radians(angle)
    #                                                for angle in [-15, -25, 15, 25]]), ):
    #     work_arm = self._work_arm
    #     robot_s = self._yumi_s
    #     jnt_values_bk = robot_s.get_jnt_values(work_arm)
    #     last_jnt_values = jnt_values_bk
    #     range_axes = [rot_range_x, rot_range_y]
    #     for action_center_pos in action_pos:
    #         for axisid in range(len(range_axes)):
    #             axis = range_axes[axisid][0]
    #             for angle in range_axes[axisid][1]:
    #                 goal_pos = action_center_pos
    #                 goal_rotmat = np.dot(rm.rotmat_from_axangle(axis, angle), action_center_rotmat)
    #                 jnt_values = robot_s.ik(component_name=work_arm,
    #                                         tgt_pos=goal_pos,
    #                                         tgt_rotmat=goal_rotmat,
    #                                         seed_jnt_values=last_jnt_values,
    #                                         max_niter=100)
    #                 if jnt_values is not None:
    #                     robot_s.fk(component_name=work_arm, jnt_values=jnt_values)
    #                     if not robot_s.is_collided():
    #                         last_jnt_values = jnt_values
    #                         self._record_loc.append(jnt_values)
    #     robot_s.fk(work_arm, jnt_values_bk)

    def add_homomats(self, homomats, work_arm="rgt_arm", load=True):
        assert work_arm in ["rgt_arm", "lft_arm"]
        cache_file_name = f"dumped_rbt_poses_{work_arm}.pkl"
        if "rgt" in work_arm:
            record_loc = self._record_loc_rgt
            init_target_pos = np.array([.3, -.1, .15])
            init_target_rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).T
        else:
            record_loc = self._record_loc_lft
            init_target_pos = np.array([.3, .1, .15])
            init_target_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]).T
        if load and fs.Path(cache_file_name).exists():
            record_loc.extend(fs.load_pickle(cache_file_name))
        else:
            robot_s = self._yumi_s
            jnt_values_bk = robot_s.get_jnt_values(work_arm)
            last_jnt_values = jnt_values_bk
            init_jnt_values = robot_s.ik(component_name=work_arm,
                                         tgt_pos=init_target_pos,
                                         tgt_rotmat=init_target_rot,
                                         seed_jnt_values=last_jnt_values,
                                         max_niter=100)
            record_loc.append(init_jnt_values)
            for idx, homomat in enumerate(homomats):
                jnt_values = robot_s.ik(component_name=work_arm,
                                        tgt_pos=homomat[:3, 3],
                                        tgt_rotmat=homomat[:3, :3],
                                        seed_jnt_values=last_jnt_values,
                                        max_niter=100)
                if jnt_values is not None:
                    robot_s.fk(component_name=work_arm, jnt_values=jnt_values)
                    if not robot_s.is_collided():
                        # last_jnt_values = jnt_values
                        record_loc.append(jnt_values)
                print(idx, jnt_values)
            fs.dump_pickle(record_loc, cache_file_name)
            robot_s.fk(work_arm, jnt_values_bk)
        print(f"Feasible num of homomat is {len(record_loc)}")

    def _set_counter(self, val, work_arm="rgt_arm"):
        assert work_arm in ["rgt_arm", "lft_arm"] and isinstance(val, int)
        if "rgt" in work_arm:
            self._counter_rgt = val
        else:
            self._counter_lft = val

    def _get_counter(self, work_arm="rgt_arm") -> int:
        assert work_arm in ["rgt_arm", "lft_arm"]
        if "rgt" in work_arm:
            return self._counter_rgt
        else:
            return self._counter_lft

    def move_next_record_pos(self, work_arm="rgt_arm"):
        if self.yumi_con is None:
            return

        assert work_arm in ["rgt_arm", "lft_arm"]

        if "rgt" in work_arm:
            record_loc = self._record_loc_rgt
        else:
            record_loc = self._record_loc_lft

        if self._get_counter(work_arm) >= len(record_loc):
            print(len(record_loc))
            raise StopIteration

        self._yumi_s.fk("rgt_arm", self.yumi_con.get_jnt_values("rgt_arm"))
        self._yumi_s.fk("lft_arm", self.yumi_con.get_jnt_values("lft_arm"))
        record_jnt_val = record_loc[self._get_counter(work_arm)]
        self._set_counter(self._get_counter(work_arm) + 1, work_arm)
        rrt_path_rgt = self._rrtc_planner.plan(component_name=work_arm,
                                               start_conf=self.yumi_con.get_jnt_values(work_arm),
                                               goal_conf=np.array(record_jnt_val),
                                               obstacle_list=self._obslist,
                                               ext_dist=.01,
                                               max_time=300)
        if rrt_path_rgt is None:
            return
        if len(rrt_path_rgt) < 3:
            print("Move jnts")
            self.yumi_con.move_jnts(component_name=work_arm, jnt_vals=record_jnt_val)
        else:
            print("Move RRT")
            self.yumi_con.move_jntspace_path(component_name=work_arm, path=rrt_path_rgt
                                             , speed_n=300)
        # if self._get_counter(work_arm) % 100 == 0 and self._get_counter(work_arm) != 0:
        #     self.yumi_con.close_gripper(work_arm, force=20)


class AutoLabel:
    def __init__(self, showbase,
                 robot_mover: RobotMover,
                 tube_label="blue tube",
                 streamer=None,
                 affine_matrix=None,
                 work_arm="both",
                 save_path="./",
                 debug_path=None,
                 toggle_debug=False):

        assert work_arm in ["rgt_arm", "lft_arm", "both"]

        if work_arm == "both":
            self._use_both_arm = True
        else:
            self._use_both_arm = False
        self._work_arm = work_arm  # work arm

        # control sensor capture and robot movement
        self._streamer = streamer  # sensor streamer
        self._yumi_con = robot_mover.yumi_con  # robot controller
        self._robot_mover = robot_mover  # robot mover
        self._rbt_tcp_pos = [None, None]  # record robot tcp pos
        self._rbt_tcp_rot = [None, None]  # record robot tcp rotmat

        # params
        self._params = fs.load_json("params/params.json")

        # raw sensor data
        self._pcd = None
        self._img = None
        self._depth_img = None
        # align pcd
        self._affine_mat = affine_matrix
        self._pcd_aligned = None

        # ?
        self._box_label = None
        self._tube_label = tube_label

        # save path for information
        self.save_path = save_path

        # init gui
        self._panel = GuiFrame(showbase.tkRoot, hidden=True)
        self._init_gui()

        # boost base and nodepath of pcd
        self._gray_img_tx = gen_img_texture_on_render(spawn_window(showbase)[0])
        self._depth_img_tx = gen_img_texture_on_render(spawn_window(showbase)[0])
        self._np_pcd = None

        if self._robot_mover is not None:
            self._move_rbt_to_next_acquire_pos(save_data=False)

        if toggle_debug:
            if debug_path is None:
                raise Exception("Define the toggle path first")
            self._pcd, self._img, self._depth_img, self._rbt_tcp_pos, self._rbt_tcp_rot = fs.load_pickle(debug_path)
            if self._affine_mat is not None:
                self._pcd_aligned = rm.homomat_transform_points(self._affine_mat, points=self._pcd)
            self._render_pcd(self._pcd_aligned)
            self._render_gray_img(img_to_n_channel(depth2gray_map(self._depth_img)))
            self._render_depth_img(img_to_n_channel(enhance_gray_img(self._img)))

    def _render_pcd(self, pcd):
        if pcd is None:
            print("Input pcd is None")
            return
        if self._np_pcd is not None:
            self._np_pcd.remove()
        self._np_pcd = gm.gen_pointcloud(pcd)
        self._np_pcd.attach_to(base)

    def _render_gray_img(self, img):
        if img is None:
            print("Input img is None")
            return
        set_img_texture(img, self._gray_img_tx)

    def _render_depth_img(self, img):
        if img is None:
            print("Input img is None")
            return
        set_img_texture(img, self._depth_img_tx)

    def _init_gui(self):
        # init x direction in tcp coordinate
        self._x_bound = np.array([-.1, .1])
        self._y_bound = np.array([-.1, .1])
        self._z_bound = np.array([-.1, .1])
        _row = 1
        self._xrange = [
            self._panel.add_scale(f"x-", default_value=self._params['last_selection_val_x'][0], command=self._update,
                                  val_range=[self._x_bound[0], self._x_bound[1]],
                                  pos=(_row, 0)),
            self._panel.add_scale(f"x+", default_value=self._params['last_selection_val_x'][1], command=self._update,
                                  val_range=[self._x_bound[0], self._x_bound[1]],
                                  pos=(_row, 1))]
        # init y direction in tcp coordinate

        self._yrange = [
            self._panel.add_scale(f"y-", default_value=self._params['last_selection_val_y'][0], command=self._update,
                                  val_range=[self._y_bound[0], self._y_bound[1]],
                                  pos=(_row + 1, 0)),
            self._panel.add_scale(f"y+", default_value=self._params['last_selection_val_y'][1], command=self._update,
                                  val_range=[self._y_bound[0], self._y_bound[1]],
                                  pos=(_row + 1, 1))]
        # init z direction in tcp coordinate

        self._zrange = [
            self._panel.add_scale(f"z-", default_value=self._params['last_selection_val_z'][0], command=self._update,
                                  val_range=[self._z_bound[0], self._z_bound[1]],
                                  pos=(_row + 2, 0)),
            self._panel.add_scale(f"z+", default_value=self._params['last_selection_val_z'][1], command=self._update,
                                  val_range=[self._z_bound[0], self._z_bound[1]],
                                  pos=(_row + 2, 1))]
        _row += 3
        if self._use_both_arm:
            self._xrange += [self._panel.add_scale(f"x-", default_value=self._params['last_selection_val_x'][2],
                                                   command=self._update,
                                                   val_range=[self._x_bound[0], self._x_bound[1]],
                                                   pos=(_row, 2)),
                             self._panel.add_scale(f"x+", default_value=self._params['last_selection_val_x'][3],
                                                   command=self._update,
                                                   val_range=[self._x_bound[0], self._x_bound[1]],
                                                   pos=(_row, 3))]
            self._yrange += [self._panel.add_scale(f"y-", default_value=self._params['last_selection_val_y'][2],
                                                   command=self._update,
                                                   val_range=[self._y_bound[0], self._y_bound[1]],
                                                   pos=(_row + 1, 2)),
                             self._panel.add_scale(f"y+", default_value=self._params['last_selection_val_y'][3],
                                                   command=self._update,
                                                   val_range=[self._y_bound[0], self._y_bound[1]],
                                                   pos=(_row + 1, 3))]
            self._zrange += [self._panel.add_scale(f"z-", default_value=self._params['last_selection_val_z'][2],
                                                   command=self._update,
                                                   val_range=[self._z_bound[0], self._z_bound[1]],
                                                   pos=(_row + 2, 2)),
                             self._panel.add_scale(f"z+", default_value=self._params['last_selection_val_z'][3],
                                                   command=self._update,
                                                   val_range=[self._z_bound[0], self._z_bound[1]],
                                                   pos=(_row + 2, 3))]
            _row += 3

        self._panel.add_button(text="Enter", command=self._sure_init, pos=(_row, 0))
        self._panel.add_button(text="Get data and render", command=self._acquire_data_render, pos=(_row, 1))
        self._panel.add_button(text="Move robot to next position", command=self._move_rbt_to_next_acquire_pos,
                               pos=(_row + 1, 0))
        self._panel.add_button(text="Auto Collect Data", command=self._auto_collect, pos=(_row + 1, 1))
        self._panel.show()

    def _move_rbt_to_next_acquire_pos(self, save_data=True):
        if self._use_both_arm:
            self._robot_mover.move_next_record_pos("rgt_arm")
            self._robot_mover.move_next_record_pos("lft_arm")
        else:
            self._robot_mover.move_next_record_pos(self._work_arm)
        self._box_label = self._acquire_data_render()
        if self._box_label is None:
            return
        if save_data:
            fs.dump_pickle(
                [self._pcd, self._img, self._depth_img, self._rbt_tcp_pos, self._rbt_tcp_rot, self._box_label,
                 self._tube_label],
                self.save_path / f"{strftime('%Y%m%d-%H%M%S')}.pkl")

    def _auto_collect(self):
        def task_auto_collect(t, task):
            try:
                t._move_rbt_to_next_acquire_pos()
            except StopIteration:
                print("Finished")
                return task.done
            except Exception as e:
                # raise Exception(e)
                pass
            return task.again

        print("TASK again")
        taskMgr.doMethodLater(1, task_auto_collect, "autocollect", extraArgs=[self], appendTask=True, priority=999)

    def _acquire_data_render(self):
        self.acquire_data()
        if self._pcd is None or self._depth_img is None or self._img is None:
            return
        if self._affine_mat is not None:
            self._pcd_aligned = rm.homomat_transform_points(self._affine_mat, points=self._pcd)
        return self._update(None)

    def acquire_data(self):
        """
        1. Acquire the raw sensor data: 1)pcd 2)texture 3) depth image
        2. Acquire the robot tcp information
        """
        if self._streamer is None:
            self._pcd, self._img, self._depth_img = None, None, None
            print("Cannot acquire data")
        else:
            self._pcd, self._img, self._depth_img = vision_pipeline(self._streamer)
        if self._yumi_con is None:
            self._rbt_tcp_pos, self._rbt_tcp_rot = [None, None], [None, None]
        else:
            if self._use_both_arm:
                self._rbt_tcp_pos[0], self._rbt_tcp_rot[0] = self._yumi_con.get_pose(component_name="rgt_arm")
                self._rbt_tcp_pos[1], self._rbt_tcp_rot[1] = self._yumi_con.get_pose(component_name="lft_arm")
            else:
                self._rbt_tcp_pos[0], self._rbt_tcp_rot[0] = self._yumi_con.get_pose(component_name=self._work_arm)

    def is_prepared(self):
        """
        Make sure the necessary information about the robot is successfully acquired
        """
        return self._rbt_tcp_pos[0] is not None and self._rbt_tcp_rot[0] is not None and self._pcd is not None \
               and self._img is not None and self._depth_img is not None

    def _update(self, x):
        bad_img = False
        if not self.is_prepared():
            return
        x_range = np.array([_.get() for _ in self._xrange])
        y_range = np.array([_.get() for _ in self._yrange])
        z_range = np.array([_.get() for _ in self._zrange])

        depth_img_labeled = img_to_n_channel(depth2gray_map(self._depth_img))
        gray_img_labeled = img_to_n_channel(enhance_gray_img(self._img))

        labels = []
        self._render_pcd(self._pcd_aligned if self._pcd_aligned is not None else self._pcd)
        for indx in range(len(self._rbt_tcp_pos)):
            if self._rbt_tcp_pos[indx] is None:
                continue
            data = extract_pixel_around_hand(self._rbt_tcp_pos[indx],
                                             self._rbt_tcp_rot[indx],
                                             self._affine_mat,
                                             self._pcd,
                                             img_shape=self._img.shape,
                                             extract_area=(x_range[2 * indx:2 * indx + 2],
                                                           y_range[2 * indx:2 * indx + 2],
                                                           z_range[2 * indx:2 * indx + 2]))
            if data is None:
                bad_img = True
                continue
            h1, w1, h2, w2, extracted_pcd_idx = data
            if len(extracted_pcd_idx) < 400:
                continue
            labels.append([h1, w1, h2, w2, extracted_pcd_idx])
            depth_img_labeled = cv2.rectangle(depth_img_labeled, (w1, h1), (w2, h2),
                                              color=(255, 0, 0), thickness=3)
            gray_img_labeled = cv2.rectangle(gray_img_labeled, (w1, h1), (w2, h2),
                                             color=(255, 0, 0), thickness=3)
            gm.gen_pointcloud(
                self._pcd_aligned[extracted_pcd_idx] if self._pcd_aligned is not None else self._pcd[extracted_pcd_idx],
                rgbas=[[1, 0, 0, 1]]).attach_to(self._np_pcd)
        self._render_depth_img(depth_img_labeled)
        self._render_gray_img(gray_img_labeled)
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

    def _sure_init(self):
        self._panel.on_closing()
        pass

    def _save_params(self):
        print("III RUN")
        x_range = np.array([_.get() for _ in self._xrange])
        y_range = np.array([_.get() for _ in self._yrange])
        z_range = np.array([_.get() for _ in self._zrange])
        for i in range(len(x_range)):
            self._params["last_selection_val_x"][i] = x_range[i]
        for i in range(len(y_range)):
            self._params["last_selection_val_y"][i] = y_range[i]
        for i in range(len(z_range)):
            self._params["last_selection_val_z"][i] = z_range[i]
        fs.dump_json(self._params, "params/params.json", reminder=False)


if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.components.yumi_control.yumi_con import YumiController

    DEBUG = False
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # init the simulation robot
    yumi_s = ym.Yumi(enable_cc=True)
    work_arm = "rgt_arm"
    # # init the real robot and get joint angles
    if not DEBUG:
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
        fs.load_json(AFFINE_MAT_PATH)['affine_mat'])

    rbt_mover = RobotMover(yumi_s=yumi_s, yumi_con=yumi_con, )
    import basis.robot_math as rm

    pos_list_rgt = []
    for pos in np.array(np.meshgrid(np.linspace(.3, .4, num=2),
                                    np.linspace(-.20, -.1, num=2),
                                    np.array([.2]))).T.reshape(-1, 3):
        rots_candidate = np.array(rm.gen_icorotmats(icolevel=3,
                                                    rotation_interval=np.radians(60),
                                                    crop_normal=np.array([0, 0, 1]),
                                                    crop_angle=np.pi / 6,
                                                    toggleflat=True))
        rots_candidate[..., [2, 1]] = rots_candidate[..., [1, 2]]
        rots_candidate[..., 0] = -rots_candidate[..., 0]
        for rot in rots_candidate:
            pos_list_rgt.append(rm.homomat_from_posrot(pos, rot))
    print(f"There are {len(pos_list_rgt)} rgt poses")
    rbt_mover.add_homomats(pos_list_rgt, work_arm="rgt_arm")

    pos_list_lft = []
    for pos in np.array(np.meshgrid(np.linspace(.3, .4, num=2),
                                    np.linspace(.20, .1, num=2),
                                    np.array([.2]))).T.reshape(-1, 3):
        rots_candidate = np.array(rm.gen_icorotmats(icolevel=3,
                                                    rotation_interval=np.radians(60),
                                                    crop_normal=np.array([0, 0, 1]),
                                                    crop_angle=np.pi / 6,
                                                    toggleflat=True))
        rots_candidate[..., [2, 1]] = rots_candidate[..., [1, 2]]
        rots_candidate[..., 1] = -rots_candidate[..., 1]
        for rot in rots_candidate:
            pos_list_lft.append(rm.homomat_from_posrot(pos, rot))
    print(f"There are {len(pos_list_lft)} lft poses")
    rbt_mover.add_homomats(pos_list_lft, work_arm="lft_arm")

    rbt_mover._counter_rgt = 288
    rbt_mover._counter_lft = 288
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
    al = AutoLabel(showbase=base,
                   robot_mover=rbt_mover,
                   tube_label=TUBE_TYPE,
                   affine_matrix=affine_mat,
                   work_arm="both",
                   streamer=SensorMarkerHandler(IP_ADR),  #
                   save_path=SAVE_PATH,
                   debug_path=workdir / "examples/vision/paper/data" / f"20220216-151043.pkl",
                   toggle_debug=DEBUG)

    base.startTk()
    base.tkRoot.withdraw()
    base.run()

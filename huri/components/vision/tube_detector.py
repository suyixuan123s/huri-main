import time

import numpy as np
from typing import Dict, Optional, List
from huri.core.common_import import *
from huri.core.file_sys import load_json
from huri.core.print_tool import text_pd, print_with_border
import cv2 as cv
import basis.robot_math as rm
import huri.components.vision.extract as extract
from huri.definitions.rack_def import TubeRack, TubeLocator, TubeType, TestTube
import huri.core.base_boost as bb

YOLO_VERSION = 8
import huri.vision.yolov5x6.detect as yolo

# if YOLO_VERSION == 5:

if YOLO_VERSION == 8:
    from huri.components.vision.detect import YOLODetector
else:
    raise Exception(f"No support YOLO version {YOLO_VERSION}")

low = np.s_[..., :2]
high = np.s_[..., 2:]


def iou(A, B):
    """
    AB = iou(A[:,None],B[None])
    https://stackoverflow.com/questions/57897578/efficient-way-to-calculate-all-ious-of-two-lists
    :param A:
    :param B:
    :return:
    """
    A, B = A.copy(), B.copy()
    A[high] += 1
    B[high] += 1
    intrs = (np.maximum(0, np.minimum(A[high], B[high])
                        - np.maximum(A[low], B[low]))).prod(-1)
    return intrs / ((A[high] - A[low]).prod(-1) + (B[high] - B[low]).prod(-1) - intrs)


class NoRackException(Exception):
    def __init__(self):
        super().__init__(f"Cannot locate the position of the rack! Detection Failed")


class TestTubeDetector:
    if YOLO_VERSION == 8:
        detector = YOLODetector(fs.workdir.joinpath("components", "exe", "best.pt"))

    def __init__(self, affine_mat_path: Optional[str],
                 use_last_available_rack_yolo_pos=True,
                 rack_height_upper=.05 + .015,
                 rack_height_lower=.02 + .02, ):
        if affine_mat_path is not None:
            self.affine_mat = self._load_affine_mat(affine_mat_path)
        else:
            self.affine_mat = np.eye(4)
        self.use_last_available_rack_yolo_pos = use_last_available_rack_yolo_pos
        self.last_available_rack_yolo_pos = {}

        self.rack_height_upper = rack_height_upper
        self.rack_height_lower = rack_height_lower

        self.rack_tf_init = None

    def _load_affine_mat(self, affine_mat_path: str) -> np.ndarray:
        return np.asarray(load_json(affine_mat_path)["affine_mat"])

    def analyze_scene(self,
                      rack_proto: TubeRack,
                      pcd: np.ndarray,
                      texture_img: np.ndarray,
                      std_out=None,
                      onscreen_img=None,
                      canvas=None,
                      toggle_detect_tube_pos=False,
                      toggle_yolo=False,
                      toggle_info=True,
                      rack_tf_init=None,
                      save_detect=False,
                      toggle_time=False, ) -> (Dict[str, np.ndarray],
                                               TubeRack,
                                               np.ndarray):

        if toggle_time:
            time_1 = time.time()
        rack_instance = rack_proto.copy()
        # dict to restore the label - detected pose
        detected_results = {}
        # yolo detect the scene
        enhanced_image = cv.equalizeHist(texture_img)
        img_shape = enhanced_image.shape
        if YOLO_VERSION == 8:
            yolo_img, yolo_results = self.detector.val(img=np.stack((enhanced_image,) * 3, axis=-1), toggle=toggle_yolo,
                                                       save_detect=save_detect)
        elif YOLO_VERSION == 5:
            yolo_img, yolo_results = self._yolo_detect(texture_img=texture_img, toggle=toggle_yolo)
        # if save_detect:
        #     cv2.imwrite("yolo_tmp.jpg", yolo_img)
        if onscreen_img is not None:
            import huri.components.utils.plot_projection as pp
            canvas = pp.Plot(x_size=500, is_plot_grid=True)

        yolo_results_rack = yolo_results[yolo_results[:, 0] < 1]
        if len(yolo_results_rack) < 1 and self.use_last_available_rack_yolo_pos:
            yolo_results_rack = self.last_available_rack_yolo_pos
        if len(yolo_results_rack) > 0:
            print("RACCCCCCCCK  IN NAME")
            self.last_available_rack_yolo_pos = yolo_results_rack

        # if rack_tf_init is None:
        #     rack_tf_init = self.rack_tf_init

        if toggle_time:
            yolo_time = time.time() - time_1

        # print(self.last_available_rack_yolo_pos)
        # # remove the same
        # tube_centers = np.zeros((len(yolo_results), 2))
        # tube_names = []
        # for _, (_key, _item) in enumerate(yolo_results.items()):
        #     if "rack" in _key:
        #         continue
        #     pos_center = np.sum(_item['pos'], axis=1) / 2
        #     tube_centers[_] = pos_center
        #     tube_names.append(_key)
        # remove_list = []
        # for _ind, _tube_center in enumerate(tube_centers):
        #     r = np.linalg.norm(tube_centers - _tube_center, axis=1)
        #     bad_ind_list = np.where((r > 0) & (r < 3))[0]
        #     t_name = tube_names[_ind]
        #     bad_t_name_list = [tube_names[_] for _ in bad_ind_list if t_name.split(" ")[0] not in tube_names[_]]
        #     for bad_t_name in bad_t_name_list:
        #         if bad_t_name in remove_list:
        #             continue
        #         if get_iou(yolo_results[t_name]["pos"], yolo_results[bad_t_name]["pos"]) < 0.5:
        #             continue
        #         if yolo_results[t_name]["conf"] >= yolo_results[bad_t_name]["conf"]:
        #             print(t_name, bad_t_name)
        #             remove_list.append(bad_t_name)
        #         else:
        #             print(r)
        #             print(t_name, bad_t_name)
        #             remove_list.append(t_name)
        # [yolo_results.pop(_) for _ in remove_list]
        pcd_calibrated = rm.homomat_transform_points(homomat=self.affine_mat, points=pcd)
        # uncomment for debug ###############################
        # gm.gen_pointcloud(pcd_calibrated).attach_to(base)
        # base.run()
        #######################################################
        # detect the rack and plot
        rack_t = time.time()
        rack_tf, rack_pcd = self._detect_rack(pcd_calibrated=pcd_calibrated,
                                              rack_pcd_template=rack_instance._pcd_template,
                                              yolo_results=yolo_results_rack,
                                              img_shape=img_shape,
                                              std_out=std_out)
        print(f'rack detection time is {time.time() - rack_t}')
        if toggle_time:
            rack_detect_time = time.time() - time_1 - yolo_time

        # if rack_tf_init is not None:
        #     if not rm.angle_between_vectors(rack_tf[:3, 0], rack_tf_init[:3, 0]) < np.pi / 3:
        #         print("[Trigger] RRRRRR")
        #         print(rack_tf, rack_tf_init)
        #         rack_tf, rack_pcd = self._detect_rack(pcd_calibrated=pcd_calibrated,
        #                                               rack_pcd_template=rack_instance._pcd_template,
        #                                               yolo_results=yolo_results_rack,
        #                                               img_shape=img_shape,
        #                                               std_out=std_out)
        #
        #
        #         rack_tf[:3, 0] = - rack_tf[:3, 0]
        #         rack_tf[:3, 1] = - rack_tf[:3, 1]
        # self.rack_tf_init = rack_tf.copy()

        # if the rack is not exist, raise an error
        if rack_tf is None:
            # print(len(yolo_results_rack))
            raise NoRackException()
        # else:
        #     print(rack_tf)
        detected_results["rack"] = rack_tf
        # detect the test tubes
        tube_locator = TubeLocator(rack_instance, rack_tf)
        for tube_yolo_info in yolo_results[yolo_results[:, 0] > 0]:
            tube_label = int(tube_yolo_info[0])
            tube_proto = TubeType.gen_tube_by_tubetype(tube_label)
            tube_t = time.time()
            tube_tf, tube_pcd, slot_id, slot_confidence = self._detect_test_tube(tube=tube_proto,
                                                                                 pcd_calibrated=pcd_calibrated,
                                                                                 tube_label=tube_label,
                                                                                 tube_yolo_pos=tube_yolo_info[
                                                                                               1:5].reshape(2, 2),
                                                                                 tube_locator=tube_locator,
                                                                                 img_shape=img_shape,
                                                                                 tube_height_interval=np.array(
                                                                                     [-.02, .02]),
                                                                                 toggle_detect_tube_pos=toggle_detect_tube_pos,
                                                                                 std_out=std_out, canvas=canvas)
            if tube_tf is None:
                print(f'tube detection time is {time.time() - tube_t}')
                continue
            detected_results[tube_label] = tube_tf
            rack_instance.insert_tube(slot_id=slot_id,
                                      tube=tube_proto,
                                      tube_rel_pose=np.dot(rm.homomat_inverse(rack_tf),
                                                           tube_tf) if toggle_detect_tube_pos else None,
                                      confidence=slot_confidence)

        if toggle_time:
            tube_detect_time = time.time() - time_1 - yolo_time - rack_detect_time
            print(f"Time consumption: YOLO time: {yolo_time},"
                  f" Rack Detection: {rack_detect_time},"
                  f"Tube Detection: {tube_detect_time}")

        # show information
        if toggle_info:
            print("---" * 20)
            rack_state_text = text_pd(rack_instance.rack_status)
            print(
                f"Number of tubes detected: {len(rack_instance.rack_status[rack_instance.rack_status > 0])}")
            print_with_border(f"rack status", width=38)
            print(repr(rack_instance.rack_status))
            print(rack_state_text)

            # for coord in np.vstack(np.where(rack_instance.rack_status > 0)).T:
            #     print_with_border(f"{coord} test tube pose", width=20)
            #     print(np.array_str(rack_instance.tubes_pose[tuple(coord)], precision=2, suppress_small=True))
            # print("---" * 20)

            # print(np.array_str(rack_instance.rack_tube_pos, precision=2, suppress_small=True))

        if onscreen_img is not None:
            im = canvas.get_img()
            yolo_img[:im.shape[0], :im.shape[1], :] = im
            onscreen_img.update_img(yolo_img)
            base.graphicsEngine.renderFrame()
            base.graphicsEngine.renderFrame()

        return detected_results, rack_instance, rack_tf, yolo_img

    # def analyze_scene_2(self, pcd: np.ndarray,
    #                     texture_img: np.ndarray,
    #                     std_out=None,
    #                     canvas=None,
    #                     toggle_yolo=False,
    #                     toggle_info=True,
    #                     save_detect=False) -> (Dict[str, np.ndarray],
    #                                            Rack_LightThinner_Proto,
    #                                            np.ndarray):
    #     """
    #     Analyze the scene that may contains two test tube rack
    #     :param pcd:
    #     :param texture_img:
    #     :param std_out:
    #     :param canvas:
    #     :param toggle_yolo:
    #     :param toggle_info:
    #     :param save_detect:
    #     :return:
    #     """
    #     detected_racks = []
    #     # yolo detect the scene
    #     yolo_img, yolo_results = self._yolo_detect(texture_img=texture_img, toggle=toggle_yolo)
    #     if save_detect:
    #         cv2.imwrite("yolo_tmp.jpg", yolo_img)
    #
    #     pcd_calibrated = rm.homomat_transform_points(homomat=self.affine_mat, points=pcd)
    #
    #     yolo_results_rack = {}
    #     for _name in list(yolo_results.keys()):
    #         if "rack" in _name:
    #             yolo_results_rack[_name] = yolo_results.pop(_name)
    #
    #     for item in yolo_results_rack.items():
    #         rack_tf, rack_pcd = self._detect_rack(pcd_calibrated=pcd_calibrated,
    #                                               yolo_results={item[0]: item[1]},
    #                                               img_shape=yolo_img.shape,
    #                                               std_out=std_out)
    #         if rack_tf is None:
    #             # print(len(yolo_results_rack))
    #             continue
    #         rack_light_thinner = Rack_LightThinner_Proto.copy()
    #         # dict to restore the label - detected pose
    #         detected_results = {}
    #         detected_results["rack"] = rack_tf
    #         # detect the test tubes
    #         tube_locator = TubeLocator(rack_light_thinner, rack_tf)
    #         for tube_label, tube_yolo_info in yolo_results.items():
    #             if "rack" in tube_label:
    #                 continue
    #             tube_proto = TubeType.gen_tube_by_name(tube_label)
    #             tube_tf, tube_pcd, slot_id, slot_confidence = self._detect_test_tube(tube=tube_proto,
    #                                                                                  pcd_calibrated=pcd_calibrated,
    #                                                                                  tube_label=tube_label,
    #                                                                                  tube_yolo_pos=tube_yolo_info[
    #                                                                                      'pos'],
    #                                                                                  tube_locator=tube_locator,
    #                                                                                  img_shape=yolo_img.shape,
    #                                                                                  tube_height_interval=np.array(
    #                                                                                      [-.02, .02]),
    #                                                                                  std_out=std_out, canvas=canvas)
    #             if tube_tf is None:
    #                 continue
    #             detected_results[tube_label] = tube_tf
    #             rack_light_thinner.insert_tube(slot_id=slot_id,
    #                                            tube=tube_proto,
    #                                            # tube_rel_pose=np.dot(rm.homomat_inverse(rack_tf), tube_tf),
    #                                            confidence=slot_confidence)
    #         detected_racks.append([detected_results, rack_light_thinner, rack_tf])
    #     return detected_racks

    def analyze_tubes_given_rack_tf_yolo(self,
                                         rack_proto: TubeRack,
                                         rack_tf: np.ndarray,
                                         pcd: np.ndarray,
                                         yolo_results: np.ndarray,
                                         yolo_img: np.ndarray,
                                         std_out=None,
                                         onscreen_img=None,
                                         canvas=None,
                                         toggle_detect_tube_pos=False,
                                         toggle_info=True,
                                         downsampling_voxelsize=.002,
                                         save_detect=False) -> (Dict[str, np.ndarray],
                                                                TubeRack,
                                                                np.ndarray):
        rack_instance = rack_proto.copy()

        # restore undetected yolo results
        detected_yolo_results_id = []

        pcd_calibrated = rm.homomat_transform_points(homomat=self.affine_mat, points=pcd)
        # if the rack is not exist, raise an error
        if rack_tf is None:
            # print(len(yolo_results_rack))
            raise NoRackException()
        # detect the test tubes
        tube_locator = TubeLocator(rack_instance, rack_tf)
        for tube_yolo_id, tube_yolo_info in enumerate(yolo_results):
            if int(tube_yolo_info[0]) == 0:
                continue
            tube_label = int(tube_yolo_info[0])
            # _toggle_detect_tube_pos = True if tube_label == 1 else toggle_detect_tube_pos
            _toggle_detect_tube_pos = toggle_detect_tube_pos
            # print(tube_label, _toggle_detect_tube_pos)
            tube_proto = TubeType.gen_tube_by_tubetype(tube_label)
            tube_tf, tube_pcd, slot_id, slot_confidence = self._detect_test_tube(tube=tube_proto,
                                                                                 pcd_calibrated=pcd_calibrated,
                                                                                 tube_label=tube_label,
                                                                                 tube_yolo_pos=tube_yolo_info[
                                                                                               1:5].reshape(2, 2),
                                                                                 tube_locator=tube_locator,
                                                                                 img_shape=yolo_img.shape,
                                                                                 tube_height_interval=np.array(
                                                                                     [-.008, .008]),
                                                                                 toggle_detect_tube_pos=_toggle_detect_tube_pos,
                                                                                 downsampling_voxelsize=downsampling_voxelsize,
                                                                                 std_out=std_out, canvas=canvas)
            if tube_tf is None or slot_confidence < .1:
                continue
            detected_yolo_results_id.append(tube_yolo_id)
            rack_instance.insert_tube(slot_id=slot_id,
                                      tube=tube_proto,
                                      tube_rel_pose=np.dot(rm.homomat_inverse(rack_tf),
                                                           tube_tf) if _toggle_detect_tube_pos else None,
                                      confidence=slot_confidence)

        # show information
        if toggle_info:
            print("---" * 20)
            rack_state_text = text_pd(rack_instance.rack_status)
            print(
                f"Number of tubes detected: {len(rack_instance.rack_status[rack_instance.rack_status > 0])}")
            print_with_border(f"rack status", width=38)
            print(repr(rack_instance.rack_status))
            print(rack_state_text)

            # for coord in np.vstack(np.where(rack_instance.rack_status > 0)).T:
            #     print_with_border(f"{coord} test tube pose", width=20)
            #     print(np.array_str(rack_instance.tubes_pose[tuple(coord)], precision=2, suppress_small=True))
            # print("---" * 20)

            # print(np.array_str(rack_instance.rack_tube_pos, precision=2, suppress_small=True))

        if onscreen_img is not None:
            im = canvas.get_img()
            yolo_img[:im.shape[0], :im.shape[1], :] = im
            onscreen_img.update_img(yolo_img)
            base.graphicsEngine.renderFrame()
            base.graphicsEngine.renderFrame()

        return np.delete(yolo_results, detected_yolo_results_id, axis=0), rack_instance, rack_tf

    def yolo_detect(self, texture_img: np.ndarray, yolo_weights_path: str, imgsz=(1376, 1376), detect_rack=False,
                    toggle=False):
        yolo_img, yolo_results = yolo.detect(source=texture_img,
                                             weights=yolo_weights_path,
                                             cache_model=True,
                                             imgsz=imgsz, )
        if not detect_rack:
            for i in list(yolo_results.keys()):
                if "rack" in i:
                    del yolo_results[i]
        result = np.ones((len(yolo_results), 6)) * -1
        cnt = 0
        for key, v in yolo_results.items():
            tube_id = TubeType.get_tubetype_by_name(key)
            result[cnt] = np.array([tube_id, *v['pos'].reshape(-1), v['conf']])
            cnt += 1
        if toggle:
            # cv.imshow("YOLO Detection Result", cv2.resize(yolo_img, (480, 320)))
            cv.imshow("YOLO Detection Result", yolo_img)
            cv.waitKey(0)
        return yolo_img, result

    def _yolo_detect(self, texture_img: np.ndarray, toggle=False) -> (np.ndarray, Dict):
        enhanced_image = cv.equalizeHist(texture_img)
        # cv2.imshow("original image", enhanced_image)
        # cv2.waitKey(0)

        yolo_img_1, yolo_result_1 = yolo.detect(
            # weights=fs.workdir_vision.joinpath("yolov6", "weights", "ral2022", "syn_800_real_1600.pt"),
            source=np.stack((enhanced_image,) * 3, axis=-1),
            cache_model=True)

        yolo_img_2, yolo_result_2 = yolo.detect(
            # weights=fs.workdir_vision.joinpath("yolov6", "weights", "ral2022", "syn_800_real_1600.pt"),
            weights=fs.workdir_vision.joinpath("yolov5x6", "T30_t080.pt"),
            source=np.stack((enhanced_image,) * 3, axis=-1),
            cache_model=True, model_id=1)

        result_2 = np.ones((len(yolo_result_2), 6)) * -1
        cnt = 0
        blue_id = TubeType.get_tubetype_by_name("blue")
        white_id = TubeType.get_tubetype_by_name("white")
        purple_id = TubeType.get_tubetype_by_name("purple")
        purple_ring_id = TubeType.get_tubetype_by_name("purple ring")
        for key, v in yolo_result_2.items():
            if "purple ring" in key:
                result_2[cnt] = np.array([purple_ring_id, *v['pos'].reshape(-1), v['conf']])
                cnt += 1
                continue
            if "blue" in key:
                result_2[cnt] = np.array([blue_id, *v['pos'].reshape(-1), v['conf']])
            if "white" in key:
                result_2[cnt] = np.array([white_id, *v['pos'].reshape(-1), v['conf']])
            if "purple" in key:
                result_2[cnt] = np.array([purple_id, *v['pos'].reshape(-1), v['conf']])
            cnt += 1

        result_1 = np.ones((len(yolo_result_1), 6)) * -1
        cnt = 0
        for key, v in yolo_result_1.items():
            if "rack" in key:
                result_1[cnt] = np.array([0, *v['pos'].reshape(-1), v['conf']])
            if "blue" in key:
                continue  # # TODO for demo only
                result_1[cnt] = np.array([blue_id, *v['pos'].reshape(-1), v['conf']])
            if "white" in key:
                # result_1[cnt] = np.array([white_id, *v['pos'].reshape(-1), v['conf']])
                # # TODO for demo only
                continue
                result_1[cnt] = np.array([purple_ring_id, *v['pos'].reshape(-1), v['conf']])
            if "purple" in key:
                # # TODO for demo only
                continue
                result_1[cnt] = np.array([purple_id, *v['pos'].reshape(-1), v['conf']])
            cnt += 1
        r_12 = iou(result_1[:, 1:5][:, None], result_2[:, 1:5][None])
        # r_2 = np.sum(r_12, axis=1)
        r_1 = np.sum(r_12, axis=0)
        # miss_r2 = np.where(r_2 < .3)[0]
        repeat_r1 = np.where(r_1 > 1)[0]
        miss_r1 = np.where(r_1 < .3)[0]
        new_result = result_1.copy()
        delete_row = []
        # remove repeated
        if len(repeat_r1) > 0:
            for n, _ in enumerate(r_12[:, repeat_r1].T):
                repeat_id = np.where(_ > .5)[0]
                if len(repeat_id) <= 1:
                    continue
                else:
                    r2_id = result_2[repeat_r1[n]][0]
                    r1_id = result_1[repeat_id][:, 0]
                    r2_r1_check = r2_id != r1_id
                    if np.sum(r2_r1_check) < 0:
                        continue
                    else:
                        delete_row.extend(repeat_id[r2_r1_check].tolist())
        new_result = np.delete(new_result, list(set(delete_row)), 0)
        if len(miss_r1) > 0:
            new_result = np.vstack((new_result, result_2[miss_r1]))

        if toggle:
            # cv.imshow("YOLO Detection Result", cv2.resize(yolo_img, (480, 320)))
            cv.imshow("YOLO Detection Result _ manually labeled", yolo_img_2)
            cv.imshow("YOLO Detection Result _ 1", yolo_img_1)
            cv.waitKey(0)
        return yolo_img_2, new_result

    def _detect_rack(self, pcd_calibrated: np.ndarray,
                     rack_pcd_template: np.ndarray,
                     yolo_results: Dict[str, np.ndarray],
                     img_shape: np.ndarray,
                     std_out=None) -> (Optional[np.ndarray],
                                       Optional[np.ndarray]):
        # TODO extract the test tube through its height
        # TODO remove the unrecognized test tube

        # --- new setup ---
        rack_trans, rack_pcd, outliner = extract.extrack_rack(pcd=pcd_calibrated,
                                                              rack_pcd_template=rack_pcd_template,
                                                              results=yolo_results,
                                                              img_shape=img_shape,
                                                              height_lower=self.rack_height_lower,
                                                              height_upper=self.rack_height_upper,
                                                              std_out=std_out)
        if rack_trans is None:
            return None, None
        # plot
        # if std_out is not None:
        #     rack_pcd_gm = gm.gen_pointcloud(points=rack_pcd,
        #                                     rgbas=[[0, 0, 0, 1]])
        #     std_out.attach(node=rack_pcd_gm, name="rack pcd gm")
        #     # rack_mdl = Rack_LightThinner_Proto.gen_geo_model()
        #     rack_mdl = Rack_Hard_Proto.gen_mesh_model()
        #     rack_mdl.set_homomat(rack_trans)
        #     # rack_mdl.set_pos(rack_mdl.get_pos() + np.array([0, 0, .057]))
        #     # rack_mdl.show_localframe()
        #     std_out.attach(node=rack_mdl, name="rack model")
        #     # rack_mdl.show_localframe()
        #     # base.run()
        return rack_trans, rack_pcd

    def _detect_test_tube(self,
                          tube: TestTube,
                          pcd_calibrated: np.ndarray,
                          tube_label: int,
                          tube_yolo_pos: List,
                          img_shape: np.ndarray,
                          tube_locator: TubeLocator,
                          tube_height_interval: np.ndarray,
                          downsampling_voxelsize: float = 0.002,
                          toggle_detect_tube_pos: bool = False,
                          std_out=None, canvas=None) -> (Optional[np.ndarray],
                                                         Optional[np.ndarray],
                                                         Optional[np.ndarray]):
        lt_pos, rb_pos = tube_yolo_pos
        tube_pcd_yolo = extract.extrack_tube(pcd=pcd_calibrated,
                                             lt_pos=lt_pos,
                                             rb_pos=rb_pos,
                                             img_shape=img_shape)

        # if toggle_detect_tube_pos:
        # gm.gen_pointcloud(tube_pcd_yolo, rgbas=[[1, 0, 0, 1]], pntsize=5).attach_to(base)
        # base.run()

        tube_pcd_yolo_racktf = rm.homomat_transform_points(homomat=rm.homomat_inverse(homomat=tube_locator.rack_tf),
                                                           points=tube_pcd_yolo)
        tube_pcd_racktf = tube_pcd_yolo_racktf[
            (tube_pcd_yolo_racktf[:, 2] > tube.height + tube_height_interval[0]) & (
                    tube_pcd_yolo_racktf[:, 2] < tube.height + tube_height_interval[1])]

        # gm.gen_pointcloud(tube_pcd_racktf, rgbas=[[1, 0, 0, 1]], pntsize=5).attach_to(base)
        # base.run()
        if downsampling_voxelsize is not None:
            tube_pcd_racktf = extract.tube_rm_outlier(pcd_tube=tube_pcd_racktf,
                                                      downsampling_voxelsize=downsampling_voxelsize)

        if tube_pcd_racktf is None:
            print("Not tube detect")
            return None, None, None, None
        # print(len(tube_pcd_racktf))
        # import basis.trimesh as trimesh
        # gm.gen_pointcloud(tube_pcd_racktf,[rm.random_rgba()]).attach_to(base)
        # ext_pcd = tube_pcd_racktf
        # x_range =  [ext_pcd[:, 0].min(), ext_pcd[:, 0].max()]
        # y_range =  [ext_pcd[:, 1].min(), ext_pcd[:, 1].max()]
        # z_range =   [ext_pcd[:, 2].min(), ext_pcd[:, 2].max()]
        # extract_region = trimesh.primitives.Box(
        #     box_extents=[(x_range[1] - x_range[0]), (y_range[1] - y_range[0]), (z_range[1] - z_range[0]), ],
        #     box_center=[(x_range[1] + x_range[0]) / 2, (y_range[1] + y_range[0]) / 2,
        #                 (z_range[1] + z_range[0]) / 2, ])
        # bx_gm = gm.GeometricModel(extract_region)
        # bx_gm.set_rgba([1, 0, 0, .3])
        # # bx_gm.set_homomat(origin_frame)
        # bx_gm.attach_to(base)
        tube_trans, hole_coord, slot_confidence = tube_locator.locate_tube_coord_from_pcd(
            tube_pcd_racktf=tube_pcd_racktf,
            tube_label=tube_label,
            detect_pos=toggle_detect_tube_pos,
            canvas=canvas)
        # filter the confident
        if slot_confidence is None or slot_confidence < .1:  # this filter is only target for the view multiple times
            return None, None, None, None

        # print("confident is ", slot_confidence, np.asarray(hole_coord))

        if tube_trans is None:
            return None, None, None, None
        # plot
        if std_out is not None:
            tube_pcd_gm = gm.gen_pointcloud(points=rm.homomat_transform_points(homomat=tube_locator.rack_tf,
                                                                               points=tube_pcd_racktf),
                                            rgbas=[[0, 0, 0, 1]])
            tube_trans_offset = tube_trans.copy()
            tube_trans_offset[:3, 3] += tube.height * tube_trans_offset[:3, 2]
            # gm.gen_box([.02,.02,.02],homomat=tube_trans_offset,rgba=tube.color).attach_to(base)
            std_out.attach(node=tube_pcd_gm, name="tube pcd gm")
            tube_mdl = tube.gen_mesh_model()
            tube_mdl.set_homomat(tube_trans)
            std_out.attach(node=tube_mdl, name="tube model")
        return tube_trans, tube_pcd_racktf, np.asarray(hole_coord), slot_confidence


if __name__ == "__main__":
    from huri.core.file_sys import workdir, load_pickle, dump_json
    from huri.components.pipeline.data_pipeline import RenderController

    detector = TestTubeDetector(workdir / "data/calibration/affine_mat.json")
    # load test data
    filename = workdir / "test" / "vision" / "data" / "white_cap"
    _, _, img, _, pcd = load_pickle(filename)
    # cv.imshow("test", img)
    # cv.waitKey(0)
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    std_out = RenderController(base.tkRoot, base)
    detected_test_tube, rack = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0], std_out=std_out)

    base.startTk()
    base.tkRoot.withdraw()

    base.run()

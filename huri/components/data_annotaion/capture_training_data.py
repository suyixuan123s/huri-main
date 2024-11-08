import time

import cv2

from huri.core.constants import SENSOR_INFO
import huri.core.file_sys as fs
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler, img_pipeline
from huri.vision.yolov6.detect import detect
from huri.paper_draw.tase2022.data_syn.copy_paste import box_iou_yolo, write_yolo_annotation_file

import numpy as np


def check_id(name, label_info={
    'rack': 0, 'blue': 1, 'purple': 2, 'white': 3, 'purple ring': 4, 'white cap small': 5}):
    for _n in label_info.keys():
        if _n == name:
            return label_info[_n]


def check_ious(n, e, e_confident):
    ious = box_iou_yolo(n, e)
    id_ = np.where(ious > .5)


def test(save_data_path=None):
    if save_data_path is not None:
        save_data_path = fs.Path(save_data_path)
    save_img_path = save_data_path.joinpath("img")
    save_label_path = save_data_path.joinpath("label")
    if not save_img_path.exists():
        save_img_path.mkdir()
    if not save_label_path.exists():
        save_label_path.mkdir()
    streamer = SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG)
    while True:
        # Capture img
        if save_data_path is not None:
            data_name = time.strftime('%Y%m%d-%H%M%S')
            file_name = save_data_path.joinpath(f"{data_name}.pkl")
        else:
            file_name = None
        # pcd, img, _, _, _ = vision_pipeline(streamer=SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG),
        #                                     dump_path=None, rgb_texture=False)
        img = img_pipeline(streamer=streamer)
        # Increase Brightness
        enhanced_image = cv2.equalizeHist(img)
        # Show
        # _image = cv2.putText(img, str("a: New Image || q: Quit"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 6,
        #                      (100, 200, 200), 4)

        yolo_img, yolo_results = detect(
            # imgsz=(1280, 1280),
            weights=fs.workdir_vision.joinpath("yolov6", "weights", "ral2022", "syn_800_real_1600.pt"),
            source=cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR),
            device="cpu",
            conf_thres=.7,
            visualize=False)
        print(yolo_results)
        cv2.imshow(f"YOLO detection result.", yolo_img)
        img_shape = enhanced_image.shape
        x_px, y_px = img_shape[1], img_shape[0]
        if save_label_path is not None:
            tube_id_lists = []
            e_bboxes_xywh = []
            e_confidents = []
            for key, item in yolo_results.items():
                tube_id = check_id(" ".join(key.lstrip().split(" ")[:-1]))
                pos = item['pos']
                _x, _y = pos[0][0] / x_px, pos[0][1] / y_px
                wh = (pos[1] - pos[0])
                _w, _h = wh[0] / x_px, wh[1] / y_px

                b_xywh = np.array([_x + _w / 2, _y + _h / 2, _w, _h])

                if len(e_bboxes_xywh) < 1:
                    pass
                else:
                    ious = box_iou_yolo(b_xywh.reshape(-1, 4), np.array(e_bboxes_xywh))
                    overlap_id = np.where(ious >= .5)
                    if len(overlap_id[0]) > 0:
                        print("! overlap happens")
                        continue

                tube_id_lists.append(tube_id)
                e_bboxes_xywh.append(b_xywh)
                e_confidents.append(item['conf'])
            write_yolo_annotation_file(tube_id_lists, np.array(e_bboxes_xywh),
                                       str(save_label_path / f"{data_name}.txt"))

        cv2.imwrite(str(save_img_path / f"{data_name}.jpg"), enhanced_image)

        k = cv2.waitKey(0)
        if k == ord("a"):  # key "a" reduce -1
            if file_name is not None and file_name.is_file() and file_name.exists():
                file_name.unlink()
            continue
        elif k == ord("q"):
            exit(0)
        else:
            pass


if __name__ == "__main__":
    test(save_data_path=fs.workdir_data.joinpath("data_annotation", "EXP", "TEST_DATA"))

from huri.core.common_import import *
import huri.vision.yolov6.detect as yyd
import cv2 as cv
from time import strftime
from pathlib import Path
from huri.vision.phoxi_capture import vision_read_data
from tqdm import tqdm
from huri.core.constants import ANNOTATION_0_0_2

ANN_FORMAT = ANNOTATION_0_0_2.IN_HAND_ANNOTATION_SAVE_FORMAT


def to_path(p):
    if isinstance(p, str):
        return Path(p)
    elif isinstance(p, Path):
        return p
    else:
        raise Exception("Unsupport path format")


def check_id(name, label_info):
    for _n in label_info.keys():
        if _n in name:
            return label_info[_n]


def auto_labeling(yolo_weight_path,
                  file_dir: str,
                  label_info: dict,
                  save_img_path: str = None,
                  save_label_path: str = None,
                  is_gray_img=True,
                  toggle_debug=False, ):
    file_dir = to_path(file_dir)
    if save_img_path is not None:
        save_img_path = to_path(save_img_path)

    if save_label_path is not None:
        save_label_path = to_path(save_label_path)

    sampled_file_paths = list(file_dir.glob("*"))
    cv2_img_code = cv2.IMREAD_GRAYSCALE if is_gray_img else cv2.IMREAD_COLOR
    for idx, file_path in enumerate(tqdm(sampled_file_paths, desc='Scanning images', total=len(sampled_file_paths))):

        if file_path.name.lower().endswith(('pkl', 'pickle')):
            data = ANN_FORMAT(*fs.load_pickle(file_path))
            img = data.color_img
        else:
            print("Unsupport format")
            continue
        yolo_img, yolo_results = yyd.detect(source=img, weights=yolo_weight_path, cache_model=True)
        # img_name = strftime('%Y%m%d-%H%M%S')
        img_name = file_path.name.split(".")[0]
        img_shape = img.shape
        x_px, y_px = img_shape[1], img_shape[0]
        if save_label_path is not None:
            if save_img_path is not None:
                label_name = img_name
            else:
                label_name = file_path.name.split(".")[0]
            with open(str(save_label_path / f"{label_name}.txt"), "w") as f:
                for key, item in yolo_results.items():
                    tube_id = label_info[" ".join(key.split(" ")[:-1])]
                    pos = item['pos']
                    _x, _y = pos[0][0] / x_px, pos[0][1] / y_px
                    wh = (pos[1] - pos[0])
                    _w, _h = wh[0] / x_px, wh[1] / y_px
                    f.writelines(f"{tube_id} {_x + _w / 2} {_y + _h / 2} {_w} {_h}\n")
        if save_img_path is not None:
            cv.imwrite(str(save_img_path / f"{img_name}.jpg"), img)

        if toggle_debug:
            cv2.imshow(f"YOLO detection result.", yolo_img)
            cv2.waitKey(0)


if __name__ == "__main__":
    # setup the file sampled
    # file_dir = fs.workdir / "data" / "vision_exp" / "exp_20211222193921"
    # file_dir = "D:\chen\yolo_huri\imgs\imgs2"
    file_dir = "D:\\chen\\huri_shared\\huri\\work\\data_annotation\\data"
    # save_img_path = fs.workdir / "data" / "training_data" / "train_imgs"
    # save_label_path = fs.workdir / "data" / "training_data" / "labels"
    # save_img_path = "D:\chen\yolo_huri\\training_data\images\\val"
    # save_img_path = None
    save_img_path = "D:\chen\huri_shared\huri\work\data_annotation\img_annotation\image"
    # save_label_path = "D:\chen\yolo_huri\\training_data\labels\\val"
    save_label_path = "D:\chen\huri_shared\huri\work\data_annotation\img_annotation\label"
    # save_label_path = None
    # label_info = {
    #     "blue cap": 0,
    #     "rack": 1,
    #     "purple cap": 2,
    #     "white cap": 3,
    #     "red cap": 4,
    #     "purple ring": 5,
    #     "white cap small": 6,
    # }

    label_info = {
        'rack 1': 0, 'blue cap': 1, 'purple cap': 2, 'white cap': 3, 'purple ring': 4, 'white cap small': 5
    }

    label_info = {
        'rack': 0, 'blue': 1, 'purple': 2, 'white': 3, 'purple ring': 4, 'white cap small': 5
    }

    auto_labeling(yolo_weight_path="D:\\chen\\huri_shared\\huri\\work\\best.pt",
                  file_dir=file_dir,
                  label_info=label_info,
                  save_img_path=save_img_path,
                  save_label_path=save_label_path,
                  is_gray_img=False,
                  toggle_debug=False)

from pathlib import Path

from tqdm import tqdm

from constant import ANNOTATION, DATA_ANNOT_PATH
from huri.core.common_import import *

ANNOTAION_FORMAT = ANNOTATION.IN_HAND_ANNOTATION_SAVE_FORMAT
LABEL = ANNOTATION.LABEL
BBOX = ANNOTATION.BBOX_XYXY


def to_path(p):
    if isinstance(p, str):
        return Path(p)
    elif isinstance(p, Path):
        return p
    else:
        raise Exception("Unsupport path format")


def boudingbox2yololabel(size, box: np.ndarray):
    """
    size: the size of the image (w, h)
    box: the bounding box contains four values (x_min, x_max, y_min, y_max)
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def output_annotation(file_dir: str or list,
                      label_id: int,
                      save_img_path: str or fs.Path = None,
                      save_label_path: str or fs.Path = None, ):
    if save_img_path is not None:
        save_img_path = to_path(save_img_path)
    if save_label_path is not None:
        save_label_path = to_path(save_label_path)
    if isinstance(file_dir, str) or isinstance(file_dir, fs.Path):
        file_dir = to_path(file_dir)
        sampled_file_paths = list(file_dir.glob("**/*"))
    else:
        sampled_file_paths = file_dir
    for idx, file_path in enumerate(tqdm(sampled_file_paths, desc='Scanning images', total=len(sampled_file_paths))):
        print(f"The {idx + 1} sample: {file_path.name.lower()}")
        if file_path.name.lower().endswith(('pkl', 'pickle')):
            try:
                data = fs.load_pickle(file_path)
            except:
                print("eorrr ")
                continue
            annotations = ANNOTAION_FORMAT(*data)
            saved_img = cv2.equalizeHist(annotations.gray_img)
            img_name = file_path.name.split(".")[0]
            img_shape = saved_img.shape
            x_px, y_px = img_shape[1], img_shape[0]
            if save_label_path is not None:
                if save_img_path is not None:
                    label_name = img_name
                else:
                    label_name = file_path.name.split(".")[0]
                with open(str(save_label_path / f"{label_name}.txt"), "w") as f:
                    for box_label in annotations.annotations:
                        box_label = LABEL(*box_label)
                        bbox = BBOX(*box_label.bboxes)
                        _x, _y, _w, _h = boudingbox2yololabel(size=(x_px, y_px),
                                                              box=np.array([bbox.w1, bbox.w2, bbox.h1, bbox.h2]))
                        f.writelines(f"{label_id} {_x} {_y} {_w} {_h}\n")
            if save_img_path is not None:
                cv2.imwrite(str(save_img_path / f"{img_name}.jpg"), saved_img)


if __name__ == "__main__":
    from constant import SAVE_PATH, TUBE_ID

    save_data_path = r'G:\dataset\08222023'

    # create save_image and save_label path
    save_img_path = fs.Path(f'{save_data_path}').joinpath('images')
    save_label_path = fs.Path(f'{save_data_path}').joinpath('labels')
    save_img_path.mkdir(parents=True, exist_ok=True)
    save_label_path.mkdir(parents=True, exist_ok=True)

    output_annotation(file_dir=SAVE_PATH,
                      label_id=TUBE_ID,
                      save_img_path=save_img_path,
                      save_label_path=save_label_path, )

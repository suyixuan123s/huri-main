import os
import requests
from ultralytics import YOLO
import numpy as np
import importlib
import cv2
from ultralytics.utils.plotting import Annotator
from tqdm import tqdm

COLOR_LIST_RGB = [
    (255, 87, 51),  # Reddish
    (51, 255, 87),  # Greenish
    (51, 87, 255),  # Blueish
    (240, 51, 255),  # Purple
    (255, 195, 0),  # Yellow
    # You can add more colors as needed
]


def download_file(url, weight_path):
    """
    Download a file from a URL if the file doesn't already exist.

    Args:
    - url (str): The URL to download the file from.
    - filename (str): The local filename to save the downloaded file to.

    Returns:
    - bool: True if the file was downloaded, False if the file already existed.
    """
    if os.path.exists(weight_path):
        print(f"{weight_path} already exists. No need to download.")
        return False
    else:
        print(f"Downloading {weight_path}...")
        try:
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Raises a HTTPError if the response was an error

            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 Kilobyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

            with open(weight_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()

            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")
                return False

            print(f"{weight_path} has been downloaded.")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {weight_path}: {e}")
            return False


# Example usage
WEIGHT_URL = 'https://chenhao.info/research/best.pt'

class YOLODetector(object):
    model = None

    def __init__(self, weight_path=None):
        if weight_path is not None:
            # examine if the weight_path exists else download the weight_path from the internet
            download_file(url=WEIGHT_URL, weight_path=weight_path)
            self.model = YOLO(weight_path)
        assert self.model is not None, "Please indicate the model for the YOLO detector"

    # 1367
    def val(self, img, imgsz=1399, toggle=True, save_detect=False):
        img_shape = img.shape
        if len(img_shape) == 2 or (len(img_shape) == 3 and img_shape[2] == 1):
            img = np.stack((img,) * 3, axis=-1)
        result = self.model(img, save=False, imgsz=imgsz, conf=.6, max_det=51, show_labels=False, show_conf=False)
        img_tmp = result[0].plot(line_width=2)
        if toggle or save_detect:

            if save_detect:
                result[0].names[1] = 'cap 1'
                result[0].names[2] = 'cap 2'
                cv2.imwrite("img_tmp.jpg", img_tmp)
            if toggle:
                cv2.imshow("TEST", img_tmp)
                cv2.waitKey(1)
        # annotator = Annotator(img)
        # for label in result.boxes.data:
        #     annotator.box_label(
        #         label[0:4],
        #         f"{result.names[label[-1].item()]} {round(label[-2], 2)}",
        #         COLOR_LIST_RGB[label[-1].item()])

        labels = result[0].boxes.cls
        bboxes = result[0].boxes.xyxy
        if len(labels) > 0:
            r = np.hstack((labels[..., None].cpu().numpy(), bboxes.cpu().numpy()))
        else:
            r = np.array([])
        ## ---
        r[(r[:, 0] == 2), 0] = 6
        # r[(r[:, 0] == 4), 0] = 2
        # r[(r[:, 0] == 5), 0] = 4
        r[(r[:, 0] == 4), 0] = 2
        r[(r[:, 0] == 6), 0] = 4
        return img_tmp, r

import numpy as np
from huri.core.file_sys import workdir
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
import cv2
import huri.vision.yolov6.detect as yyd
import matplotlib as mpl
import time

mpl.use('TkAgg')

IP_ADR = "192.168.125.100:18300"
SAVE_PATH = workdir / "data" / "vision_exp" / f"{time.strftime('%Y%m%d-%H%M%S')}.pkl"


def test():
    while True:
        # Capture img
        pcd, img = vision_pipeline(streamer=SensorMarkerHandler(IP_ADR),
                                   dump_path=SAVE_PATH)
        # Increase Brightness
        enhanced_image = cv2.equalizeHist(img)
        # YOLO results
        yolo_img, yolo_results = yyd.detect(source=np.stack((enhanced_image,) * 3, axis=-1))
        # Show
        _image = cv2.putText(yolo_img, str("a: New Image || q: Quit"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 6,
                             (100, 200, 200), 4)
        cv2.imshow(f"YOLO detection result.", _image)
        # Keyboard Input
        k = cv2.waitKey(0)
        if k == ord("a"):  # key "a" reduce -1
            continue
        elif k == ord("q"):
            exit(0)
        else:
            pass


if __name__ == "__main__":
    # test()
    import huri.core.file_sys as fs
    img = cv2.imread("test.bmp")
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # yolo_img, yolo_results = yyd.detect(source=np.stack((img_g,) * 3, axis=-1), weights=fs.workdir_vision.joinpath("yolov6", "baseline.pt"))
    yolo_img, yolo_results = yyd.detect(source=np.stack((img_g,) * 3, axis=-1),
                                        weights=fs.workdir_vision.joinpath("yolov6", "weights", "ral2022",
                                                                           "syn_800_real_1600.pt"))
    cv2.imshow(f"YOLO detection result.", yolo_img)
    cv2.waitKey(0)

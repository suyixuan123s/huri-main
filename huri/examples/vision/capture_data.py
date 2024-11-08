from huri.core.constants import SENSOR_INFO
import huri.core.file_sys as fs
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
import cv2
import time


def test(save_data_path=None):
    if save_data_path is not None:
        save_data_path = fs.Path(save_data_path)
    while True:
        # Capture img
        if save_data_path is not None:
            file_name = save_data_path.joinpath(f"{time.strftime('%Y%m%d-%H%M%S')}.pkl")
        else:
            file_name = None
        pcd, img, _, _, _ = vision_pipeline(streamer=SensorMarkerHandler(SENSOR_INFO.IP_ADR_DEPTH_SENSOR_DEBUG),
                                            dump_path=file_name, rgb_texture=False)
        # Increase Brightness
        enhanced_image = cv2.equalizeHist(img)
        # Show
        # _image = cv2.putText(img, str("a: New Image || q: Quit"), (10, 100), cv2.FONT_HERSHEY_PLAIN, 6,
        #                      (100, 200, 200), 4)
        cv2.imshow(f"YOLO detection result.", enhanced_image)
        # Keyboard Input
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
    test(save_data_path=fs.workdir_data.joinpath("vision_exp"))

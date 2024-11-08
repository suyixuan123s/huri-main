import cv2


class ExtCam(object):

    def __init__(self):
        self.camcapid = 0
        self.camcap = cv2.VideoCapture(self.camcapid, cv2.CAP_DSHOW)
        self.camcap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.camcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        # self.camcap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        # self.camcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camcap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.camcap.set(cv2.CAP_PROP_FOCUS, 0)

    def get_img(self):
        self.camcap.read()
        return self.camcap.read()[1]


if __name__ == "__main__":
    ec = ExtCam()
    dnpa = ec.get_img()
    print(dnpa.shape)

    from huri.vision.yolov6.detect import detect
    from huri.core.common_import import fs
    from huri.components.utils.img_utils import letterbox, crop_center

    while True:
        im_o = ec.get_img()
        im_n, im_start_coord = crop_center(im_o, cropx=1920, cropy=1080)
        im0, detect_result = detect(im_n, weights="D:\chen\yolo_huri\\runs\\train\exp37\weights\\best.pt")
        for v in detect_result.values():
            _pos = v['pos']
            p1 = _pos[0] + im_start_coord
            p2 = _pos[1] + im_start_coord
            cv2.rectangle(im_o, p1.tolist(), p2.tolist(), color=(0, 255, 0), thickness=2)

        cv2.imshow("sdfds", im0)
        cv2.imshow("asd", im_o)
        cv2.waitKey(100)
    # while True:
    #     dnpa = ec.get_img()
    #     cv2.imshow("Depth", crop_center(dnpa, cropx=1920, cropy=1080))
    #     cv2.waitKey(100)
    # cv2.imwrite("D:\chen\phoxi_server_tst\calib_external_cam_custom\Data\\1.bmp", ec.get_img())

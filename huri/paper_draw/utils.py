from typing import List
from PIL import Image

import numpy as np
import cv2

from huri.core.common_import import cm, gm, rm, fs

root = fs.workdir.joinpath("paper_draw")


class Phoxi(object):
    def __init__(self, show_cone=True, real_pos = False):
        scanner_p1 = cm.CollisionModel(initor=str(root.joinpath("model/scanner_part_1.stl")))
        scanner_p2 = cm.CollisionModel(initor=str(root.joinpath("model/scanner_part_2.stl")))
        cone_cam = cm.CollisionModel(initor=str(root.joinpath("model/cone_cam.stl")))
        cone_proj = cm.CollisionModel(initor=str(root.joinpath("model/cone_projector.stl")))
        cone_scan_vol = cm.CollisionModel(initor=str(root.joinpath("model/cone_scan_volume.stl")))
        origin = gm.gen_sphere(radius=.0001)
        scanner_p1.objpdnp.reparentTo(origin.objpdnp)
        scanner_p2.objpdnp.reparentTo(origin.objpdnp)
        if show_cone:
            cone_cam.objpdnp.reparentTo(origin.objpdnp)
            cone_proj.objpdnp.reparentTo(origin.objpdnp)
            cone_scan_vol.objpdnp.reparentTo(origin.objpdnp)
        scanner_p1.set_rgba(np.array([14 / 255, 17 / 255, 17 / 255, 1]))
        scanner_p2.set_rgba(np.array([129 / 255, 136 / 255, 140 / 255, 1]))
        cone_cam.set_rgba(np.array([0, 0, 0, .1]))
        cone_proj.set_rgba(np.array([0, 0, 0, .1]))
        cone_scan_vol.set_rgba(np.array([31 / 255, 191 / 255, 31 / 255, .3]))
        self.base_mdl = cm.CollisionModel(origin.objpdnp)
        self.origin = np.array([-.17272, -.01741, -.0027])
        # setup position
        if real_pos:
            # fs.load_json()
            pass
        else:
            self.set_pos(np.array([0.31, 0, 1.04]))
            self.set_rpy(np.radians(90), 0, np.radians(90))

    def attach_to(self, *args, **kwargs):
        self.base_mdl.attach_to(*args, **kwargs)

    def set_pos(self, *args, **kwargs):
        self.base_mdl.set_pos(*args, **kwargs)

    def set_rpy(self, *args, **kwargs):
        self.base_mdl.set_rpy(*args, **kwargs)

    def set_homotmat(self, *args, **kwargs):
        self.base_mdl.set_homomat(*args, **kwargs)

    def get_origin(self):
        return rm.homomat_transform_points(self.base_mdl.get_homomat(), self.origin)


phoxi = Phoxi()
phoxi_nocone = Phoxi(show_cone=False)


def conv2eps(file_path):
    file_path = fs.Path(file_path).resolve()
    img_paths = []
    if file_path.is_dir():
        for f in file_path.glob("*"):
            if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_paths.append(f)
    elif file_path.is_file():
        if file_path.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_paths.append(file_path)
    for img_path in img_paths:
        img_ppath = img_path.parent
        img = Image.open(str(img_path))
        img.save(str(img_ppath.joinpath(f"{img_path.name.split('.')[0]}.eps")), lossless=True)


def remove_white_space(file_path):
    file_path = fs.Path(file_path).resolve()
    img_paths = []
    if file_path.is_dir():
        for f in file_path.glob("*"):
            if f.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                img_paths.append(f)
    elif file_path.is_file():
        if file_path.name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img_paths.append(file_path)
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        img_ppath = img_path.parent
        img_name = img_path.name.split(".")[0]

        # process
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        # get contours
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) < 1:
            return
        x = 999999999
        y = 999999999
        x2 = 0
        y2 = 0
        for cnt in cnts:
            ## (4) Crop and save it
            x_1, y_1, w, h = cv2.boundingRect(cnt)
            x_2 = x_1 + w
            y_2 = y_1 + h
            x = min(x, x_1)
            y = min(y, y_1)
            x2 = max(x2, x_2)
            y2 = max(y2, y_2)
        print(cv2.boundingRect(sorted(cnts, key=cv2.contourArea)[-1]))
        print(x, y, x2, y2)
        dst = img[y:y2, x:x2]

        cv2.imwrite(str(img_ppath.joinpath(f"{img_name}.jpg")), dst)

if __name__ == "__main__":
    remove_white_space(f"tase2022/collision_check_1.jpg")
    remove_white_space(f"tase2022/collision_check_2.jpg")
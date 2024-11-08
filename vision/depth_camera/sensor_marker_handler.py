from huri.vision.phoxi_capture import SensorMarkerHandler
import cv2.aruco as aruco
import numpy as np
import cv2
from huri.core.file_sys import workdir
import os
import vision.depth_camera.util_functions as dcuf
from shapely.geometry import Polygon, Point


def increase_brightness_constrast(img):
    alpha = 1  # Contrast control (1.0-3.0)
    beta = 10  # Brightness control (0-100)

    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted


clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def get_indices_inside_polygon(img, indices):
    image_height, image_width = img.shape[:2]
    # Define the polygon using the given indices
    polygon = Polygon(indices)

    # Calculate the bounding box of the polygon
    min_x, min_y = polygon.bounds[0], polygon.bounds[1]
    max_x, max_y = polygon.bounds[2], polygon.bounds[3]

    # Iterate through all pixels within the bounding box
    indices_inside_polygon = []
    for x in range(int(min_x), int(max_x) + 1):
        for y in range(int(min_y), int(max_y) + 1):
            point = Point(x, y)
            if polygon.contains(point):
                indices_inside_polygon.append((x, y))
    return np.array(indices_inside_polygon)


def marker_pos_in_hand2(pcd, img, corners, ids, aruco_target_id_list, downsampling_voxelsize=.001):
    """
    Deprecated: It has an error around 5 mm
    :param pcd:
    :param img:
    :param corners:
    :param ids:
    :param aruco_target_id_list:
    :param downsampling_voxelsize:
    :return:
    """
    h, w = img.shape[:2]
    bounds_id0 = corners[np.where(ids.squeeze() == aruco_target_id_list[0])[0][0]][0]
    bounds_id1 = corners[np.where(ids.squeeze() == aruco_target_id_list[1])[0][0]][0]

    extracted_indices = get_indices_inside_polygon(img, bounds_id0)
    extracted_points = pcd[extracted_indices[:, 1] * w + extracted_indices[:, 0], :]
    p = dcuf.remove_outlier(src_nparray=extracted_points,
                            downsampling_voxelsize=downsampling_voxelsize,
                            radius=downsampling_voxelsize * 1.5)[0]
    mean_point_0 = np.mean(p, axis=0)
    extracted_indices = get_indices_inside_polygon(img, bounds_id1)
    extracted_points = pcd[extracted_indices[:, 1] * w + extracted_indices[:, 0], :]
    p = dcuf.remove_outlier(src_nparray=extracted_points,
                            downsampling_voxelsize=downsampling_voxelsize,
                            radius=downsampling_voxelsize * 1.5)[0]
    mean_point_1 = np.mean(p, axis=0)
    return (mean_point_0 + mean_point_1) / 2

# def marker_pos_in_hand(pcd, img, corners, ids, aruco_target_id_list, downsampling_voxelsize=.001):
#     h, w = img.shape[:2]
#     bounds_id0 = corners[np.where(ids.squeeze() == aruco_target_id_list[0])[0][0]][0]
#     bounds_id1 = corners[np.where(ids.squeeze() == aruco_target_id_list[1])[0][0]][0]
#
#     center = (np.mean(bounds_id0, axis=0) + np.mean(bounds_id1, axis=0)).astype(np.int32)
#     pcd_center = pcd[center[1] * w + center[0]]
#     return pcd_center

def plot_x_on_image(image, x, y, size=5, color=(255, 0, 0)):
    """
    Plots an 'X' on the given image at location (x, y).

    Parameters:
    - image: 2D numpy array representing the image.
    - x: X-coordinate of the location to plot the 'X'.
    - y: Y-coordinate of the location to plot the 'X'.
    - size: The size of the 'X', default is 5.
    """
    # Calculate the indices for the two diagonals
    x_indices = np.arange(-size, size+1) + x
    y_indices_diag1 = np.arange(-size, size+1) + y
    y_indices_diag2 = -np.arange(-size, size+1) + y

    # Filter out indices that are outside the image bounds
    valid_indices = (x_indices >= 0) & (x_indices < image.shape[1]) & \
                    ((y_indices_diag1 >= 0) & (y_indices_diag1 < image.shape[0]) | \
                     (y_indices_diag2 >= 0) & (y_indices_diag2 < image.shape[0]))
    x_indices = x_indices[valid_indices]
    y_indices_diag1 = y_indices_diag1[valid_indices]
    y_indices_diag2 = y_indices_diag2[valid_indices]

    # Update the image pixels to draw the 'X'
    image[y_indices_diag1, x_indices] = color
    image[y_indices_diag2, x_indices] = color
    return image

class SensorMarkerHandler2(SensorMarkerHandler):

    def __init__(self, ip_adr="192.168.125.100:18300", aruco_target_id_list=(0, 1), toggle_cv_show=False, debug=False):
        super().__init__(ip_adr)
        self.aruco_parameters = aruco.DetectorParameters_create()
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        if os.environ.get('https_proxy'):
            del os.environ['https_proxy']
        if os.environ.get('http_proxy'):
            del os.environ['http_proxy']
        self.aruco_target_id_list = aruco_target_id_list
        self.toggle_cv_show = toggle_cv_show
        self.counter = 0
        self.debug = debug
        # self.img_dump_path = workdir / "data" / "calibration_img"
        # for jpg in self.img_dump_path.glob("*.jpg"):
        #     os.remove(jpg)

    def get_marker_center(self):
        if self.debug:
            return None
        self.sensor_client.triggerframe()
        texture_img = self.sensor_client.gettextureimg()
        enhanced_img = cv2.equalizeHist(texture_img)
        img = enhanced_img
        pcd = self.sensor_client.getpcd() * .001
        width = img.shape[1]
        # detect markers
        counter = 0
        while counter < 5:
            corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict,
                                                                  parameters=self.aruco_parameters)
            # print(ids)
            if ids is None or len(ids) < len(self.aruco_target_id_list):
                corners, ids, rejectedImgPoints = aruco.detectMarkers(texture_img, self.aruco_dict,
                                                                      parameters=self.aruco_parameters)
                img = texture_img
                if ids is None or len(ids) < len(self.aruco_target_id_list):
                    for i in range(self.counter + 1):
                        # img = increase_brightness_constrast(texture_img)
                        img = clahefilter.apply(texture_img)
                else:
                    break
            else:
                break
            counter += 1
        if ids is None or len(ids) < len(self.aruco_target_id_list):
            print("Failed")
        if len(corners) < len(self.aruco_target_id_list):
            return None
        if ids[0] not in self.aruco_target_id_list or ids[1] not in self.aruco_target_id_list:
            return None
        # center = np.mean(np.mean(corners, axis=0), axis=1)[0].astype(np.int32)
        # marker_pos_in_sensor = pcd[width * center[1] + center[0]]
        # return marker_pos_in_sensor
        # return marker_pos_in_hand2(pcd, img, corners, ids, self.aruco_target_id_list)

        bounds_id0 = corners[np.where(ids.squeeze() == self.aruco_target_id_list[0])[0][0]][0]
        bounds_id1 = corners[np.where(ids.squeeze() == self.aruco_target_id_list[1])[0][0]][0]

        center = (np.mean(bounds_id0, axis=0)/2 + np.mean(bounds_id1, axis=0)/2).astype(np.int32)
        pcd_center = pcd[center[1] * width + center[0]]


        if self.toggle_cv_show:
            img_enhanced = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # img_no_enhanced = cv2.cvtColor(texture_img, cv2.COLOR_GRAY2RGB)
            aruco.drawDetectedMarkers(img_enhanced, corners, ids, borderColor=[0, 0, 255])
            # aruco.drawDetectedMarkers(img_no_enhanced, corners, borderColor=[255, 255, 0])
            # cv2.imwrite(str(self.img_dump_path / f"img_enhanced_{self.counter}.jpg"), img_enhanced)
            img_enhanced = plot_x_on_image(img_enhanced, center[0], center[1])
            cv2.imshow('img', img_enhanced)
            cv2.waitKey(10)
            # cv2.imwrite(str(workdir/"data"/"calibration_img"/f"img_no_enhanced_{self.counter}.jpg"), img_no_enhanced)
            self.counter += 1
        return pcd_center
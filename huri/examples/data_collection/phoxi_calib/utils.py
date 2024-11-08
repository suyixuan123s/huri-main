"""
This function contains the utility functions to operate the calibration matrix generated by Phoxi Calibration Program
Author: Hao Chen
Date: 20220321
"""
from typing import Union, Tuple

import cv2

from huri.core.common_import import fs, np, rm
from huri.components.utils.img_utils import scale_img


def read_calibration_info(calib_path: Union[fs.Path, str]):
    """
    Load calibration information generated by Phoxi Calibration
    :param calib_path: path of the calibration information
    :return: intrinsic matrix , distortion coefficients, homogeneous matrix (from external camera coord to phoxi cam coord), camera resolution
    """
    calib_path = fs.Path(calib_path)
    with calib_path.open("r") as f:
        raw_calibration_info = f.read().split("\n")
    # first row: camera matrix
    # second row: Distortion Coefficients
    # third row: Rotation Vector
    # fourth row: Translation vector
    # fifth row: Camera Resolution
    row_cnt = 0
    cam_mat, dist_coef, rot_mat, tran_vec, cam_res = None, None, None, None, None
    for row in raw_calibration_info:
        if len(row) < 1:
            continue
        row_cnt += 1
        parsed_data = np.array([float(ele) for ele in row.rstrip().lstrip().split(" ")])
        if row_cnt == 1:
            cam_mat = parsed_data.reshape(3, 3)
        elif row_cnt == 2:
            dist_coef = parsed_data
        elif row_cnt == 3:
            rot_mat = parsed_data.reshape(3, 3)
        elif row_cnt == 4:
            tran_vec = parsed_data
        elif row_cnt == 5:
            cam_res = parsed_data.astype(int)
    if cam_mat is not None and dist_coef is not None and rot_mat is not None and tran_vec is not None and cam_res is not None:
        print("Calibration information loaded successfully")
        homo_mat = rm.homomat_from_posrot(tran_vec, rot_mat)
        return cam_mat, dist_coef, homo_mat, cam_res
    else:
        raise Exception("Calibration information is not correctly loaded")


# TODO A common function for project points of pcd for `project_pcd2extcamimg` and `project_region_extcamimg2pcd`
def project_pcd2extcamimg(pcd_raw: np.ndarray,
                          phoxi2extcam_homo_mat: np.ndarray,
                          cam_mat: np.ndarray,
                          extcam_res: np.ndarray,
                          dist_coef: np.ndarray, scale: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project the 3D point cloud data to 2D image captured external camera
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    :param pcd_raw: raw point cloud data captured by phoxi
    :param phoxi2extcam_homo_mat: 4x4 homogenous transform from phoxi coordinate to external camera coordinate
    :param cam_mat: 3x3 intrinsic matrix transform from 3D to 2D
    :param extcam_res: resolution of external camera
    :param dist_coef: distortion coefficient
    :param scale: scale factor (the unit of raw pcd is in mm, the unit of wrs is in meter)
    :return: 2D coordinate of point cloud in image, feasible point cloud indices ( can be projected into image)
    """

    proj_coord = \
        cv2.projectPoints(pcd_raw * scale, rvec=cv2.Rodrigues(phoxi2extcam_homo_mat[:3, :3])[0],
                          tvec=phoxi2extcam_homo_mat[:3, 3],
                          cameraMatrix=cam_mat, distCoeffs=dist_coef)[0]
    proj_coord = np.round(proj_coord).astype(int).reshape(-1, 2)
    valid_proj_coord_index = np.where((proj_coord[:, 0] >= 0) & (proj_coord[:, 0] <= extcam_res[0])
                                      & (proj_coord[:, 1] >= 0) & (proj_coord[:, 1] <= extcam_res[1]))[0]
    return proj_coord[valid_proj_coord_index], valid_proj_coord_index


def project_region_extcamimg2pcd(pcd_raw: np.ndarray,
                                 image_region: np.ndarray,
                                 phoxi2extcam_homo_mat: np.ndarray,
                                 cam_mat: np.ndarray,
                                 dist_coef: np.ndarray, scale: int = 1000) -> np.ndarray:
    """
    Project the region in 2D image captured external camera to 3D point cloud data
    https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    :param pcd_raw: raw point cloud data captured by phoxi
    :param image_region: [x_left, x_right, y_top, y_bottom] indicates the region in image
    :param phoxi2extcam_homo_mat: 4x4 homogenous transform from phoxi coordinate to external camera coordinate
    :param cam_mat: 3x3 intrinsic matrix transform from 3D to 2D
    :param extcam_res: resolution of external camera
    :param dist_coef: distortion coefficient
    :param scale: scale factor (the unit of raw pcd is in mm, the unit of wrs is in meter)
    :return: point cloud indices
    """

    proj_coord = cv2.projectPoints(pcd_raw * scale, rvec=cv2.Rodrigues(phoxi2extcam_homo_mat[:3, :3])[0],
                                   tvec=phoxi2extcam_homo_mat[:3, 3],
                                   cameraMatrix=cam_mat, distCoeffs=dist_coef)[0]
    proj_coord = np.round(proj_coord).astype(int).reshape(-1, 2)
    return np.where((proj_coord[:, 0] >= image_region[0]) & (proj_coord[:, 0] <= image_region[1])
                    & (proj_coord[:, 1] >= image_region[2]) & (proj_coord[:, 1] <= image_region[3]))[0]


if __name__ == "__main__":
    cam_mat, dist_coef, homo_mat, cam_res = read_calibration_info(
        "D:\chen\phoxi_server_tst\calib_external_cam_custom\calibration.txt")

    pho2extcam_homo_mat = np.linalg.inv(homo_mat)

    from huri.vision.phoxi_capture import vision_read_data

    #
    # sensor_streamer = SensorMarkerHandler(HuriConstants.IP_ADR_DEPTH_SENSOR_DEBUG)
    # SAVE_PATH = fs.workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
    # pcd, img, depth_img, rgb_texture, extcam_img = vision_pipeline(sensor_streamer, dump_path=SAVE_PATH)
    pcd, img, depth_img, rgb_texture, extcam_img = vision_read_data(
        filepath="D:\chen\huri_shared\huri\data\\vision_exp\\20220322-173059.pkl")
    CALIB_MAT_PATH = fs.workdir / "data/calibration/qaqqq.json"
    affine_mat = np.asarray(fs.load_json(CALIB_MAT_PATH)["affine_mat"])
    # pcd = rm.homomat_transform_points(affine_mat, pcd)
    # pcd = pcd[(pcd[:, 2] < .065) & (pcd[:, 2] > .03) & (pcd[:, 0] > .2)]
    from huri.core.common_import import fs, np, rm, wd, gm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    # gm.gen_pointcloud(pcd).attach_to(base)

    # pcd = rm.homomat_transform_points(np.linalg.inv(affine_mat), pcd)
    # proj_points = \
    #     cv2.projectPoints(pcd * 1000, rvec=cv2.Rodrigues(pho2extcam_homo_mat[:3, :3])[0],
    #                       tvec=pho2extcam_homo_mat[:3, 3],
    #                       cameraMatrix=cam_mat, distCoeffs=dist_coef)[0]
    # proj_points = np.round(proj_points).astype(int).reshape(-1, 2)
    # feasible_points = proj_points[
    #     (proj_points[:, 0] >= 0) & (proj_points[:, 0] <= cam_res[1]) & (proj_points[:, 1] >= 0) & (
    #             proj_points[:, 1] <= cam_res[0])]

    # feasible_points, feasible_points_indx = project_pcd2extcamimg(pcd,
    #                                                               pho2extcam_homo_mat,
    #                                                               cam_mat,
    #                                                               extcam_res=cam_res,
    #                                                               dist_coef=dist_coef, )
    #
    # lefttop = feasible_points[:, 0].min(), feasible_points[:, 1].min()
    # rightbottom = feasible_points[:, 0].max(), feasible_points[:, 1].max()
    # extcam_img_r = cv2.rectangle(extcam_img, lefttop, rightbottom, (0, 0, 255), 2)
    extcam_img_r = cv2.rectangle(extcam_img, (1216, 948), (1277, 994), (0, 0, 255), 2)


    # print(lefttop, rightbottom)

    def click_event(event, x, y, flags, params):

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(extcam_img_r, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', extcam_img_r)

        # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = extcam_img_r[y, x, 0]
            g = extcam_img_r[y, x, 1]
            r = extcam_img_r[y, x, 2]
            cv2.putText(extcam_img_r, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x, y), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image', extcam_img_r)


    cv2.imshow("image", scale_img(extcam_img_r, .7))
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)

    index = project_region_extcamimg2pcd(pcd,
                                         image_region=np.array([1216, 1277, 948, 994]),
                                         phoxi2extcam_homo_mat=pho2extcam_homo_mat,
                                         cam_mat=cam_mat,
                                         dist_coef=dist_coef, )
    gm.gen_pointcloud(pcd).attach_to(base)
    gm.gen_pointcloud(pcd[index], rgbas=[[1, 0, 0, 1]]).attach_to(base)
    # feasbile range in image
    # proj_points[(proj_points[:,]>)]
    base.run()
    txt_height, txt_width = sensor_streamer.texture_size
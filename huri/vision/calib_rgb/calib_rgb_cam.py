import cv2
import cv2.aruco as aruco
import yaml
import glob


def calib_charucoboard(nrow,
                       ncolumn,
                       aruco_markerdict=aruco.DICT_6X6_250,
                       square_markersize=25,
                       imgs_path='./',
                       img_format='png',
                       save_name='mycam_charuco_data.yaml'):
    """
    :param nrow:
    :param ncolumn:
    :param marker_dict:
    :param imgs_path:
    :param save_name:
    :return:
    author: weiwei
    date: 20190420
    """
    # read images and detect cornders
    aruco_dict = aruco.Dictionary_get(aruco_markerdict)
    allCorners = []
    allIds = []
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 27, 0.0001)
    board = aruco.CharucoBoard_create(ncolumn, nrow, square_markersize, .57 * square_markersize, aruco_dict)
    print(imgs_path)
    images = glob.glob(f"{imgs_path}\\" '*.' + img_format)
    candfiles = []
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner, winSize=(2, 2), zeroZone=(-1, -1), criteria=criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            # require len(res2[1]) > nrow*ncolumn/2 at least half of the corners are detected
            if res2[1] is not None and res2[2] is not None and len(res2[1]) > (nrow - 1) * (ncolumn - 1) / 2:
                allCorners.append(res2[1])
                allIds.append(res2[2])
                candfiles.append((fname.split("/")[-1]).split("\\")[-1])
            imaxis = aruco.drawDetectedMarkers(img, corners, ids)
            imaxis = aruco.drawDetectedCornersCharuco(imaxis, res2[1], res2[2], (255, 255, 0))
            cv2.imshow('img', imaxis)
            cv2.waitKey(100)
    # The calibratedCameraCharucoExtended function additionally estimate calibration errors
    # Thus, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors are returned
    # We dont use them here though
    # see https://docs.opencv.org/3.4.6/d9/d6a/group__aruco.html for details
    (ret, mtx, dist, rvecs, tvecs,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors) = \
        cv2.aruco.calibrateCameraCharucoExtended(charucoCorners=allCorners, charucoIds=allIds, board=board,
                                                 imageSize=gray.shape, cameraMatrix=None, distCoeffs=None,
                                                 flags=cv2.CALIB_RATIONAL_MODEL,
                                                 criteria=(
                                                     cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    print(ret, mtx, dist, rvecs, tvecs, candfiles)
    if ret:
        with open(save_name, "w") as f:
            yaml.dump([mtx, dist, rvecs, tvecs, candfiles], f)

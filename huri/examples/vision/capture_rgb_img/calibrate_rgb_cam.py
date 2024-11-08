"""
This is an example to calibrate the RGB camera. In this example, charucoboard is used for camera calibration
1. Find the intrinsic and extrinsic properties of a camera
2. Undistort images
"""

if __name__ == "__main__":
    import drivers.rpc.extcam.cameras as camera

    from huri.core.common_import import fs
    from huri.core.print_tool import print_with_border
    from huri.vision.calib_rgb.calib_rgb_cam import calib_charucoboard
    from huri.examples.vision.capture_rgb_img.capture_rgb_img import capture_images

    image_save_path = fs.Path(__file__).parent / "data"
    calibration_save_path = fs.Path(__file__).parent
    marker_size = 40
    marker_pattern = (7, 5)  # a 7 by 5 charucoboard

    print_with_border("Start calibration")
    # capture images, press `Esc` to stop capture
    capture_images(image_save_path=image_save_path,
                   rgbcam_streamer=camera.ExtCam())

    # calibrate the camera according to capture images
    calib_charucoboard(nrow=marker_pattern[0], ncolumn=marker_pattern[1],
                       square_markersize=marker_size,
                       imgs_path=str(image_save_path),
                       save_name=str(calibration_save_path / 'rgbcam_calib.yaml'), img_format="jpg")

    print_with_border("Finished")

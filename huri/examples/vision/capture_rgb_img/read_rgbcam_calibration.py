if __name__ == '__main__':
    from huri.core.common_import import fs
    import yaml
    import numpy as np

    calibration_path = fs.Path(__file__).parent / "rgbcam_calib.yaml"
    mtx, dist, rvecs, tvecs, candfiles = yaml.unsafe_load(calibration_path.open())
    print(f"focal length f_x {mtx[0, 0]}, f_y {mtx[1, 1]}")
    print(f"optical center is [{mtx[0, 2]},{mtx[1, 2]}] ")


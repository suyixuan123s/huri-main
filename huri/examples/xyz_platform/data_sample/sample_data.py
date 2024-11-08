"""
This is an example to sample the data using xyz platform
"""
from robot_con.xyz_platform.xyz_platform import XYZPlatformController
from robot_sim.xyz_platform.xyz_platform import XYZPlatform


def sample_data(xyz_sim: XYZPlatform,
                xyz_con: XYZPlatformController,
                num_sample=10,
                call_back_func=lambda: None,
                region=None):
    for i in range(num_sample):
        print(f"This is the {i + 1} sample")
        rand_conf = xyz_sim.rand_conf("all", region=region)
        xyz_con.set_pos("all", pos=rand_conf)
        call_back_func()
        print(f"The x platform pos is {rand_conf[0]}\n"
              f"The y platform pos is {rand_conf[1]}\n"
              f"The z platform ang is {np.rad2deg(rand_conf[2])}")
        print("-" * 20)


if __name__ == "__main__":
    from huri.core.common_import import *
    from huri.vision.phoxi_capture import SensorMarkerHandler, vision_pipeline
    from time import strftime

    DEBUG = False
    if DEBUG:
        cap_img_func = lambda: print("Capture a image")
    else:
        sensor_handle = SensorMarkerHandler()
        # create a folder to save img
        file_dir = fs.workdir / "data" / "vision_exp" / f"8_8_8_10"
        if not file_dir.exists():
            file_dir.mkdir()
        else:
            raise Exception("Folder exists!")
        cap_img_func = lambda: vision_pipeline(sensor_handle,
                                               file_dir / f"exp_{strftime('%Y%m%d-%H%M%S')}.pkl")

    # generate the xyz platform in simulation
    xyz_platform = XYZPlatform()

    # generate the xyz platform controller
    # xyz_con = XYZPlatformController(debug=DEBUG)
    xyz_con = XYZPlatformController(debug=DEBUG)
    # calibrate the xyz platform
    xyz_con.calibrate()

    # sample data
    sample_data(xyz_sim=xyz_platform, xyz_con=xyz_con, num_sample=20, call_back_func=cap_img_func,
                region=np.array([[.05, .27],
                                 [.15, .45],
                                 [0, np.radians(350)], ]))

    xyz_con.set_pos("all", pos=[0, 0, 0])

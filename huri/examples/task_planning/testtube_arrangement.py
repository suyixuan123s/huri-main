import numpy as np

from huri.core.file_sys import workdir
from huri.components.pipeline.data_pipeline import RenderController
from huri.vision.phoxi_capture import vision_pipeline, SensorMarkerHandler
from huri.core.common_import import *
import huri.components.utils.plot_projection as pp
from huri.components.vision.tube_detector import TestTubeDetector
import matplotlib
from time import strftime

matplotlib.use('TkAgg')

IP_ADR = "192.168.125.100:18300"
SAVE_PATH = workdir / "data" / "vision_exp" / f"{strftime('%Y%m%d-%H%M%S')}.pkl"  # None: Do not save the data captured by phoxi
AFFINE_MAT_PATH = workdir / "data/calibration/qaqqq.json"
DEBUG = True


def test():
    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    std_out = RenderController(base.tkRoot, base)
    # 2D canvas to show the projection of point clouds
    canvas = pp.Plot(x_size=500)
    # Get Data From Camera
    if DEBUG:
        filename = workdir / "data" / "vision_exp" / "20211226-163347.pkl"
        pcd, img = fs.load_pickle(filename)
    else:
        pcd, img = vision_pipeline(streamer=SensorMarkerHandler(IP_ADR),
                                   dump_path=SAVE_PATH)
    # Init detector
    detector = TestTubeDetector(affine_mat_path=AFFINE_MAT_PATH)
    # Detect
    detected_results, rack_light_thinner, rack_tf = detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                           std_out=std_out, canvas=canvas,
                                                                           toggle_yolo=False)

    # init the simulation robot
    yumi_robot = ym.Yumi(enable_cc=True)
    yumi_robot.gen_meshmodel().attach_to(base)

    # Show results
    # canvas.show()
    print(repr(rack_light_thinner.rack_status))
    # task planning starts
    from huri.components.task_planning.tube_puzzle_learning_solver import DQNSolver
    from huri.learning.env.arrangement_planning_rack.env import RackStatePlot
    goal_pattern = np.array([[1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
                             [1, 1, 1, 2, 2, 2, 2, 3, 3, 3]])
    solver = DQNSolver()

    path = solver.solve(current_state=rack_light_thinner.rack_status,
                        goal_pattern=goal_pattern,
                        toggle_result=True)
    print(path)
    base.run()


if __name__ == "__main__":
    test()

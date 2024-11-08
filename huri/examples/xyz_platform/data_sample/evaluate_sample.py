from huri.core.common_import import *
from huri.components.vision.tube_detector import TestTubeDetector, NoRackException
from numpy import array

# setup the file sampled
file_dir = fs.workdir / "data" / "vision_exp" / "exp_20211222193921"
sampled_file_paths = list(file_dir.glob("**/*"))

# initialize the tube detector
CALIB_MAT_PATH = fs.workdir / "data" / "calibration" / "qaqqq.json"
tube_detector = TestTubeDetector(CALIB_MAT_PATH)

# Test 1: exp_20211222162442
# baseline_reco = array([[1, 0, 0, 0, 3, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 2, 2, 2, 2, 1, 0],
#                        [3, 3, 3, 0, 2, 2, 0, 0, 0, 2],
#                        [3, 0, 0, 0, 0, 0, 0, 0, 2, 0],
#                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
# Test 2: exp_20211222185457
baseline_reco = array([[1, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 2, 0, 1, 0],
                       [3, 3, 0, 0, 2, 0, 2, 3, 0, 3],
                       [0, 0, 0, 0, 2, 0, 0, 0, 2, 0],
                       [0, 0, 0, 1, 0, 0, 2, 0, 0, 0]])
# Test 3: exp_20211222191529
baseline_reco = array([[1, 0, 2, 2, 0, 3, 1, 0, 1, 0],
                       [0, 2, 0, 0, 1, 0, 0, 2, 0, 0],
                       [2, 0, 3, 2, 3, 1, 3, 0, 2, 2],
                       [0, 1, 0, 0, 2, 2, 1, 0, 0, 3],
                       [0, 0, 3, 0, 0, 3, 0, 3, 0, 0]])
# Test 4: exp_20211222192739
baseline_reco = array([[2, 0, 2, 0, 0, 0, 2, 0, 3, 0],
                       [0, 3, 3, 1, 3, 2, 0, 0, 0, 2],
                       [3, 2, 0, 2, 1, 0, 2, 1, 0, 1],
                       [2, 0, 0, 0, 0, 1, 0, 0, 2, 0],
                       [0, 0, 3, 0, 0, 1, 3, 3, 0, 1]])
# Test 5: exp_20211222193921
baseline_reco = array([[1, 0, 0, 3, 3, 0, 1, 2, 0, 3],
                       [0, 2, 3, 0, 3, 1, 0, 0, 0, 2],
                       [1, 0, 2, 0, 0, 1, 2, 0, 2, 3],
                       [1, 3, 0, 2, 2, 0, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0, 3, 2, 2, 0, 0]])

baseline_reco_flip = np.flip(baseline_reco)
num_test_tubes = len(np.where(baseline_reco > 0)[0])

for idx, file_path in enumerate(sampled_file_paths):
    print(f"The {idx + 1} sample")
    pcd, img = fs.load_pickle(file_path)
    try:
        detected_results, rack, _, = tube_detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0], toggle_yolo=True)
        is_correct = np.array_equal(baseline_reco, rack.rack_status) or np.array_equal(baseline_reco_flip,
                                                                                       rack.rack_status)
        number_of_success = max(
            len(np.where(((baseline_reco == rack.rack_status) & ~(baseline_reco == 0)) > 0)[0]),
            len(np.where(((baseline_reco_flip == rack.rack_status) & ~(baseline_reco_flip == 0)) > 0)[0])
        )
    except NoRackException:
        print("Rack Cannot be recognized")
        is_correct = False
        number_of_success = 0
    except Exception as e:
        raise Exception(e)
    print(f"is reco correct?", f"{is_correct} ")
    print(f"Detected successful tube : {number_of_success}/{num_test_tubes}")
    print(f"-------------------------")

from huri.core.common_import import *
from huri.components.vision.tube_detector import TestTubeDetector, NoRackException
from numpy import array

# setup the file sampled
file_dir = fs.workdir / "data" / "vision_exp" / "4_4_4_1"
sampled_file_paths = list(file_dir.glob("**/*"))

# initialize the tube detector
CALIB_MAT_PATH = fs.workdir / "data" / "calibration" / "qaqqq.json"
tube_detector = TestTubeDetector(CALIB_MAT_PATH)

baseline_reco = array([[1, 0, 0, 0, 3, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 2, 2, 2, 2, 1, 0],
                       [3, 3, 3, 0, 2, 2, 0, 0, 0, 2],
                       [3, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])


def eval_sample(baseline_reco, sampled_file_paths):
    is_rack_reco = []
    num_tube_reco = []
    num_tube_correct_reco = []
    baseline_reco_flip = np.flip(baseline_reco)
    num_test_tubes = len(np.where(baseline_reco > 0)[0])

    result_dict = {
        "is_rack_reco": is_rack_reco,
        "num_tube_reco": num_tube_reco,
        "num_tube_correct_reco": num_tube_correct_reco,
        "num_tubes_baseline": num_test_tubes
    }

    for idx, file_path in enumerate(sampled_file_paths):
        print(f"The {idx + 1} sample")
        pcd, img = fs.load_pickle(file_path)
        try:
            detected_results, rack, _, = tube_detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                     toggle_yolo=False)
            is_correct = np.array_equal(baseline_reco, rack.rack_status) or np.array_equal(baseline_reco_flip,
                                                                                           rack.rack_status)
            number_of_success = max(
                len(np.where(((baseline_reco == rack.rack_status) & ~(baseline_reco == 0)) > 0)[0]),
                len(np.where(((baseline_reco_flip == rack.rack_status) & ~(baseline_reco_flip == 0)) > 0)[0])
            )
            is_rack_reco.append(True)
            num_tube_reco.append(len(np.where(rack.rack_status > 0)[0]))
            num_tube_correct_reco.append(number_of_success)
        except NoRackException:
            print("Rack Cannot be recognized")
            is_correct = False
            number_of_success = 0
            is_rack_reco.append(False)
            num_tube_reco.append(0)
            num_tube_correct_reco.append(0)

        except Exception as e:
            raise Exception(e)
        print(f"is reco correct?", f"{is_correct} ")
        print(f"Detected successful tube : {number_of_success}/{num_test_tubes}")
        print(f"-------------------------")
    return result_dict


r = eval_sample(baseline_reco=baseline_reco, sampled_file_paths=sampled_file_paths)
print(r)
with open("result", "w") as f:
    f.writelines('is_rack_reco\n')
    [f.writelines(f'{i}\n') for i in r['is_rack_reco']]
    f.writelines('num_tube_correct_reco\n')
    [f.writelines(f'{i}\n') for i in r['num_tube_correct_reco']]
    f.writelines('num_tube_reco\n')
    [f.writelines(f'{i}\n') for i in r['num_tube_reco']]

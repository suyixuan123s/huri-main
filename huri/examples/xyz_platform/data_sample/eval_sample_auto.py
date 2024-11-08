from huri.core.common_import import *
from huri.components.vision.tube_detector import TestTubeDetector, NoRackException
from numpy import array
from huri.definitions.tube_def import TubeType

TubeType.TUBE_TYPE_1 = 1
TubeType.TUBE_TYPE_2 = 2


def eval_sample(baseline_reco, sampled_file_paths, group_info):
    print(f"Group Id = {group_info}")
    is_rack_reco = []
    num_tube_reco = []
    num_tube_correct_reco = []
    baseline_reco_flip = np.flip(baseline_reco)
    num_test_tubes = len(np.where(baseline_reco > 0)[0])

    baseline_reco_class_1_id = np.where(baseline_reco == 1)
    baseline_reco_class_2_id = np.where(baseline_reco == 2)
    baseline_reco_class_3_id = np.where(baseline_reco == 3)

    baseline_reco_flip_class_1_id = np.where(baseline_reco_flip == 1)
    baseline_reco_flip_class_2_id = np.where(baseline_reco_flip == 2)
    baseline_reco_flip_class_3_id = np.where(baseline_reco_flip == 3)

    num_class_1 = len(baseline_reco_flip_class_1_id[0])
    num_class_2 = len(baseline_reco_flip_class_2_id[0])
    num_class_3 = len(baseline_reco_flip_class_3_id[0])

    total_num_class_1 = 0
    total_num_class_2 = 0
    total_num_class_3 = 0

    currect_num_class_1 = 0
    currect_num_class_2 = 0
    currect_num_class_3 = 0

    wrong_num_class_1 = 0
    wrong_num_class_2 = 0
    wrong_num_class_3 = 0

    total_rack_num = 0
    correct_rack_num = 0

    for idx, file_path in enumerate(sampled_file_paths):
        print(f"The {idx + 1} sample")
        pcd, img = fs.load_pickle(file_path)
        total_rack_num += 1
        try:
            detected_results, rack, _, = tube_detector.analyze_scene(pcd=pcd, texture_img=img[:, :, 0],
                                                                     toggle_yolo=True, toggle_info=False)
            rack_state = rack.rack_status
            reference_mat = baseline_reco
            class_1_id = baseline_reco_class_1_id
            class_2_id = baseline_reco_class_2_id
            class_3_id = baseline_reco_class_3_id
            if len(np.where(((baseline_reco_flip == rack.rack_status) & ~(baseline_reco_flip == 0)) > 0)[0]) > \
                    len(np.where(((baseline_reco == rack.rack_status) & ~(baseline_reco == 0)) > 0)[0]):
                reference_mat = baseline_reco_flip
                class_1_id = baseline_reco_flip_class_1_id
                class_2_id = baseline_reco_flip_class_2_id
                class_3_id = baseline_reco_flip_class_3_id

            total_num_class_1 += num_class_1
            total_num_class_2 += num_class_2
            total_num_class_3 += num_class_3

            # update
            currect_num_class_1 += len(np.where(rack_state[class_1_id] == 1)[0])
            currect_num_class_2 += len(np.where(rack_state[class_2_id] == 2)[0])
            currect_num_class_3 += len(np.where(rack_state[class_3_id] == 3)[0])
            wrong_num_class_1 += len(np.where((rack_state[class_1_id] > 0) & (rack_state[class_1_id] != 1))[0])
            wrong_num_class_2 += len(np.where((rack_state[class_2_id] > 0) & (rack_state[class_2_id] != 2))[0])
            wrong_num_class_3 += len(np.where((rack_state[class_3_id] > 0) & (rack_state[class_3_id] != 3))[0])
            correct_rack_num += 1
        except NoRackException:
            print("Rack Cannot be recognized")

        except Exception as e:
            raise Exception(e)
    print("FINAL RESULT is !")
    print(f"Class 1 correct {currect_num_class_1}/{total_num_class_1}, wrong {wrong_num_class_1}",
          f"Class 2 correct {currect_num_class_2}/{total_num_class_2}, wrong {wrong_num_class_2}",
          f"Class 3 correct {currect_num_class_3}/{total_num_class_3}, wrong {wrong_num_class_3}",
          f"Rack correct {correct_rack_num}/{total_rack_num}")
    return (total_num_class_1,
            total_num_class_2,
            total_num_class_3, currect_num_class_1,
            currect_num_class_2,
            currect_num_class_3,
            correct_rack_num,
            total_rack_num)


# setup the file sampled
file_dir_list = [
    fs.workdir / "data" / "vision_exp" / "4_4_4_2",
    fs.workdir / "data" / "vision_exp" / "4_4_4_3",
    fs.workdir / "data" / "vision_exp" / "4_4_4_4",
    fs.workdir / "data" / "vision_exp" / "4_4_4_5",
    fs.workdir / "data" / "vision_exp" / "4_4_4_6",
    fs.workdir / "data" / "vision_exp" / "4_4_4_7",
    fs.workdir / "data" / "vision_exp" / "4_4_4_8",
    fs.workdir / "data" / "vision_exp" / "4_4_4_9",
    fs.workdir / "data" / "vision_exp" / "4_4_4_10",
    fs.workdir / "data" / "vision_exp" / "6_6_6_1",
    fs.workdir / "data" / "vision_exp" / "6_6_6_2",
    fs.workdir / "data" / "vision_exp" / "6_6_6_3",
    fs.workdir / "data" / "vision_exp" / "6_6_6_4",
    fs.workdir / "data" / "vision_exp" / "6_6_6_5",
    fs.workdir / "data" / "vision_exp" / "6_6_6_6",
    fs.workdir / "data" / "vision_exp" / "6_6_6_7",
    fs.workdir / "data" / "vision_exp" / "6_6_6_8",
    fs.workdir / "data" / "vision_exp" / "6_6_6_9",
    fs.workdir / "data" / "vision_exp" / "6_6_6_10",
    fs.workdir / "data" / "vision_exp" / "8_8_8_1",
    fs.workdir / "data" / "vision_exp" / "8_8_8_2",
    fs.workdir / "data" / "vision_exp" / "8_8_8_3",
    fs.workdir / "data" / "vision_exp" / "8_8_8_4",
    fs.workdir / "data" / "vision_exp" / "8_8_8_5",
    fs.workdir / "data" / "vision_exp" / "8_8_8_6",
    fs.workdir / "data" / "vision_exp" / "8_8_8_7",
    fs.workdir / "data" / "vision_exp" / "8_8_8_8",
    fs.workdir / "data" / "vision_exp" / "8_8_8_9",
    fs.workdir / "data" / "vision_exp" / "8_8_8_10", ]

reference_list = [
    # 4_4_4_2
    array([[0, 3, 0, 0, 0, 0, 3, 0, 1, 0],
           [0, 0, 2, 0, 0, 2, 0, 0, 0, 0],
           [0, 0, 0, 2, 0, 3, 0, 1, 0, 0],
           [0, 3, 0, 2, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    # 4_4_4_3
    array([[0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
           [0, 0, 0, 3, 0, 2, 0, 0, 0, 1],
           [0, 0, 0, 2, 1, 0, 0, 1, 0, 0],
           [2, 0, 0, 0, 0, 0, 3, 0, 1, 0],
           [0, 0, 3, 0, 0, 3, 0, 0, 0, 0]]),
    # 4_4_4_4
    array([[0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 2, 1, 1, 0, 0],
           [0, 0, 2, 0, 1, 0, 0, 1, 0, 0],
           [0, 0, 0, 3, 3, 0, 3, 0, 2, 0],
           [0, 0, 0, 0, 0, 3, 0, 0, 0, 0]]),
    # 4_4_4_5
    array([[0, 2, 0, 0, 0, 1, 3, 0, 1, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
           [2, 0, 0, 2, 0, 1, 0, 0, 3, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 3, 0],
           [0, 0, 1, 0, 0, 0, 2, 0, 0, 0]]),
    # 4_4_4_6
    array([[0, 0, 3, 2, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 2, 2, 0, 2, 0],
           [0, 1, 3, 0, 0, 0, 1, 0, 0, 0],
           [0, 1, 0, 3, 0, 0, 3, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
    # 4_4_4_7
    array([[0, 3, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 3, 0, 0, 0, 0, 0, 0],
           [0, 0, 2, 1, 3, 1, 0, 0, 0, 1],
           [2, 3, 0, 0, 2, 0, 0, 0, 0, 0],
           [2, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    # 4_4_4_8
    array([[0, 0, 0, 0, 0, 2, 0, 3, 0, 0],
           [3, 0, 2, 0, 0, 2, 0, 0, 0, 2],
           [0, 0, 3, 0, 1, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
    # 4_4_4_9
    array([[0, 2, 0, 3, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 3, 0, 0, 1, 0, 3],
           [0, 0, 0, 0, 0, 1, 0, 2, 0, 0],
           [2, 0, 2, 0, 0, 3, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]),
    # 4_4_4_10
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [3, 2, 0, 0, 3, 0, 3, 0, 0, 2],
           [0, 0, 0, 0, 1, 0, 0, 1, 3, 0],
           [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0, 2]]),
    ####################################################################
    # 6_6_6_1
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 0, 3, 0, 3, 3, 0, 0],
           [0, 0, 1, 2, 0, 2, 0, 0, 0, 1],
           [1, 0, 0, 3, 3, 0, 2, 2, 2, 0],
           [0, 0, 2, 0, 0, 3, 0, 0, 0, 1]]),
    # 6_6_6_2
    array([[0, 0, 2, 2, 3, 0, 0, 0, 2, 0],
           [0, 1, 0, 1, 0, 2, 1, 1, 0, 2],
           [0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
           [3, 0, 0, 3, 0, 3, 3, 0, 0, 0],
           [3, 0, 0, 0, 0, 0, 0, 0, 1, 0]]),
    # 6_6_6_3
    array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 3],
           [0, 1, 0, 0, 0, 2, 3, 3, 1, 1],
           [0, 0, 2, 0, 2, 0, 0, 0, 0, 0],
           [3, 1, 0, 2, 0, 0, 0, 2, 0, 0],
           [0, 0, 3, 0, 0, 3, 2, 0, 1, 0]]),
    # 6_6_6_4
    array([[3, 1, 0, 2, 3, 0, 0, 2, 0, 3],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [1, 3, 1, 2, 0, 0, 3, 0, 0, 1],
           [0, 3, 0, 0, 0, 0, 0, 0, 1, 0],
           [2, 0, 0, 2, 0, 2, 0, 0, 0, 0]]),
    # 6_6_6_5
    array([[0, 3, 0, 0, 3, 0, 0, 2, 1, 0],
           [1, 0, 1, 1, 0, 2, 1, 0, 0, 0],
           [0, 0, 1, 0, 0, 2, 3, 3, 0, 0],
           [0, 0, 2, 3, 3, 0, 0, 0, 0, 0],
           [0, 0, 0, 2, 0, 2, 0, 0, 0, 0]]),
    # 6_6_6_6
    array([[0, 1, 1, 1, 3, 1, 0, 0, 2, 0],
           [2, 0, 0, 0, 0, 0, 0, 0, 0, 2],
           [3, 0, 0, 0, 0, 3, 0, 0, 0, 2],
           [1, 0, 0, 2, 3, 0, 1, 0, 3, 0],
           [0, 0, 0, 3, 0, 2, 0, 0, 0, 0]]),
    # 6_6_6_7
    array([[0, 1, 0, 0, 3, 1, 0, 0, 0, 0],
           [2, 1, 2, 1, 0, 0, 2, 0, 0, 0],
           [0, 2, 1, 0, 3, 0, 0, 0, 0, 0],
           [0, 3, 0, 2, 3, 0, 1, 0, 0, 0],
           [3, 0, 0, 3, 0, 2, 0, 0, 0, 0]]),
    # 6_6_6_8
    array([[2, 0, 2, 3, 2, 0, 3, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 3, 0],
           [0, 0, 3, 2, 3, 0, 3, 0, 1, 1],
           [0, 0, 0, 0, 2, 0, 0, 1, 0, 2],
           [0, 0, 0, 1, 0, 1, 0, 0, 0, 0]]),
    # 6_6_6_9
    array([[2, 0, 0, 3, 0, 0, 0, 0, 2, 0],
           [0, 0, 1, 0, 0, 0, 1, 3, 0, 0],
           [2, 1, 1, 0, 0, 0, 1, 0, 2, 0],
           [0, 2, 3, 0, 0, 3, 1, 2, 3, 0],
           [0, 0, 0, 0, 3, 0, 0, 0, 0, 0]]),
    # 6_6_6_10
    array([[0, 0, 0, 3, 2, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 2, 2, 3, 0, 0],
           [0, 0, 0, 2, 3, 3, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 2, 1, 0, 0, 0],
           [0, 0, 0, 1, 3, 0, 3, 2, 0, 0]]),
    ####################################################################
    # 8_8_8_1
    array([[0, 3, 2, 0, 1, 0, 0, 3, 2, 0],
           [3, 2, 0, 1, 0, 3, 3, 0, 0, 0],
           [3, 0, 0, 1, 2, 0, 1, 0, 0, 2],
           [0, 0, 2, 0, 3, 2, 1, 0, 0, 1],
           [1, 3, 0, 1, 0, 0, 0, 2, 0, 0]]),
    # 8_8_8_2
    array([[0, 3, 0, 0, 0, 0, 0, 3, 0, 0],
           [2, 2, 0, 1, 0, 0, 3, 0, 2, 2],
           [3, 0, 2, 1, 0, 3, 2, 3, 2, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 3],
           [1, 3, 0, 1, 1, 1, 1, 0, 2, 0]]),
    # 8_8_8_3
    array([[0, 0, 1, 0, 3, 1, 0, 0, 0, 0],
           [0, 0, 0, 1, 2, 2, 3, 3, 1, 2],
           [0, 0, 2, 0, 0, 0, 2, 0, 0, 0],
           [3, 0, 3, 0, 2, 1, 3, 3, 1, 2],
           [0, 0, 2, 0, 1, 3, 1, 0, 0, 0]]),
    # 8_8_8_4
    array([[0, 3, 0, 0, 0, 0, 0, 0, 0, 0],
           [2, 0, 3, 0, 1, 1, 3, 0, 1, 2],
           [0, 0, 2, 0, 0, 1, 2, 3, 3, 0],
           [3, 2, 0, 0, 0, 1, 0, 0, 3, 2],
           [3, 0, 2, 2, 1, 0, 1, 1, 0, 0]]),
    # 8_8_8_5
    array([[0, 0, 0, 3, 3, 3, 3, 0, 0, 0],
           [0, 3, 3, 0, 2, 0, 0, 0, 1, 2],
           [1, 0, 2, 3, 0, 1, 0, 0, 0, 1],
           [2, 0, 0, 2, 2, 1, 0, 2, 0, 1],
           [1, 0, 3, 2, 0, 0, 0, 0, 0, 1]]),
    # 8_8_8_6
    array([[3, 3, 1, 0, 0, 0, 3, 3, 2, 0],
           [0, 1, 0, 3, 0, 0, 2, 0, 3, 3],
           [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
           [2, 1, 2, 2, 0, 0, 0, 3, 0, 0],
           [2, 0, 0, 0, 0, 0, 2, 1, 0, 2]]),
    # 8_8_8_7
    array([[0, 0, 0, 0, 0, 3, 0, 2, 3, 0],
           [0, 3, 3, 0, 1, 2, 2, 2, 0, 0],
           [0, 1, 1, 3, 2, 2, 0, 0, 0, 1],
           [1, 0, 1, 2, 0, 0, 3, 0, 1, 0],
           [0, 3, 0, 3, 0, 2, 0, 1, 0, 0]]),
    # 8_8_8_8
    array([[2, 0, 3, 0, 2, 0, 3, 0, 3, 3],
           [0, 1, 3, 2, 0, 0, 2, 0, 0, 0],
           [1, 0, 0, 1, 1, 0, 0, 0, 0, 2],
           [0, 2, 1, 3, 0, 1, 0, 0, 0, 0],
           [2, 2, 0, 0, 3, 3, 1, 1, 0, 0]]),
    # 8_8_8_9
    array([[0, 0, 2, 0, 0, 2, 0, 1, 0, 1],
           [3, 3, 0, 0, 2, 2, 2, 1, 0, 1],
           [3, 0, 0, 2, 2, 1, 1, 0, 0, 1],
           [0, 0, 3, 0, 3, 0, 2, 0, 1, 0],
           [0, 3, 3, 3, 0, 0, 0, 0, 0, 0]]),
    # 8_8_8_10
    array([[2, 0, 0, 3, 0, 2, 0, 0, 1, 0],
           [0, 1, 2, 0, 2, 0, 0, 3, 0, 3],
           [1, 0, 1, 3, 1, 3, 3, 0, 0, 0],
           [0, 2, 0, 3, 2, 3, 2, 0, 1, 0],
           [1, 0, 1, 0, 0, 0, 0, 0, 0, 2]])]
# initialize the tube detector
CALIB_MAT_PATH = fs.workdir / "data" / "calibration" / "qaqqq.json"
tube_detector = TestTubeDetector(CALIB_MAT_PATH, use_last_available_rack_yolo_pos=False,
                                 rack_height_lower=.01, rack_height_upper=.03)
for file_dir_id, file_dir in enumerate(file_dir_list):
    sampled_file_paths = list(file_dir.glob("**/*"))
    baseline_reco = reference_list[file_dir_id]
    r = eval_sample(baseline_reco=baseline_reco,
                    sampled_file_paths=sampled_file_paths,
                    group_info=file_dir_id)

# print(r)
# with open("result", "w") as f:
#     f.writelines('is_rack_reco\n')
#     [f.writelines(f'{i}\n') for i in r['is_rack_reco']]
#     f.writelines('num_tube_correct_reco\n')
#     [f.writelines(f'{i}\n') for i in r['num_tube_correct_reco']]
#     f.writelines('num_tube_reco\n')
#     [f.writelines(f'{i}\n') for i in r['num_tube_reco']]

from huri.learning.env.arrangement_planning_rack.utils import (mask_ucbc,
                                                               mask_crcl,
                                                               mask_ul,
                                                               mask_ur,
                                                               mask_bl,
                                                               mask_br, )

GRASP_FILTER_MASK = [mask_ucbc,
                     mask_crcl,
                     mask_ul,
                     mask_ur,
                     mask_bl,
                     mask_br, ]


class GraspFilter:
    def __init__(self, tube):
        self.grasp_list = tube.grasps_path

    def gen_feasible_grasps_id_for_masks(self, masks):
        for mask_id, mask in enumerate(masks):

            mask_name = f"mask_{mask_id}"



    def get_feasible_grasps(self, rack_state, move_slot_id):
        grasp_list = self.grasp_list


if __name__ == "__main__":
    from huri.core.file_sys import load_json
    from huri.definitions.tube_def import TubeType
    import numpy as np

    tube = TubeType.gen_tube_by_tubetype(1)
    grasp_info_list = load_json(path=tube)
    grasp_filter = GraspFilter(grasp_info_list)
    rack_state = np.array([[0, 0, 0, 1, 0, 3, 0, 3, 0, 0],
                           [0, 0, 1, 1, 3, 3, 1, 0, 1, 0],
                           [0, 3, 0, 3, 3, 1, 3, 3, 0, 0],
                           [0, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                           [0, 0, 0, 0, 3, 0, 3, 0, 0, 0]])
    move_slot_id = np.array([1, 2])
    grasp_filter.get_feasible_grasps(rack_state, move_slot_id)

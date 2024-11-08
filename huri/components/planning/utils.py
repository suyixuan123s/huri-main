from collections import namedtuple

_REQ_PACKET = namedtuple('_REQ_PACKET', ['chunk_init_id',
                                         'grasp_info_list',
                                         'hnd_name',
                                         'obj_init_homomat',
                                         'sym_obj_tgt_homomats',
                                         'pick_approach_pos_offset',
                                         'place_approach_pos_offset',
                                         'obstacle_list',
                                         'feasible_ranges_rad',
                                         'rack_tf'])

_MOTION_SEG = namedtuple('_MOTION_SEG', ["name", "conf_ls", "jaww_ls", "objpose_ls"])

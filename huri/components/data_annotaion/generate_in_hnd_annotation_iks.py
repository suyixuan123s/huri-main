import numpy as np
from huri.core.constants import SENSOR_INFO, ANNOTATION
from huri.definitions.tube_def import TubeType
from huri.core.common_import import cm, ym, rm, fs
import math
from huri.paper_draw.tase2022.gen_obs_poses.rayhit import rayhit_check_2, rayhit_check_2_lft
from modeling.model_collection import ModelCollection

def gen_icorotmats(icolevel=1,
                   rotation_interval=(0),
                   crop_normal=np.array([0, 0, 1]),
                   crop_angle=math.pi,
                   toggleflat=False):
    """
    generate rotmats using icospheres and rotationaangle each origin-vertex vector of the icosphere
    :param icolevel, the default value 1 = 42vertices
    :param rotation_interval
    :param crop_normal: crop results around a normal with crop_angle (crop out a cone section)
    :param crop_angle:
    :return: [[rotmat3, ...], ...] size of the inner list is size of the angles
    author: weiwei
    date: 20191015osaka
    """
    returnlist = []
    icos = rm.trm.creation.icosphere(icolevel)
    for vert in icos.vertices:
        if crop_angle < math.pi:
            if rm.angle_between_vectors(vert, crop_normal) > crop_angle:
                continue
        z = -vert
        x = rm.orthogonal_vector(z)
        y = rm.unit_vector(np.cross(z, x))
        temprotmat = np.eye(3)
        temprotmat[:, 0] = x
        temprotmat[:, 1] = y
        temprotmat[:, 2] = z
        returnlist.append([])
        for angle in rotation_interval:
            returnlist[-1].append(np.dot(rm.rotmat_from_axangle(z, angle), temprotmat))
    if toggleflat:
        return rm.functools.reduce(rm.operator.iconcat, returnlist, [])
    return returnlist


def generate(robot):
    tube = TubeType.gen_tube_by_name(tube_name="blue cap")
    # setup robot mover
    rbt_mover = RobotMover(yumi_s=robot, yumi_con=None, obj_cm=tube.gen_collision_model())

    pos_list_rgt = []
    for pos in np.array(np.meshgrid(np.linspace(.2, .5, num=4),
                                    np.linspace(-.25, 0, num=3, endpoint=False),
                                    np.array([.2]))).T.reshape(-1, 3):
        rots_candidate = np.array(gen_icorotmats(icolevel=3,
                                                 rotation_interval=[np.radians(360)],
                                                 crop_normal=np.array([0, 0, 1]),
                                                 crop_angle=np.pi / 6,
                                                 toggleflat=True))
        rots_candidate[..., [2, 1]] = rots_candidate[..., [1, 2]]
        # rots_candidate[..., 0] = -rots_candidate[..., 0]
        rots_candidate[..., 2] = -rots_candidate[..., 2]
        # print(rots_candidate)
        for rot in rots_candidate:
            pos_list_rgt.append(rm.homomat_from_posrot(pos, rot))
    print(f"There are {len(pos_list_rgt)} rgt poses")
    rbt_mover.add_wp_homomats(pos_list_rgt, armname="rgt_arm", load=False)

    pos_list_lft = []
    for pos in np.array(np.meshgrid(np.linspace(.2, .5, num=4),
                                    np.linspace(0, .25, num=3, endpoint=False),
                                    np.array([.2]))).T.reshape(-1, 3):
        rots_candidate = np.array(gen_icorotmats(icolevel=3,
                                                 rotation_interval=[0],
                                                 crop_normal=np.array([0, 0, 1]),
                                                 crop_angle=np.pi / 6,
                                                 toggleflat=True))
        rots_candidate[..., [0, 2]] = rots_candidate[..., [2, 0, ]]
        # rots_candidate[..., [0, 1, 2]] = rots_candidate[..., [2, 0, 1]]
        rots_candidate[..., [1, 0]] = rots_candidate[..., [0, 1]]
        rots_candidate[..., 1] = -rots_candidate[..., 1]
        rots_candidate[..., 0] = -rots_candidate[..., 0]
        for rot in rots_candidate:
            pos_list_lft.append(rm.homomat_from_posrot(pos, rot))
    print(f"There are {len(pos_list_lft)} lft poses")
    rbt_mover.add_wp_homomats(pos_list_lft, armname="lft_arm", load=False)


def show(robot, base, animation = False):
    wp_cache_path_rgt = SEL_PARAM_PATH.joinpath("wp_rgt.cache")
    wp_cache_path_lft = SEL_PARAM_PATH.joinpath("wp_lft.cache")
    rbt_mover = RobotMover(yumi_s=robot, yumi_con=None, obj_cm=None)
    rbt_mover.add_wp_homomats(wp_cache_path_rgt, armname="rgt_arm", load=True)
    rbt_mover.add_wp_homomats(wp_cache_path_lft, armname="lft_arm", load=True)
    base.boost.add_key(["a", "d"])
    md = [None]

    if not animation:
        def task_a(task):
            if base.inputmgr.keymap["a"]:
                base.inputmgr.keymap["a"] = False
                rbt_mover.goto_next_wp("rgt_arm")
                if md[0] is not None:
                    md[0].remove()
                md[0] = robot.gen_meshmodel(toggle_tcpcs=True)
                md[0].attach_to(base)
            return task.again

        def task_b(task):
            if base.inputmgr.keymap["d"]:
                base.inputmgr.keymap["d"] = False
                rbt_mover.goto_next_wp("lft_arm")
                if md[0] is not None:
                    md[0].remove()
                md[0] = robot.gen_meshmodel(toggle_tcpcs=True)
                md[0].attach_to(base)
            return task.again
        base.boost.add_task(task_a)
        base.boost.add_task(task_b)
    else:
        is_start = [False]
        def animation(task):
            if base.inputmgr.keymap["space"] or is_start[0]:
                is_start[0]  = True
                rbt_mover.goto_next_wp("rgt_arm")
                rbt_mover.goto_next_wp("lft_arm")
                if md[0] is not None:
                    md[0].remove()
                emc = ModelCollection()
                rbt_cm_list = robot.gen_meshmodel(toggle_tcpcs=True).cm_list
                for cmm in rbt_cm_list[4:] + [rbt_cm_list[1]]:  # 23:27 hand
                    cmm.attach_to(emc)
                md[0] =emc
                md[0].attach_to(base)
            return task.again
        base.boost.add_task(animation, timestep = 0.1)


def check(robot: ym.Yumi):
    rbt_mover = RobotMover(yumi_s=robot, yumi_con=None, obj_cm=None)
    rbt_mover.add_wp_homomats(None, armname="rgt_arm", load=True)
    rbt_mover.add_wp_homomats(None, armname="lft_arm", load=True)
    cnt = 0
    feasible_id = []
    for i in range(len(rbt_mover._wp_rgt)):
        # for i in range(1):
        print("ii", i)
        robot.fk("rgt_arm", rbt_mover._wp_rgt[i])
        robot.fk("lft_arm", rbt_mover._wp_lft[i])
        if robot.is_collided():
            continue
        rbt_modl = robot.gen_meshmodel()
        if rayhit_check_2_lft(rbt_model=rbt_modl,
                              phoxi_origin=np.array([0.31255359, - 0.15903892, 0.94915224]),
                              toggle_debug=False) or rayhit_check_2(rbt_model=rbt_modl,
                                                                    phoxi_origin=np.array(
                                                                        [0.31255359, - 0.15903892, 0.94915224]),
                                                                    toggle_debug=False):
            # robot.gen_meshmodel().attach_to(base)
            continue

        feasible_id.append(i)
    print(feasible_id)


def test():
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30,
         31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
         59,
         60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
         88,
         89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
         113,
         114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
         136,
         137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
         159,
         160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
         182,
         183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204,
         205,
         206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
         228,
         229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250,
         251,
         252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273,
         274,
         275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296,
         297,
         298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319,
         320,
         321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342,
         343,
         344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365,
         366,
         367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388,
         389,
         390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411,
         412,
         413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434,
         435,
         436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457,
         458,
         459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480,
         481,
         482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503,
         504,
         505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526,
         527,
         528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549,
         550,
         551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572,
         573,
         574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595,
         596,
         597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618,
         619,
         620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641,
         642,
         643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664,
         665,
         666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687,
         688,
         689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710,
         711,
         712, 713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733,
         734,
         735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756,
         757,
         758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779,
         780,
         781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802,
         803,
         804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825,
         826,
         827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841]
    print(len(a))


if __name__ == "__main__":
    from huri.core.common_import import cm, wd, ym, rm, fs
    from huri.components.data_annotaion.in_hand_annotation import RobotMover
    from huri.components.data_annotaion._constants import *
    from huri.core.base_boost import zoombase, boost_base

    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    base = boost_base(base)
    # init the simulation robot
    yumi_s = ym.Yumi(enable_cc=True)
    generate(robot=yumi_s)
    # show(yumi_s, base, animation=True)
    # check(robot=yumi_s)
    # test()
    base.run()

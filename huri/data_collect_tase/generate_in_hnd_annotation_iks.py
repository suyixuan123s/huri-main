import math

from tqdm import tqdm
import numpy as np

from utils.utils import is_yumi_lft_arm_pose_occluded, is_yumi_rgt_arm_pose_occluded
from modeling.model_collection import ModelCollection
from huri.core.common_import import cm, ym, rm, fs
from huri.definitions.tube_def import TubeType


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


def generate(robot, tube_name: str = "blue cap"):
    tube = TubeType.gen_tube_by_name(tube_name=tube_name)
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


def show(robot, base, animation=False):
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
                try:
                    rbt_mover.goto_next_wp("rgt_arm")
                except StopIteration:
                    rbt_mover._set_wp_ctr(0, 'rgt_arm')
                if md[0] is not None:
                    md[0].remove()
                md[0] = robot.gen_meshmodel(toggle_tcpcs=True)
                md[0].attach_to(base)
            return task.again

        def task_b(task):
            if base.inputmgr.keymap["d"]:
                base.inputmgr.keymap["d"] = False
                try:
                    rbt_mover.goto_next_wp("lft_arm")
                except StopIteration:
                    rbt_mover._set_wp_ctr(0, 'lft_arm')
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
                is_start[0] = True
                try:
                    rbt_mover.goto_next_wp("rgt_arm")
                    rbt_mover.goto_next_wp("lft_arm")
                except StopIteration:
                    rbt_mover._set_wp_ctr(0, 'rgt_arm')
                    rbt_mover._set_wp_ctr(0, 'lft_arm')
                if md[0] is not None:
                    md[0].remove()
                emc = ModelCollection()
                rbt_cm_list = robot.gen_meshmodel(toggle_tcpcs=True).cm_list
                for cmm in rbt_cm_list[4:] + [rbt_cm_list[1]]:  # 23:27 hand
                    cmm.attach_to(emc)
                md[0] = emc
                md[0].attach_to(base)
            return task.again

        base.boost.add_task(animation, timestep=0.1)


def check(robot: ym.Yumi):
    rbt_mover = RobotMover(yumi_s=robot, yumi_con=None, obj_cm=None)
    rbt_mover.add_wp_homomats(None, armname="rgt_arm", load=True)
    rbt_mover.add_wp_homomats(None, armname="lft_arm", load=True)
    feasible_id = []
    phoxi_origin = np.array([0.31255359, - 0.15903892, 0.94915224])
    print("Start occlusion check")
    for i in tqdm(range(min(len(rbt_mover._wp_rgt), len(rbt_mover._wp_lft),)), desc='Pose index'):
        robot.fk("rgt_arm", rbt_mover._wp_rgt[i])
        robot.fk("lft_arm", rbt_mover._wp_lft[i])
        if robot.is_collided():
            continue
        rbt_mdl = robot.gen_meshmodel()
        if is_yumi_lft_arm_pose_occluded(rbt_mdl,
                                         phoxi_origin) or is_yumi_rgt_arm_pose_occluded(rbt_mdl,
                                                                                        phoxi_origin):
            continue

        feasible_id.append(i)
    fs.dump_pickle([rbt_mover._wp_rgt[i] for i in feasible_id], SEL_PARAM_PATH.joinpath("wp_rgt.cache"), reminder=False)
    fs.dump_pickle([rbt_mover._wp_lft[i] for i in feasible_id], SEL_PARAM_PATH.joinpath("wp_lft.cache"), reminder=False)

    print(feasible_id)


if __name__ == "__main__":
    from huri.core.common_import import cm, wd, ym, rm, fs
    from in_hand_annotation import RobotMover
    from huri.core.base_boost import zoombase, boost_base
    from constant import SEL_PARAM_PATH

    # 3D environment
    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    base = boost_base(base)
    # init the simulation robot
    yumi_s = ym.Yumi(enable_cc=True)
    generate(robot=yumi_s, tube_name='blue cap')  # Generate observation poses
    check(robot=yumi_s)  # Check IF observation poses are occluded

    # if animation is True: Press [Space]
    # if animation is False: Press [a] and [d]
    show(yumi_s, base, animation=False)
    base.run()

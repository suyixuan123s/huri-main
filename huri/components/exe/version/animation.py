from huri.core.common_import import wd, np, ym, fs, gm, rm
import copy
from direct.stdpy import threading
import time
from direct.gui.OnscreenText import OnscreenText
from huri.core.constants import SENSOR_INFO
import cv2
from huri.vision.phoxi_capture import enhance_gray_img
from huri.components.utils.panda3d_utils import ImgOnscreen
from huri.definitions.utils_structure import MotionElement

MotionElement.counter_incremental = 6

base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
yumi_robot = ym.Yumi(enable_cc=True)
# mb_cnt, mb_data = fs.load_pickle("D:\chen\huri_shared\huri\components\exe\\version\\animation.pkl")
ANIMATION_PATH = fs.Path("animation.pkl")
PCD_PATH = fs.Path("vision_data.pkl")
IMG_PATH = fs.Path("img_tmp.jpg")

# unlink
[_.unlink() for _ in [ANIMATION_PATH, PCD_PATH, IMG_PATH, ] if _.exists()]

_mb_data = [None]
_mb_cnt = [-1]
_pcd_data_id = [None]
_pcd_data = [None, None, None]
plot_node = [None]

# image
img_size = (2064, 1544)
scale = 2
img_display = ImgOnscreen(size=(int(img_size[0] / scale), int(img_size[1] / scale)),
                          pos=(1 - int(img_size[0] / scale) / 3840, 1 - int(img_size[1] / scale) / 2160),
                          parent_np=base)

# add on screen image
font = loader.loadFont('arial.ttf')
textObject = OnscreenText(text='my text string',
                          pos=(-1.1, .8),
                          scale=0.1,
                          bg=(.3, .3, .3, 1),
                          fg=(203 / 255, 185 / 255, 148 / 255, 1),
                          font=font)


# TODO

def load_animation_files():
    while True:
        if ANIMATION_PATH.exists():
            try:
                mb_cnt, mb_data = fs.load_pickle(ANIMATION_PATH)
                if _mb_cnt[0] == mb_cnt:
                    pass
                else:
                    _mb_cnt[0] = mb_cnt
                    _mb_data[0] = mb_data
            except:
                print("load animation error")
        time.sleep(.5)


def load_pcd_files():
    while True:
        if PCD_PATH.exists():
            try:
                uuid, pcd_data, img_data = fs.load_pickle(PCD_PATH)
                _pcd_data[0] = uuid
                _pcd_data[1] = pcd_data
                _pcd_data[2] = img_data
            except:
                print("load pcd error")
        time.sleep(.5)


thread1 = threading.Thread(target=load_animation_files)
thread1.start()

thread2 = threading.Thread(target=load_pcd_files)
thread2.start()

affine_matrix = np.asarray(fs.load_json(SENSOR_INFO.PNT_CLD_CALIBR_MAT_PATH)["affine_mat"])

yumi_rbt = ym.Yumi()

yumi_rbt = yumi_rbt


def crop_image_by_percentage(img, crop_percentage):
    # Calculate the crop dimensions
    height, width = img.shape[:2]
    crop_height = int(height * crop_percentage)
    crop_width = int(width * crop_percentage)

    # Define the new dimensions
    start_row, start_col = crop_height, crop_width
    end_row, end_col = height - crop_height, width - crop_width

    # Crop the image
    cropped_img = img[start_row:end_row, start_col:end_col]

    return cropped_img


def crop_image_custom_percentages(img, upper_percentage, lower_percentage, right_percentage, left_percentage):
    # Image dimensions
    height, width = img.shape[:2]

    # Calculate the crop dimensions
    start_row = int(height * upper_percentage)
    end_row = height - int(height * lower_percentage)
    start_col = int(width * left_percentage)
    end_col = width - int(width * right_percentage)

    # Crop the image
    cropped_img = img[start_row:end_row, start_col:end_col]

    return cropped_img

def update(robot_s,
           motion_batch,
           pcd_data,
           task):
    if pcd_data[0] is not None and _pcd_data_id[0] != pcd_data[0]:
        if plot_node[0] is not None:
            plot_node[0].remove()
        _pcd_data_id[0] = pcd_data[0]
        collected_im = enhance_gray_img(pcd_data[2])
        collected_im = cv2.cvtColor(collected_im, cv2.COLOR_GRAY2BGR)
        color_c3 = collected_im.copy().reshape(-1, 3)
        color_c4 = np.ones((len(color_c3), 4), dtype=float)
        color_c4[..., :3] = color_c3 / 255
        plot_node[0] = gm.gen_pointcloud(points=rm.homomat_transform_points(affine_matrix, pcd_data[1]),
                                         rgbas=color_c4)
        plot_node[0].attach_to(base)

        im = cv2.imread(str(IMG_PATH))
        im = np.flip(im, axis=1)
        img_display.update_img(crop_image_custom_percentages(im, 0.2, 0.35, 0.45, 0.1,))
    textObject.text = f" Current Tube ID {0 if _mb_cnt[0] < 0 else _mb_cnt[0]} "
    # textObject.text = f" 当前试管ID {0 if _mb_cnt[0] < 0 else _mb_cnt[0]}" + " "
    if motion_batch[0] is None:
        return task.again
    motion_element = motion_batch[0].current
    try:
        if motion_element.is_end():
            raise StopIteration
            # return task.again
        objcm, obj_pose, pose, jawwdith, hand_name, obs_list = next(motion_element)
    except StopIteration:
        try:
            next(motion_batch[0])  # try to get the next motion element
        except StopIteration:
            motion_batch[0]._counter = 0
        return task.again
    except Exception as e:
        return Exception(e)
    robot_s.fk(hand_name, pose)
    robot_s.jaw_to(hand_name, jawwdith)
    robot_meshmodel = robot_s.gen_meshmodel()
    robot_meshmodel.attach_to(base)
    objb_copied = objcm.copy()
    objb_copied.set_homomat(obj_pose)
    objb_copied.attach_to(base)
    obs_list_copied = copy.deepcopy(obs_list)
    obs_list_copied.attach_to(base)
    motion_element.reset_robot_gm(robot_meshmodel)
    motion_element.reset_obj_gm(objb_copied)
    motion_element.reset_obs_list_gm(obs_list_copied)
    return task.again


taskMgr.doMethodLater(0.005, update, "update",
                      extraArgs=[yumi_robot,
                                 _mb_data,
                                 _pcd_data],
                      appendTask=True)
base.run()

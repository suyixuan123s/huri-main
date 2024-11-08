from typing import Union
from pathlib import Path

import numpy as np
import cv2
import albumentations as A
import random
import time

from huri.paper_draw.tase2022.utils import parse_collected_data
from huri.vision.phoxi_capture import enhance_gray_img
from huri.core.common_import import fs
from huri.components.data_annotaion.utils import mask_2_bbox_xyxy, img_to_n_channel
from huri.components.utils.annotation_utils import scale_label_format, bboxes_xyxy2xywh, bboxes_xywh2xyxy, \
    write_yolo_annotation_file


def extract_img_bbox(img, bboxes):
    extracted_imgs = []
    for bbox_xyxy in bboxes:
        bbox_xyxy_int = np.array(bbox_xyxy).astype(int)
        extracted_imgs.append(img[bbox_xyxy_int[1]: bbox_xyxy_int[3], bbox_xyxy_int[0]: bbox_xyxy_int[2]].copy())

    return extracted_imgs


def box_area(arr):
    # arr: np.array([[x1, y1, x2, y2]])
    width = arr[:, 2] - arr[:, 0]
    height = arr[:, 3] - arr[:, 1]
    return width * height


def _box_inter_union(arr1, arr2):
    # arr1 of [N, 4]
    # arr2 of [N, 4]
    area1 = box_area(arr1)
    area2 = box_area(arr2)

    # Intersection
    top_left = np.maximum(arr1[:, :2], arr2[:, :2])  # [[x, y]]
    bottom_right = np.minimum(arr1[:, 2:], arr2[:, 2:])  # [[x, y]]
    wh = bottom_right - top_left
    # clip: if boxes not overlap then make it zero
    intersection = wh[:, 0].clip(0) * wh[:, 1].clip(0)

    # union
    union = area1 + area2 - intersection
    return intersection, union


def box_iou(boxes1_xyxy, boxes2_xyxy):
    """
    Copy from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    :param boxes1_xyxy: arr1[N, 4]
    :param boxes2_xyxy: arr2[N, 4]
    :return:
    """
    inter, union = _box_inter_union(boxes1_xyxy, boxes2_xyxy)
    iou = inter / union
    return iou


def box_iou_yolo(boxes1_xywh, boxes2_xywh):
    """
    Copy from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    :param boxes1_xywh: arr1[N, 4]
    :param boxes2_xywh: arr2[N, 4]
    :return:
    """
    return box_iou(bboxes_xywh2xyxy(boxes1_xywh), bboxes_xywh2xyxy(boxes2_xywh))


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray):
    """Calculate the Intersection of Unions (IoUs) between masks.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`mask_a` and :obj:`mask_b` need to be
    same type.
    The output is same type as the type of the inputs.
    Args:
        mask_a (array): An array whose shape is :math:`(N, H, W)`.
            :math:`N` is the number of masks.
            The dtype should be :obj:`numpy.bool`.
        mask_b (array): An array similar to :obj:`mask_a`,
            whose shape is :math:`(K, H, W)`.
            The dtype should be :obj:`numpy.bool`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th mask in :obj:`mask_a` and :math:`k` th mask \
        in :obj:`mask_b`.
    """
    if mask_a.shape[1:] != mask_b.shape[1:]:
        raise IndexError

    n_mask_a = len(mask_a)
    n_mask_b = len(mask_b)
    iou = np.empty((n_mask_a, n_mask_b), dtype=np.float32)
    for n, m_a in enumerate(mask_a):
        for k, m_b in enumerate(mask_b):
            intersect = np.bitwise_and(m_a, m_b).sum()
            union = np.bitwise_or(m_a, m_b).sum()
            iou[n, k] = intersect / union
    return iou


def copy_paste_img(bg_img: np.ndarray,
                   p_img: np.ndarray,
                   mask: np.ndarray,
                   p_img_left_top_coords=(0, 0),
                   match_hist=False):
    bg_img = bg_img.copy()

    p_imgs_h, p_imgs_w = mask.shape[:2]

    coord = (slice(int(p_img_left_top_coords[1]), int(p_img_left_top_coords[1] + p_imgs_h)),
             slice(int(p_img_left_top_coords[0]), int(p_img_left_top_coords[0] + p_imgs_w)))

    if p_img is not None:
        im = p_img * mask + bg_img[coord] * (1 - mask)
    else:
        im = bg_img[coord] * (1 - mask)
    bg_img[coord] = im
    return bg_img


def highlight_mask(img, mask, color=(0, 255, 0), show_bbox=False):
    # color to fill
    color = np.array(color, dtype='uint8')

    # equal color where mask, else image
    # this would paint your object silhouette entirely with `color`
    masked_img = np.where(mask[..., None], color, img)

    # use `addWeighted` to blend the two images
    # the object will be tinted toward `color`
    out = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)

    if show_bbox:
        idx_in_pixel = np.where(mask)
        h1, h2 = idx_in_pixel[0].min(), idx_in_pixel[0].max()
        w1, w2 = idx_in_pixel[1].min(), idx_in_pixel[1].max()
        out = cv2.rectangle(out, (w1, h1), (w2, h2),
                            color=color.tolist(), thickness=2)
    # merge the channels back together
    return out


def draw_YOLO_BBox_Masks(img, bboxes, classes, masks=None):
    img = img.copy()
    bboxes = bboxes_xywh2xyxy(scale_label_format(bboxes, img.shape, op="scale"))
    color = [
        (0, 255, 0),
        (204, 50, 153),
        (0, 0, 255),
        (0, 191, 255),
    ]
    for i, bbox_xyxy in enumerate(bboxes):
        bbox_xyxy_int = bbox_xyxy.astype(int)
        class_id = classes[i]
        if class_id == 0:
            continue
        # cv2.rectangle(img, (bbox_xyxy_int[0], bbox_xyxy_int[1]), (bbox_xyxy_int[2], bbox_xyxy_int[3]),
        #               color[class_id - 1], 2)
        if masks is not None:
            bbox_xyxy_int = bbox_xyxy.astype(int)
            m = np.zeros((img.shape[0], img.shape[1]))
            mask_shape = masks[i].shape
            m[bbox_xyxy_int[1]:bbox_xyxy_int[3] + 2, bbox_xyxy_int[0]:bbox_xyxy_int[2] + 2][:mask_shape[0],
            :mask_shape[1]] = masks[i]
            img = highlight_mask(img, m, color=color[class_id - 1])

            # get largest contour
            contours = cv2.findContours(cv2.inRange(m, 1, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contours[0] if len(contours) == 2 else contours[1]
            big_contour = max(contours, key=cv2.contourArea)
            # draw filled contour on black background
            cv2.drawContours(img, [big_contour], 0, color[class_id - 1], 3, )

    return img


def crop_img_mask(imgs: list,
                  masks: list,
                  img_classes: list,
                  rand_scale=False,
                  rand_blur=False,
                  rand_brightness=False):
    """
    Return bounding box of cropped images
    :param imgs: List[np.array, np.array ... ] a list of images
    :param masks: List[List,List ...] a list of masks for each image in the imgs
    :return:
    """
    crop_imgs = []
    crop_img_masks = []
    rep_copy_imgs_classes = []
    for ind in range(len(imgs)):
        img_mask = masks[ind]
        if len(img_mask) < 1:
            continue
        copy_img_bbox = [mask_2_bbox_xyxy(mask) for mask in img_mask]
        _crop_imgs = extract_img_bbox(imgs[ind], copy_img_bbox)
        _crop_img_mask = [extract_img_bbox(mask, [copy_img_bbox[_]])[0] for _, mask in enumerate(img_mask)]
        aug = []
        if rand_scale:
            aug.append(A.RandomScale(scale_limit=(-0.1, 0), p=.5))
        if rand_blur:
            aug.append(A.Blur(blur_limit=3, p=.5))
        if rand_brightness:
            aug.append(A.RandomBrightnessContrast(p=.5, brightness_limit=.1, contrast_limit=.1))
        if len(aug) > 0:
            transform = A.Compose(aug, )
            aug = [transform(image=_, mask=__) for (_, __) in zip(_crop_imgs, _crop_img_mask)]
            _crop_imgs = [_["image"] for _ in aug]
            _crop_img_mask = [_["mask"] for _ in aug]
        crop_imgs.extend(_crop_imgs)
        crop_img_masks.extend(_crop_img_mask)
        rep_copy_imgs_classes.extend([img_classes[ind] for _ in range(len(_crop_imgs))])

    return crop_imgs, crop_img_masks, rep_copy_imgs_classes


def copy_paste_augmentation(copy_imgs: list,
                            copy_img_classes: list,
                            copy_img_masks: list,
                            bg_img: np.ndarray,
                            bg_img_class: int,
                            bg_img_mask: np.ndarray,
                            iou_threshold=.02,
                            rand_scale=False,
                            rand_blur=False,
                            rand_brightness=False,
                            intersection_area_thresholds=.2,
                            toggle_debug=False,
                            save_tube_cap=False):
    print(f"Intersection area threshold is {intersection_area_thresholds}")
    assert len(copy_imgs) == len(copy_img_masks)
    # crop images from copy images
    crop_imgs, crop_img_masks, rep_copy_imgs_classes = crop_img_mask(imgs=copy_imgs,
                                                                     masks=copy_img_masks,
                                                                     img_classes=copy_img_classes,
                                                                     rand_blur=rand_blur,
                                                                     rand_brightness=rand_brightness,
                                                                     rand_scale=rand_scale, )
    if save_tube_cap:
        for ii, _ in enumerate(crop_imgs):
            rgba = [_, _, _, (crop_img_masks[ii] * 255).astype(np.uint8)]
            dst = cv2.merge(rgba, 4)
            cv2.imwrite(f"debug/cap_{ii}.png", dst)

    if isinstance(intersection_area_thresholds, float):
        th = intersection_area_thresholds
        intersection_area_thresholds = [th if _ != 1 else .4 for _ in rep_copy_imgs_classes]
        # intersection_area_thresholds = [th for _ in rep_copy_imgs_classes]
    if isinstance(intersection_area_thresholds, list):
        pass
    else:
        raise Exception("Incorrect format for intersection area threshold. It should be a number or list")

    if toggle_debug:
        for i in range(len(crop_imgs)):
            # cv2.imshow(f"img{i}", crop_imgs[i])
            cv2.imwrite(f"debug/img{i}.jpg", crop_imgs[i])
            # cv2.imshow("img2", crop_img_masks[i] * 255)

    old_mask_areas = []
    new_maskes = []
    new_bboxes = []
    new_bboxes_xyxy = []
    new_bboxes_class = []
    new_image = bg_img.copy()

    # mask_sampled_area = cv2.dilate(bg_img_mask, kernal, iterations=1)
    bg_mask_sampled_area = bg_img_mask.copy()
    # kernal = np.eye(3)
    kernal = np.ones((np.average([mask.shape for mask in crop_img_masks], axis=0) / 2).astype(int))
    bg_mask_sampled_area = cv2.erode(bg_mask_sampled_area.astype(np.uint8), kernal.astype(np.uint8), iterations=1)

    for ind, c_crop_img in enumerate(crop_imgs):
        mask = crop_img_masks[ind]
        bboxes_xyxy_on_img = np.array(new_bboxes_xyxy)
        n_retry = 30
        cnt = 0
        is_continue = False
        possible_coords = np.vstack(np.where(bg_mask_sampled_area)).T
        random_id = random.sample(range(len(possible_coords)), n_retry)
        while True:
            random_coord = possible_coords[random_id[cnt]]
            cnt += 1
            if cnt >= n_retry:
                # print("?")
                is_continue = True
                break

            _w1 = random_coord[1]
            _h1 = random_coord[0]
            w1 = int(_w1 - c_crop_img.shape[1] / 2)
            h1 = int(_h1 - c_crop_img.shape[0] / 2)
            w2 = int(_w1 + c_crop_img.shape[1] / 2)
            h2 = int(_h1 + c_crop_img.shape[0] / 2)
            if w1 < 0 or h1 < 0 or w2 >= bg_img.shape[1] - 2 or h2 >= bg_img.shape[0] - 2:
                continue
            annot_xyxy = np.array([w1, h1, w2, h2])
            annotation = bboxes_xyxy2xywh(scale_label_format(annot_xyxy, img_size=new_image.shape))
            if len(bboxes_xyxy_on_img) < 1:
                break
            IoUs = box_iou(np.repeat(annot_xyxy[None,], len(bboxes_xyxy_on_img), axis=0),
                           bboxes_xyxy_on_img)
            if len(np.where(IoUs > intersection_area_thresholds[0])[0]) > 0:
                continue
            occ_ids = np.where(IoUs > 0)[0]
            # print(IoUs, occ_ids)
            if len(occ_ids) > 0:
                is_occ_vaild = True
                for occ_id in occ_ids:
                    m_tmp = new_maskes[occ_id].copy()
                    b_xyxy_tmp = new_bboxes_xyxy[occ_id].copy()
                    intersec = np.array([max(b_xyxy_tmp[0], annot_xyxy[0]),
                                         max(b_xyxy_tmp[1], annot_xyxy[1]),
                                         min(b_xyxy_tmp[2], annot_xyxy[2]),
                                         min(b_xyxy_tmp[3], annot_xyxy[3]), ])
                    i_x0, i_x1 = intersec[0], intersec[2]
                    i_y0, i_y1 = intersec[1], intersec[3]
                    # occ
                    tmp_x0, tmp_x1 = i_x0 - b_xyxy_tmp[0], i_x1 - b_xyxy_tmp[0]
                    tmp_y0, tmp_y1 = i_y0 - b_xyxy_tmp[1], i_y1 - b_xyxy_tmp[1]
                    # annotation
                    ms_x0, ms_x1 = i_x0 - annot_xyxy[0], i_x1 - annot_xyxy[0]
                    ms_y0, ms_y1 = i_y0 - annot_xyxy[1], i_y1 - annot_xyxy[1]

                    if (tmp_y1 - tmp_y0) == 0 or (tmp_x0 - tmp_x1) == 0 or (ms_y0 - ms_y1) == 0 or (ms_x0 - ms_x1) == 0:
                        break

                    m_tmp[tmp_y0:tmp_y1, tmp_x0:tmp_x1][mask[ms_y0: ms_y1, ms_x0:ms_x1].astype(bool)] = 0
                    # smooth edge
                    kernel = np.ones((3, 3))
                    erode = cv2.erode(m_tmp, kernel)
                    m_tmp = cv2.dilate(erode, kernel)

                    # print("occ_id", occ_id)
                    # print(intersection_area_thresholds)
                    # print(intersection_area_thresholds[occ_id])
                    if np.count_nonzero(m_tmp == 1) / old_mask_areas[occ_id] <= (
                            1 - intersection_area_thresholds[occ_id]):
                        is_occ_vaild = False
                        break
                    else:
                        m_tmp_new_bboxes_xyxy = np.array(mask_2_bbox_xyxy(m_tmp))
                        new_maskes[occ_id] = m_tmp[m_tmp_new_bboxes_xyxy[1]:m_tmp_new_bboxes_xyxy[3] + 1,
                                             m_tmp_new_bboxes_xyxy[0]:m_tmp_new_bboxes_xyxy[2] + 1]
                        old_mask_areas[occ_id] = np.count_nonzero(new_maskes[occ_id] == 1)
                        # cv2.imshow("sfds", new_maskes[occ_id] * 255)
                        # cv2.waitKey(0)
                        # print(m_tmp_new_bboxes_xyxy)
                        m_tmp_new_bboxes_xyxy[[0, 2]] += b_xyxy_tmp[0]
                        m_tmp_new_bboxes_xyxy[[1, 3]] += b_xyxy_tmp[1]
                        new_bboxes_xyxy[occ_id] = m_tmp_new_bboxes_xyxy
                        new_bboxes[occ_id] = bboxes_xyxy2xywh(
                            scale_label_format(m_tmp_new_bboxes_xyxy, img_size=new_image.shape))
                if is_occ_vaild:
                    break
                else:
                    continue
            else:
                break
        if is_continue:
            continue
        old_mask_areas.append(np.count_nonzero(mask == 1))
        new_bboxes.append(annotation)
        new_bboxes_xyxy.append(annot_xyxy)
        new_bboxes_class.append(rep_copy_imgs_classes[ind])
        new_maskes.append(mask)
        new_image = copy_paste_img(bg_img=new_image,
                                   p_img=c_crop_img,
                                   p_img_left_top_coords=(w1, h1),
                                   mask=mask,
                                   match_hist=False)
        # reduce mask
        bg_mask_sampled_area[h1:h2, w1:w2] = bg_mask_sampled_area[h1:h2, w1:w2] * (1 - mask)
        # cv2.imshow("mask", mask_sampled_area * 255)
        # cv2.waitKey(0)
    if len(new_bboxes_class) < 1:
        return None, None, None, None
    if bg_img_class > -1:
        return (new_image,  # new image
                np.append(np.array(new_bboxes_class), bg_img_class),  # new class
                np.concatenate((np.array(new_bboxes), bboxes_xyxy2xywh(
                    scale_label_format(np.array([mask_2_bbox_xyxy(bg_img_mask)]), img_size=new_image.shape))),
                               axis=0),
                new_maskes)  # new annotations
    else:
        return (new_image,
                np.array(new_bboxes_class),
                np.array(new_bboxes),
                new_maskes)


def copy_paste_annotation(cp_paths_and_classes,
                          bg_paths_and_classes,
                          save_path="data2",
                          num_imgs=20,
                          num_tubes=20,
                          intersection_area_thresholds=.1,
                          random_bg=False,
                          toggle_debug=False,
                          no_rack_bg=False,
                          no_bg=False,
                          save_tube_cap=False):
    sampled_cache = {}
    bg_sampled_cache = {}
    random_bg_sampled_cache = {}
    # load file path for copy image
    sampled_file_paths = []
    for (path, data_class) in cp_paths_and_classes:
        if isinstance(path, list):
            file_paths = path
        else:
            path = fs.Path(path)
            file_paths = list(path.glob("*.pkl"))
        sampled_file_paths.extend([(_, data_class) for _ in file_paths])
    # load file path for background image
    bg_sampled_file_paths = []
    for (path, data_class) in bg_paths_and_classes:
        path = fs.Path(path)
        file_paths = list(path.glob("*.pkl"))
        bg_sampled_file_paths.extend([(_, data_class) for _ in file_paths])

    path = fs.Path(fs.workdir.joinpath("paper_draw", "train"))
    random_bg_sampled_file_paths = list(path.glob("*"))

    for _ in range(num_imgs):
        random_indices = np.random.choice(len(sampled_file_paths), size=int(num_tubes / 2), replace=True)
        crop_imgs = []
        copy_img_masks = []
        copy_imgs_classes = []
        for i in random_indices:
            if i in sampled_cache:
                img_class, img, mask = sampled_cache[i]
            else:
                file_path, img_class = sampled_file_paths[i]
                img, mask = parse_collected_data(file_path, data_type="in_hand" if img_class > 0 else "on_table")
                if img is None:
                    continue
                img = enhance_gray_img(img)
                sampled_cache[i] = (img_class, img, mask)
            crop_imgs.append(img)
            copy_img_masks.append(mask)
            copy_imgs_classes.append(img_class)

        random_ind = np.random.randint(len(bg_sampled_file_paths))

        # random background image for the tube
        if not no_rack_bg:
            if random_ind in bg_sampled_cache:
                bg_img_class, bg_img, bg_mask = bg_sampled_cache[random_ind]
            else:
                bg_file_path, bg_img_class = bg_sampled_file_paths[random_ind]
                bg_img, bg_mask = parse_collected_data(bg_file_path, data_type="on_table")
                if bg_img is None:
                    continue
                bg_mask = bg_mask[0]
                bg_img = enhance_gray_img(bg_img)
                bg_sampled_cache[random_ind] = (bg_img_class, bg_img, bg_mask)
            while True:
                r = A.Rotate(limit=180, p=.85)(image=bg_img, mask=bg_mask)
                if abs(np.count_nonzero(r["mask"]) - np.count_nonzero(bg_mask)) < .005 * np.count_nonzero(bg_mask):
                    bg_img = r["image"]
                    bg_mask = r["mask"]
                    # cv2.imshow("a", bg_img)
                    # cv2.waitKey(0)
                    break

        # random background image for the rack
        if (random_bg and np.random.randint(2)) or no_rack_bg:
            rand_bg_id = np.random.randint(len(random_bg_sampled_file_paths))
            if rand_bg_id in random_bg_sampled_cache:
                rand_bg = random_bg_sampled_cache[rand_bg_id]
            else:
                rand_bg = cv2.imread(str(random_bg_sampled_file_paths[rand_bg_id]), 0)
                if rand_bg.shape[0] > rand_bg.shape[1]:
                    rand_bg = cv2.rotate(rand_bg, cv2.ROTATE_90_CLOCKWISE)

                rand_bg = cv2.resize(rand_bg, img.shape[:2][::-1])
                random_bg_sampled_cache[rand_bg_id] = rand_bg
            if not no_rack_bg:
                bg_img = (bg_img * bg_mask + rand_bg * (1 - bg_mask)).astype(np.uint8)
            else:
                bg_img = rand_bg
                bg_mask = np.ones(bg_img.shape[:2])
                bg_img_class = -1
        if no_bg:
            # nbg = np.zeros_like(bg_img, dtype=np.uint8)
            nbg = np.ones_like(bg_img, dtype=np.uint8) * 255
            bg_img = (bg_img * bg_mask + nbg * (1 - bg_mask)).astype(np.uint8)
        # # random occlusion image for the rack
        # rand_occ_imgs = []
        # rand_occ_masks = []
        # rand_occ_classes = []
        # if random_occlusion:
        #     random_occ_num = int(np.random.randint(num_tubes) / 2)
        #     if random_occ_num > 0:
        #         random_indices = np.random.choice(len(sampled_file_paths), size=int(num_tubes / 2), replace=True)
        #         for i in random_indices:
        #             if i in sampled_cache:
        #                 img_class, img, mask = sampled_cache[i]
        #             else:
        #                 file_path, img_class = sampled_file_paths[i]
        #                 img, mask = parse_collected_data(file_path, data_type="in_hand")
        #                 if img is None:
        #                     continue
        #                 img = enhance_gray_img(img)
        #                 sampled_cache[i] = (img_class, img, mask)
        #             rand_occ_imgs.append(img)
        #             rand_occ_masks.append(mask)
        #             rand_occ_classes.append(img_class)
        try:
            (img, b, annotations, masks) = copy_paste_augmentation(copy_imgs=crop_imgs,
                                                                   copy_img_classes=copy_imgs_classes,
                                                                   copy_img_masks=copy_img_masks,
                                                                   bg_img=bg_img,
                                                                   bg_img_mask=bg_mask,
                                                                   bg_img_class=-1,
                                                                   rand_scale=True,
                                                                   rand_blur=True,
                                                                   rand_brightness=True,
                                                                   intersection_area_thresholds=intersection_area_thresholds,
                                                                   toggle_debug=toggle_debug,
                                                                   save_tube_cap=save_tube_cap)
        except Exception as e:
            print(e)
            continue
        if not toggle_debug:
            img_name = time.strftime('%Y%m%d-%H%M%S')
            cv2.imwrite(f"{save_path}/images/{img_name}.jpg", img)
            write_yolo_annotation_file(b, bboxes=annotations, save_path=f"{save_path}/labels/{img_name}.txt")
        if _ % 100 == 0:
            print(f"Generated image {_}")
        if toggle_debug:
            cv2.imwrite("debug/o.jpg", img_to_n_channel(img))
            img = draw_YOLO_BBox_Masks(img_to_n_channel(img), annotations, classes=b, masks=masks)
            cv2.imwrite("debug/r.jpg", img)
            cv2.imshow("test", img)
            cv2.waitKey(0)


def random_sample_paths(path, num=300, save_name=None):
    path = fs.Path(path)
    file_paths = list(path.glob("*.pkl"))
    selected_p = random.sample(file_paths, num)
    if save_name is not None:
        fs.dump_pickle(selected_p, save_name)
    return selected_p


if __name__ == "__main__":
    from huri.core.constants import DATA_ANNOT_PATH


    def test_1():
        path = "C:\\Users\\WRS\\Desktop\\data\\BLUE_TUBE"
        path = fs.Path(path)
        sampled_file_paths = list(path.glob("**/*"))
        file_path = sampled_file_paths[600]

        bg_path = DATA_ANNOT_PATH.joinpath("EXP", "RACK")
        bg_sampled_file_paths = list(bg_path.glob("**/*"))
        bg_file_path = bg_sampled_file_paths[10]

        crop_imgs = []
        copy_img_masks = []
        copy_imgs_classes = []

        bg_class = 0

        img_class = 1
        img, mask = parse_collected_data(file_path, data_type="in_hand")
        img = enhance_gray_img(img)
        crop_imgs.append(img)
        copy_img_masks.append(mask)
        copy_imgs_classes.append(img_class)

        bg_img, bg_mask = parse_collected_data(bg_file_path, data_type="on_table")
        bg_img = enhance_gray_img(bg_img)
        # cv2.imshow("test", bg_mask[0])
        # cv2.waitKey(0)

        img, b, annotations = copy_paste_augmentation(copy_imgs=crop_imgs,
                                                      copy_img_classes=copy_imgs_classes,
                                                      copy_img_masks=copy_img_masks,
                                                      bg_img=bg_img,
                                                      bg_img_mask=bg_mask[0],
                                                      bg_img_class=bg_class)
        img = draw_YOLO_BBox(img_to_n_channel(img), annotations)
        cv2.imshow("test", img)
        cv2.waitKey(0)


    def test_2():
        # copy_paste_annotation(cp_paths_and_classes=[["C:\\Users\\WRS\\Desktop\\data\\BLUE_TUBE", 1],
        #                                             ["C:\\Users\\WRS\\Desktop\\data\\PURPLE_TUBE", 2],
        #                                             [DATA_ANNOT_PATH.joinpath("EXP", "WHITE_TUBE"), 3],
        #                                             [DATA_ANNOT_PATH.joinpath("EXP", "WHITE_PURPLE_TUBE"), 4], ],
        #                       bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
        #                       num_imgs=200,
        #                       num_tubes=20, toggle_debug=False)

        copy_paste_annotation(cp_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), 1],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), 2],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), 3],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=800,
                              num_tubes=25, toggle_debug=True, random_bg=True, )


    def test_3(toggle_debug=True, save_path="data_syn_p_1600"):
        # copy_paste_annotation(cp_paths_and_classes=[["C:\\Users\\WRS\\Desktop\\data\\BLUE_TUBE", 1],
        #                                             ["C:\\Users\\WRS\\Desktop\\data\\PURPLE_TUBE", 2],
        #                                             [DATA_ANNOT_PATH.joinpath("EXP", "WHITE_TUBE"), 3],
        #                                             [DATA_ANNOT_PATH.joinpath("EXP", "WHITE_PURPLE_TUBE"), 4], ],
        #                       bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
        #                       num_imgs=200,
        #                       num_tubes=20, toggle_debug=False)

        copy_paste_annotation(cp_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), 1],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), 2],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), 3],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=300,
                              save_path=save_path,
                              num_tubes=35, toggle_debug=toggle_debug, random_bg=True,
                              no_rack_bg=True)


    def test_4():
        # copy_paste_annotation(cp_paths_and_classes=[["C:\\Users\\WRS\\Desktop\\data\\BLUE_TUBE", 1],
        #                                             ["C:\\Users\\WRS\\Desktop\\data\\PURPLE_TUBE", 2],
        #                                             [DATA_ANNOT_PATH.joinpath("EXP", "WHITE_TUBE"), 3],
        #                                             [DATA_ANNOT_PATH.joinpath("EXP", "WHITE_PURPLE_TUBE"), 4], ],
        #                       bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
        #                       num_imgs=200,
        #                       num_tubes=20, toggle_debug=False)

        copy_paste_annotation(cp_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), 1],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), 2],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), 3],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=800,
                              save_path="data4",
                              num_tubes=25, toggle_debug=True, random_bg=True,
                              no_rack_bg=False)


    def test_5():
        # random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), num=200, save_name="blue_tube.pkl")
        random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), num=200, save_name="purple_tube.pkl")
        # random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), num=200, save_name="white_tube.pkl")
        # random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), num=200, save_name="white_purple_tube.pkl")


    def test_6():
        # real 800 random background
        copy_paste_annotation(cp_paths_and_classes=[[fs.load_pickle("blue_tube.pkl"), 1],
                                                    [fs.load_pickle("purple_tube.pkl"), 2],
                                                    [fs.load_pickle("white_tube.pkl"), 3],
                                                    [fs.load_pickle("white_purple_tube.pkl"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=200,
                              save_path="data4",
                              num_tubes=25, toggle_debug=False, random_bg=True,
                              no_rack_bg=False)


    def test_7():
        # extract 800 real images
        from huri.components.data_annotaion.auto_label_ral2022 import auto_labeling
        label_info = {
            'rack': 0, 'blue': 1, 'purple': 2, 'white': 3, 'purple ring': 4,
        }
        tube_type = 'purple ring'
        save_img_path = "D:\chen\yolo_huri\paper\\RAL2022_dataset\\purple ring_800\images"
        save_label_path = "D:\chen\yolo_huri\paper\\RAL2022_dataset\\purple ring_800\labels"
        auto_labeling(file_dir=fs.load_pickle("white_purple_tube.pkl"),
                      label_id=label_info[tube_type],
                      save_img_path=save_img_path,
                      save_label_path=save_label_path, )


    def test_8():
        # real 800 fixed background
        copy_paste_annotation(cp_paths_and_classes=[[fs.load_pickle("blue_tube.pkl"), 1],
                                                    [fs.load_pickle("purple_tube.pkl"), 2],
                                                    [fs.load_pickle("white_tube.pkl"), 3],
                                                    [fs.load_pickle("white_purple_tube.pkl"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=100,
                              save_path="data5",
                              num_tubes=25, toggle_debug=False, random_bg=False,
                              no_rack_bg=False)


    def test_9():
        # real 800 random no rack background
        copy_paste_annotation(cp_paths_and_classes=[[fs.load_pickle("blue_tube.pkl"), 1],
                                                    [fs.load_pickle("purple_tube.pkl"), 2],
                                                    [fs.load_pickle("white_tube.pkl"), 3],
                                                    [fs.load_pickle("white_purple_tube.pkl"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=500,
                              save_path="data6",
                              num_tubes=35, toggle_debug=True, random_bg=True,
                              no_rack_bg=True)


    def test_10(num=400):
        each_num = int(num / 4)
        random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), num=each_num,
                            save_name=f"blue_tube_{each_num}.pkl")
        random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), num=each_num,
                            save_name=f"purple_tube_{each_num}.pkl")
        random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), num=each_num,
                            save_name=f"white_tube_{each_num}.pkl")
        random_sample_paths(DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), num=each_num,
                            save_name=f"white_purple_tube_{each_num}.pkl")


    def test_11(num=400, num_tubes=30, intersection_area_thresholds=.15):
        # real num random background
        each_num = int(num / 4)
        if num == 800:
            a = [[fs.load_pickle("blue_tube.pkl"), 1],
                 [fs.load_pickle("purple_tube.pkl"), 2],
                 [fs.load_pickle("white_tube.pkl"), 3],
                 [fs.load_pickle("white_purple_tube.pkl"), 4], ]
        elif num == 1600:
            a = [[DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), 1],
                 [DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), 2],
                 [DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), 3],
                 [DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), 4], ]
        else:
            a = [[fs.load_pickle(f"blue_tube_{each_num}.pkl"), 1],
                 [fs.load_pickle(f"purple_tube_{each_num}.pkl"), 2],
                 [fs.load_pickle(f"white_tube_{each_num}.pkl"), 3],
                 [fs.load_pickle(f"white_purple_tube_{each_num}.pkl"), 4], ]
        copy_paste_annotation(cp_paths_and_classes=a,
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=820,
                              save_path=f"data_real_{num}_syn_p",
                              num_tubes=num_tubes, toggle_debug=False, random_bg=True,
                              intersection_area_thresholds=intersection_area_thresholds,
                              no_rack_bg=False)


    def test_12(num=400):
        # extract 400 real images
        # real num random background
        each_num = int(num / 4)
        from huri.components.data_annotaion.auto_label_ral2022 import auto_labeling
        label_info = {
            'rack': 0, 'blue': 1, 'purple': 2, 'white': 3, 'purple ring': 4,
        }
        data = [['blue', [
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\blue_{num}\images",
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\blue_{num}\labels"
        ], f"blue_tube_{each_num}.pkl"], ['purple', [
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\purple_{num}\images",
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\purple_{num}\labels"
        ], f"purple_tube_{each_num}.pkl"], ['white', [
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\white_{num}\images",
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\white_{num}\labels"
        ], f"white_tube_{each_num}.pkl", ], ['purple ring', [
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\purple ring_{num}\images",
            f"D:\chen\yolo_huri\paper\\RAL2022_dataset\\purple ring_{num}\labels"
        ], f"white_purple_tube_{each_num}.pkl"]]
        for (tube_type, (save_img_path, save_label_path), sample_file_dir) in data:
            save_img_path = fs.Path(save_img_path)
            save_label_path = fs.Path(save_label_path)
            if not save_img_path.exists():
                save_img_path.mkdir(parents=True)
            if not save_label_path.exists():
                save_label_path.mkdir(parents=True)
            auto_labeling(file_dir=fs.load_pickle(sample_file_dir),
                          label_id=label_info[tube_type],
                          save_img_path=save_img_path,
                          save_label_path=save_label_path, )


    def test_13():
        # real 800 fixed background
        copy_paste_annotation(cp_paths_and_classes=[[fs.load_pickle("blue_tube.pkl"), 1],
                                                    [fs.load_pickle("purple_tube.pkl"), 2],
                                                    [fs.load_pickle("white_tube.pkl"), 3],
                                                    [fs.load_pickle("white_purple_tube.pkl"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=80,
                              save_path="data_real_800_nbg_syn",
                              num_tubes=25, toggle_debug=False, random_bg=False,
                              no_rack_bg=False, no_bg=True)


    def test_14():

        copy_paste_annotation(cp_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), 1],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), 2],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), 3],
                                                    [DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=800,
                              save_path="data_real1600_syn800",
                              num_tubes=30, toggle_debug=False, random_bg=True,
                              no_rack_bg=False)


    def test_15(num_tubes=25, save_path="data_T"):
        # real 800 random background
        copy_paste_annotation(cp_paths_and_classes=[[fs.load_pickle("blue_tube.pkl"), 1],
                                                    [fs.load_pickle("purple_tube.pkl"), 2],
                                                    [fs.load_pickle("white_tube.pkl"), 3],
                                                    [fs.load_pickle("white_purple_tube.pkl"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=2,
                              save_path=f"{save_path}_{num_tubes}",
                              num_tubes=num_tubes, toggle_debug=False, random_bg=True,
                              intersection_area_thresholds=.1,
                              no_rack_bg=False)


    def test_16(intersection_area_thresholds=.1, num_tubes=30, save_path="data_t"):
        # real 800 random background
        copy_paste_annotation(cp_paths_and_classes=[[fs.load_pickle("blue_tube.pkl"), 1],
                                                    [fs.load_pickle("purple_tube.pkl"), 2],
                                                    [fs.load_pickle("white_tube.pkl"), 3],
                                                    [fs.load_pickle("white_purple_tube.pkl"), 4], ],
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=20,
                              save_path=f"{save_path}",
                              num_tubes=num_tubes, toggle_debug=True, random_bg=True,
                              intersection_area_thresholds=intersection_area_thresholds,
                              no_rack_bg=False)


    def test_17(save_path, num=400, number_tubes=35, toggle_debug=False):
        # real num random background
        each_num = int(num / 4)
        if num == 800:
            a = [[fs.load_pickle("blue_tube.pkl"), 1],
                 [fs.load_pickle("purple_tube.pkl"), 2],
                 [fs.load_pickle("white_tube.pkl"), 3],
                 [fs.load_pickle("white_purple_tube.pkl"), 4], ]
        elif num == 1600:
            a = [[DATA_ANNOT_PATH.joinpath("EXP", "BLUE"), 1],
                 [DATA_ANNOT_PATH.joinpath("EXP", "PURPLE"), 2],
                 [DATA_ANNOT_PATH.joinpath("EXP", "WHITE"), 3],
                 [DATA_ANNOT_PATH.joinpath("EXP", "WHITE PURPLE"), 4], ]
        else:
            a = [[fs.load_pickle(f"blue_tube_{each_num}.pkl"), 1],
                 [fs.load_pickle(f"purple_tube_{each_num}.pkl"), 2],
                 [fs.load_pickle(f"white_tube_{each_num}.pkl"), 3],
                 [fs.load_pickle(f"white_purple_tube_{each_num}.pkl"), 4], ]
        copy_paste_annotation(cp_paths_and_classes=a,
                              bg_paths_and_classes=[[DATA_ANNOT_PATH.joinpath("EXP", "RACK"), 0]],
                              num_imgs=700,
                              save_path=save_path,
                              num_tubes=number_tubes, toggle_debug=toggle_debug, random_bg=True,
                              intersection_area_thresholds=.15,
                              no_rack_bg=True)


    # test_11(num=200)
    # test_12(num=200)
    # test_14()
    # test_15(num_tubes=10, )
    # test_15(num_tubes=20, )
    # test_15(num_tubes=80, )
    # test_16(intersection_area_thresholds=.15, save_path="data_t_0_15")
    # test_16(intersection_area_thresholds=.40, save_path="data_t_0_40")
    # test_16(intersection_area_thresholds=.20, save_path="data_t_0_20")
    # test_16(intersection_area_thresholds=.60, save_path="data_t_0_60")
    # test_16(intersection_area_thresholds=.10, save_path="data_t_0_10")
    # test_16(intersection_area_thresholds=.40, num_tubes=30, save_path="data_t_0_80")

    # test_3()
    # test_17(num=200, save_path='data_syn_r_200', toggle_debug=False)
    # test_17(num=400, save_path='data_syn_r_400', toggle_debug=False)
    # test_17(num=800, save_path='data_syn_r_800', toggle_debug=False)
    # test_17(num=1600, number_tubes=300, save_path='data_syn_r_1600_T_300', )
    test_11(num=1600, num_tubes=30, intersection_area_thresholds=.15)

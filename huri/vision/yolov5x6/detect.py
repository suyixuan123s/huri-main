# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from huri.vision.yolov5x6.models.common import DetectMultiBackend
from huri.vision.yolov5x6.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements,
                                                colorstr,
                                                increment_path, non_max_suppression, print_args, scale_boxes,
                                                strip_optimizer, xyxy2xywh)
from huri.vision.yolov5x6.utils.plots import Annotator, colors, save_one_box
from huri.vision.yolov5x6.utils.torch_utils import select_device, time_sync
from huri.core.file_sys import workdir_vision
from huri.vision.yolov5x6.utils.dataloaders import letterbox

LOADED_MODEL = [None, None]


@torch.no_grad()
def detect(source,
           weights=workdir_vision / "yolov6" / "best2.pt",  # model.pt path(s)
           # weights=workdir_vision / "yolov6" / "weights" / "ral2022" / "best2.pt",  # model.pt path(s)
           data=workdir_vision / "yolov6" / 'tube.yaml',  # dataset.yaml path
           # data=workdir_vision / "yolov6" / 'tube2.yaml',  # dataset.yaml path
           imgsz=(1376, 1376),  # inference size (height, width)
           conf_thres=0.7,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=True,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           visualize=False,  # visualize features
           line_thickness=1,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=False,  # use FP16 half-precision inference
           dnn=False,  # use OpenCV DNN for ONNX inference
           cache_model=False,
           model_id: int = 0,
           ):
    # Load model
    device = select_device(device)
    if not cache_model:
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    else:
        if LOADED_MODEL[model_id] is None:
            model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
            LOADED_MODEL[model_id] = model
        else:
            model = LOADED_MODEL[model_id]
    a = time.time()
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    img = letterbox(source, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    dataset = [None, img, source, None, '']

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), )  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    path, im, im0s, vid_cap, s = dataset
    t1 = time_sync()
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    visualize = Path("./") if visualize else False
    pred = model(im, augment=augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    # pred = [pred]
    dt[2] += time_sync() - t3

    # Second-stage classifier (optional)
    # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

    # Process predictions
    detect_result = {}

    # pred[0] = torch.vstack((pred[0],
    #                         torch.tensor(
    #                             [[7.80570e+02, 4.82197e+02, 1.12699e+03, 8.17163e+02, 9.71801e-01, 0.00000e+00]],
    #                             device='cuda:0')))

    for i, det in enumerate(pred):  # per image
        seen += 1
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        s += '%gx%g ' % im.shape[2:]  # print string
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            # Write results
            counter = 0
            for *xyxy, conf, cls in reversed(det):
                counter += 1
                if save_txt:  # Write to file
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                if view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    _color = None
                    if 'rack' in label:
                        _color = (255, 0, 0)
                    if "purple" in label:
                        _color = (204, 50, 153)
                    if "blue" in label:
                        _color = (0, 255, 0)
                    if "white" in label:
                        _color = (0, 0, 255)
                    if "purple ring" in label:
                        _color = (0, 255, 255)
                    # if "rack" in label:
                    #     continue
                    annotator.box_label(xyxy, label, color=_color if _color is not None else colors(c, True),
                                        txt_color=(0, 0, 0))
                    # annotator.box_label(xyxy, label, color=_color if _color is not None else colors(c, True))
                    detect_result[f'{names[c]} {counter}'] = {
                        "pos": np.array([(int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))]),
                        "conf": conf.item()}
    b = time.time()
    print("Time consumption for calculation for vision sys:", b - a)
    return im0, detect_result
#
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
#     parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='show results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--visualize', action='store_true', help='visualize features')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
#     parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
#     parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(FILE.stem, opt)
#     return opt
#
#
# def main(opt):
#     check_requirements(exclude=('tensorboard', 'thop'))
#     run(**vars(opt))
#
#
# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)

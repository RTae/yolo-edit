import cv2
import torch
import numpy as np
from numpy import random

from utils.plots import plot_one_box
from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh

def detect(img_origin, path):

    weights = "./weights/best.pt"
    save = "../dataset/wood-plank-1/result"
    imgsz = 416
    conf_thres = 0.45
    iou_thres = 0.25

    # Initialize
    device = select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # get images
    img = letterbox(img_origin, imgsz, stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Prepare image
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = img_origin
        img_name = path.split("/")[-1]

        save_path = f"{save}/result_{img_name}"  # img.jpg
        
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                print(label)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
        
        cv2.imwrite(save_path, im0)

path = "../dataset/wood-plank-1/test/test11.png"
img = cv2.imread(path)  # BGR

detect(img, path)

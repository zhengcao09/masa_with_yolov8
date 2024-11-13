import gc
import resource
import argparse
import cv2
from tqdm import tqdm

import sys
import os
import time
import numpy as np

import supervision as sv
from ultralytics import YOLO
import torch
from torch.nn import functional as F
import torch
from torch.multiprocessing import Pool, set_start_method

import mmcv
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.apis import init_detector
from mmdet.registry import VISUALIZERS
from mmcv.ops.nms import batched_nms
from mmdet.structures import DetDataSample

import warnings
warnings.filterwarnings('ignore')

import masa
from masa.apis import init_masa
from masa.apis import inference_masa, init_masa, inference_detector, build_test_pipeline
from masa.models.sam import SamPredictor, sam_model_registry

import warnings
warnings.filterwarnings('ignore')


# VIDEO_ROOT = "your/video/path/here"
# VIDEO_SINK = "your/video/sink/path/here"
VIDEO_ROOT = "/home/zc494/masa/demo/fourth.mp4"
VIDEO_SINK = "/home/zc494/masa/fourth_output.mp4"

MASA_CONFIG = "/home/zc494/masa/configs/masa-one/masa_r50_plug_and_play.py"
MASA_CHECKPOINT = "/home/zc494/masa/saved_models/masa_models/masa_r50.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
# model = YOLO('yolov8x-seg.pt').to(device)
model = YOLO("/home/zc494/masa/saved_models/pretrain_weights/model_weights.pt").to(device)

mask_annotator = sv.MaskAnnotator()
#box_annotator = sv.BoundingBoxAnnotator(thickness=1)
#track_annotator = sv.TraceAnnotator()
#label_annotator = sv.LabelAnnotator()

box_annotator = sv.BoundingBoxAnnotator(thickness=1, color_lookup = sv.ColorLookup.TRACK)
track_annotator = sv.TraceAnnotator(color_lookup = sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(color_lookup = sv.ColorLookup.TRACK, text_scale = 0.3, text_thickness = 1, text_padding = 2)

video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_ROOT)
frame_generator = sv.get_video_frames_generator(source_path=VIDEO_ROOT)

# Init the MASA tracker model
masa_model = init_masa(MASA_CONFIG, MASA_CHECKPOINT, device='cuda')
masa_test_pipeline = build_test_pipeline(masa_model.cfg)

frame_idx = 0

with sv.VideoSink(VIDEO_SINK, video_info=video_info) as sink:

    for frame in tqdm(frame_generator, total=video_info.total_frames):

        # Do the detections
        result = model(frame, imgsz = 1280,verbose=False)[0]

        # reformat the ultralytics results into detections for MASA
        det_bboxes = result.boxes.cpu().xyxy
        det_scores = result.boxes.cpu().conf
        det_labels = result.boxes.cpu().cls

        det_bboxes = torch.cat([det_bboxes,
                                det_scores.unsqueeze(1)],
                                    dim=1)

        # Do the tracking!
        track_result, fps = inference_masa(masa_model, frame, frame_id=frame_idx,
                              video_len=video_info.total_frames,
                              test_pipeline=masa_test_pipeline,
                              det_bboxes=det_bboxes,
                              det_labels=det_labels,
                              fp16=True,
                              show_fps=True)

        # get the most recent frame of tracked objects
        track_result = track_result.video_data_samples[-1]

        # print(f"track results: {track_result}")

        # create new MMDet object and put the data from the TrackDataSample object into the new DetDataSample so Supervision can process it.
        tracks = DetDataSample()
        pred_instances = track_result.pred_track_instances
        tracks.pred_instances = pred_instances

        # add the tracks to supervision
        detections = sv.Detections.from_mmdetection(tracks)
        # add the tracker_ids separately (supervision doesn't support TrackDataSample objects yet)
        detections.tracker_id = tracks.pred_instances.instances_id.cpu().numpy()

        # pred_instances are detections
        # pred_track_instances are tracked tracks

        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id, _
            in detections
        ]

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(),
            detections=detections)
        annotated_frame = track_annotator.annotate(
            scene=annotated_frame.copy(),
            detections=detections)
        annotated_frame = mask_annotator.annotate(
            scene=annotated_frame.copy(),
            detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame.copy(),
            labels=labels,
            detections=detections
        )


        sink.write_frame(frame=annotated_frame)
        frame_idx += 1


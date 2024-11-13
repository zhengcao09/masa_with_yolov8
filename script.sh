#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


#python /home/zc494/masa/demo/video_demo_with_text.py demo/crop.mp4 --out /home/zc494/masa/demo_outputs/short_output.mp4 --masa_config configs/masa-gdino/masa_gdino_swinb_inference.py --masa_checkpoint saved_models/masa_models/gdino_masa.pth --texts "fish" --score-thr 0.1 --unified --show_fps

#python /home/zc494/masa/demo/video_demo_with_text.py demo/trimmed.mp4 --out /home/zc494/masa/demo_outputs/trimmed_output.mp4 --det_config projects/mmdet_configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py --det_checkpoint saved_models/pretrain_weights/model_weights.pt --masa_config configs/masa-one/masa_r50_plug_and_play.py --masa_checkpoint saved_models/masa_models/masa_r50.pth --score-thr 0.3 --show_fps


#python /home/zc494/masa/demo/video_demo_with_text.py demo/second.mp4 --out /home/zc494/masa/demo_outputs/second_outputs.mp4 --det_config projects/mmdet_configs/yolox/yolox_x_8xb8-300e_coco.py --det_checkpoint saved_models/pretrain_weights/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth --masa_config configs/masa-one/masa_r50_plug_and_play.py --masa_checkpoint saved_models/masa_models/masa_r50.pth --score-thr 0.3 --show_fps

python /home/zc494/masa/run.py




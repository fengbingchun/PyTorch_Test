#! /bin/bash

# Blog: https://blog.csdn.net/fengbingchun/article/details/127038191

python ../../src/mmdeploy/tools/deploy.py \
    ../../src/mmdeploy/configs/mmdet/detection/detection_onnxruntime_dynamic.py \
    ../../src/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth \
    ../../data/image/1.jpg \
    --work-dir ../../data/model \
    --device cpu \
    --show

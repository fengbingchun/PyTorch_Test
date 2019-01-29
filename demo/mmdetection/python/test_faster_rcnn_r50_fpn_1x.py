import os
import subprocess
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result	
from mmdet.core import get_classes

# https://blog.csdn.net/fengbingchun/article/details/86693037

def show_and_save_result(img, result, out_dir, dataset="coco", score_thr=0.3):
	class_names = get_classes(dataset)
	labels = [
		np.full(bbox.shape[0], i, dtype=np.int32)
		for i, bbox in enumerate(result)
	]
	labels = np.concatenate(labels)
	bboxes = np.vstack(result)
	
	index = img.rfind("/")
	mmcv.imshow_det_bboxes(img, bboxes, labels, class_names, score_thr, show=True, out_file=out_dir+img[index+1:])

def main():
	model_path = "../../../data/model/"
	model_name = "faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth"
	config_name = "../../../src/mmdetection/configs/faster_rcnn_r50_fpn_1x.py"

	if os.path.isfile(model_path + model_name) == False:
		print("model file does not exist, now download ...")
		url = "https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth"
		subprocess.run(["wget", "-P", model_path, url])	

	cfg = mmcv.Config.fromfile(config_name)
	cfg.model.pretrained = None

	model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
	_ = load_checkpoint(model, model_path + model_name)

	image_path = "../../../data/image/"
	imgs = ["1.jpg", "2.jpg", "3.jpg"]
	images = list()
	for i, value in enumerate(imgs):
		images.append(image_path + value)
	
	out_dir = "../../../data/result/"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for i, result in enumerate(inference_detector(model, images, cfg)):
		print(i, images[i])
		show_and_save_result(images[i], result, out_dir)
	
	print("test finish")

if __name__ == "__main__":
	main()

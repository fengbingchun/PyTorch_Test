import os
import subprocess
import numpy as np
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.core import get_classes

# Blog: 
#	https://blog.csdn.net/fengbingchun/article/details/86693037
#	https://blog.csdn.net/fengbingchun/article/details/126199218

def show_and_save_result(img, result, out_dir, dataset="coco", score_thr=0.6):
	print("test image:", img)
	class_names = get_classes(dataset)
	labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(result)]
	labels = np.concatenate(labels)
	bboxes = np.vstack(result)
	
	index = img.rfind("/")
	mmcv.imshow_det_bboxes(img, bboxes, labels, class_names, score_thr, show=True, out_file=out_dir+img[index+1:])

def main():
	# specify the path to model config and checkpoint(model) file
	model_path = "../../data/model/"
	model_name = "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

	if os.path.isfile(model_path + model_name) == False:
		print("model file does not exist, now download ...")
		url = "http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
		subprocess.run(["wget", "-P", model_path, url])	

	# build the model from a config file and a checkpoint file
	config_file = "../../src/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
	model = init_detector(config_file, model_path+model_name, device='cuda:0')

	image_path = "../../data/image/"
	images_name = ["1.jpg", "2.jpg", "3.jpg"]
	images_name[:] = [image_path+x for x in images_name]
	
	out_dir = "../../data/result/"
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	for i, result in enumerate(inference_detector(model, images_name), start=0):
		show_and_save_result(images_name[i], result, out_dir)
	
	print("test finish")

if __name__ == "__main__":
	main()

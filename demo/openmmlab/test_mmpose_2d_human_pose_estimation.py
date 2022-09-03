import mmpose
import mmdet
import torch
from util import download_checkpoint
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result
import cv2

# Blog: https://blog.csdn.net/fengbingchun/article/details/126677075

def mmdet_human_detection(device, image, threshold=0.9):
	path = "../../data/model/"
	checkpoint = "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
	url = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmpose/demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py"
	model = init_detector(config, path+checkpoint, device)

	mmdet_results = inference_detector(model, image)
	# print(mmdet_results)

	human_results = process_mmdet_results(mmdet_results)
	# print(human_results)

	filter_results = []
	mat = cv2.imread(image)
	for result in human_results:
		print("result:", result)
		if result['bbox'][4] > threshold:
			filter_results.append(result)
			cv2.rectangle(mat, (int(result['bbox'][0]), int(result['bbox'][1])), (int(result['bbox'][2]), int(result['bbox'][3])), (255, 0, 0), 1)

	cv2.imwrite("../../data/result_mmpose_2d_human_detection.png", mat)
	cv2.imshow("show", mat)
	cv2.waitKey(0)

	return filter_results

def mmpose_human_pose_estimation(device, image, human_bbox_results):
	path = "../../data/model/"
	checkpoint = "hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
	url = "https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py"
	model = init_pose_model(config, path+checkpoint, device)

	pose_results, returned_outputs = inference_top_down_pose_model(model, image, human_bbox_results, bbox_thr=None, format='xyxy')
	print(pose_results)

	vis_pose_result(model, image, pose_results, radius=1, thickness=1, show=True, out_file="../../data/result_mmpose_2d_human_pose_estimation.png")

def main():
	print(f"mmpose version: {mmpose.__version__}; mmdet version: {mmdet.__version__}")
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print("device:", device)

	image_path = "../../data/image/"
	image_name = "human.png"

	threshold = 0.7
	human_results = mmdet_human_detection(device, image_path+image_name, threshold)

	mmpose_human_pose_estimation(device, image_path+image_name, human_results)

	print("test finish")

if __name__ == "__main__":
	main()

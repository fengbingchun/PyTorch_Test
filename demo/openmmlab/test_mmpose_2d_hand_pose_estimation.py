import mmpose
import mmdet
import torch
from util import download_checkpoint
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results, vis_pose_result
import cv2

# Blog: https://blog.csdn.net/fengbingchun/article/details/126676729

def mmdet_hand_detection(device, image):
	path = "../../data/model/"
	checkpoint = "cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth"
	url = "https://download.openmmlab.com/mmpose/mmdet_pretrained/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py"
	model = init_detector(config, path+checkpoint, device)

	mmdet_results = inference_detector(model, image)
	# print(mmdet_results)

	hand_results = process_mmdet_results(mmdet_results)

	mat = cv2.imread(image)
	for result in hand_results:
		print("result:", result)
		cv2.rectangle(mat, (int(result['bbox'][0]), int(result['bbox'][1])), (int(result['bbox'][2]), int(result['bbox'][3])), (255, 0, 0), 1)

	cv2.imwrite("../../data/result_mmpose_2d_hand_detection.png", mat)
	cv2.imshow("show", mat)
	cv2.waitKey(0)

	return hand_results

def mmpose_hand_pose_estimation(device, image, hand_bbox_results):
	path = "../../data/model/"
	checkpoint = "res50_onehand10k_256x256-e67998f6_20200813.pth"
	url = "https://download.openmmlab.com/mmpose/top_down/resnet/res50_onehand10k_256x256-e67998f6_20200813.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmpose/configs/hand/2d_kpt_sview_rgb_img/topdown_heatmap/onehand10k/res50_onehand10k_256x256.py"
	model = init_pose_model(config, path+checkpoint, device)

	pose_results, returned_outputs = inference_top_down_pose_model(model, image, hand_bbox_results, bbox_thr=None, format='xyxy')
	print(pose_results)

	vis_pose_result(model, image, pose_results, radius=1, thickness=1, show=True, out_file="../../data/result_mmpose_2d_hand_pose_estimation.png")

def main():
	print(f"mmpose version: {mmpose.__version__}; mmdet version: {mmdet.__version__}")
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print("device:", device)

	image_path = "../../data/image/"
	image_name = "hand.png"

	hand_results = mmdet_hand_detection(device, image_path+image_name)

	mmpose_hand_pose_estimation(device, image_path+image_name, hand_results)

	print("test finish")

if __name__ == "__main__":
	main()

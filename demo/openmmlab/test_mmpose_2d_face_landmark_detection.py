import mmpose
import dlib
import face_recognition
import torch
from util import download_checkpoint
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
import cv2

# Blog: https://blog.csdn.net/fengbingchun/article/details/126676309

def get_face_location(image):
	data = face_recognition.load_image_file(image)
	face_det_results = face_recognition.face_locations(data) # a list of tuples of found face locations in css (top, right, bottom, left) order
	print("face detect results:", face_det_results)

	face_bbox_results = []

	mat = cv2.imread(image)
	for rect in face_det_results:
		cv2.rectangle(mat, (rect[3], rect[0]), (rect[1], rect[2]), (0, 255, 0), 1)
		person = {}
		person["bbox"] = [rect[3], rect[0], rect[1], rect[2]]
		face_bbox_results.append(person)

	cv2.imwrite("../../data/result_mmpose_face_location.png", mat)
	cv2.imshow("show", mat)
	cv2.waitKey(0)

	print("face bbox results:", face_bbox_results)
	return face_bbox_results

def mmpose_face_landmark(device, image, face_bbox_results):
	path = "../../data/model/"
	checkpoint = "hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth"
	url = "https://download.openmmlab.com/mmpose/face/hrnetv2/hrnetv2_w18_aflw_256x256-f2bbc62b_20210125.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmpose/configs/face/2d_kpt_sview_rgb_img/topdown_heatmap/aflw/hrnetv2_w18_aflw_256x256.py"
	model = init_pose_model(config, path+checkpoint, device)

	pose_results, returned_outputs = inference_top_down_pose_model(model, image, face_bbox_results, bbox_thr=None, format='xyxy')
	# print(pose_results)

	# vis_pose_result(model, image, pose_results, radius=1, thickness=1, show=True, out_file="../../data/result_mmpose_2d_face_landmark.png")

	mat = cv2.imread(image)
	for result in pose_results:
		# print(f"bbox: {result['bbox']}, keypoints: {result['keypoints']}")
		cv2.rectangle(mat, (result['bbox'][0], result['bbox'][1]), (result['bbox'][2], result['bbox'][3]), (255, 0, 0), 1)

		for keypoints in result['keypoints']:
			cv2.circle(mat, (int(keypoints[0]), int(keypoints[1])), 1, (0, 0, 255), 1)

	cv2.imwrite("../../data/result_mmpose_face_landmark.png", mat)
	cv2.imshow("show", mat)
	cv2.waitKey(0)

def main():
	print(f"mmpose version: {mmpose.__version__}; dlib version: {dlib.__version__}; face recognition version: {face_recognition.__version__}")
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print("device:", device)

	image_path = "../../data/image/"
	image_name = "2.jpg"

	face_bbox_results = get_face_location(image_path+image_name)

	mmpose_face_landmark(device, image_path+image_name, face_bbox_results)

	print("test finish")

if __name__ == "__main__":
	main()

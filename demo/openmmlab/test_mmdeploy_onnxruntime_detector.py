import cv2
from mmdeploy_python import Detector

# Note: there is still a problem with this test
# [error] [model.cpp:45] no ModelImpl can read model ../../data/model/end2end.onnx
# [error] [model.cpp:15] load model failed. Its file path is '../../data/model/end2end.onnx'
# [error] [model.cpp:20] failed to create model: not supported (2) @ /data1/prebuild/mmdeploy/csrc/mmdeploy/core/model.cpp:46

def main():
	model = "../../data/model/end2end.onnx"
	detector = Detector(model_path=model, device_name="cpu")

	image_path = "../../data/image/"
	images_name = ["1.jpg", "2.jpg", "3.jpg"]
	img = cv2.imread(image_path+images_name[0])
	print("img shape:", img.shape)
	bboxes, labels, masks = detector(img)

	indices = [i for i in range(len(bboxes))]
	for index, bbox, label_id in zip(indices, bboxes, labels):
		[left, top, right, bottom], score = bbox[0:4].astype(int), bbox[4]
		if score < 0.3:
			continue

		cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

		if masks[index].size:
			mask = masks[index]
			blue, green, red = cv2.split(img)
			mask_img = blue[top:top + mask.shape[0], left:left + mask.shape[1]]
			cv2.bitwise_or(mask, mask_img, mask_img)
			img = cv2.merge([blue, green, red])

	cv2.imwrite('../../data/result_mmdeploy_onnxruntime_detection.png', img)
	print("test finish")

if __name__ == "__main__":
	main()

from mmedit.apis import init_model, matting_inference
import torch
from util import download_checkpoint
import cv2
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/126332219

def matting_indexnet(device, image, trimap):
	path = "../../data/model/"
	checkpoint = "indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth"
	url = "https://download.openmmlab.com/mmediting/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k_SAD-45.6_20200618_173817-26dd258d.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmediting/configs/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k.py"
	model = init_model(config, path+checkpoint, device)

	result = matting_inference(model, image, trimap) * 255

	print(f"result shape: {result.shape}; max value: {np.max(result)}") # result shape: (450, 617); max value: 255.0
	_, result = cv2.threshold(result, 254, 255, cv2.THRESH_BINARY)
	result = result.astype(np.uint8)
	cv2.imwrite("../../data/result_matting_indexnet.jpg", result)
	cv2.imshow("show_result", result)
	cv2.waitKey(0)

	mat1 = cv2.imread(image)
	mat3 = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
	mat3 = cv2.bitwise_and(mat1, mat3, result)
	# cv2.imshow("show_mat3", mat3)
	# cv2.waitKey(0)

	_, mat4 = cv2.threshold(result, 254, 255, cv2.THRESH_BINARY_INV)
	mat4 = cv2.cvtColor(mat4, cv2.COLOR_GRAY2BGR)
	mat4 = mat4.astype(np.uint8)

	mat2 = np.zeros(mat1.shape, dtype=np.uint8)
	mat2[:,:] = (0, 255, 0)
	mat2 = cv2.bitwise_and(mat2, mat4)

	mat2 = mat3 + mat2
	cv2.imwrite("../../data/result_matting_indexnet_bgr.jpg", mat2)
	cv2.imshow("show_mat2", mat2)
	cv2.waitKey(0)

def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	#print("device:", device)

	image_path = "../../data/image/"
	image_name = "5.jpg"
	trimap_name = "5_trimap.png"

	matting_indexnet(device, image_path+image_name, image_path+trimap_name)

	print("test finish")

if __name__ == "__main__":
	main()

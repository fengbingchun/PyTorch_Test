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

	config = "../../src/mmediting//configs/mattors/indexnet/indexnet_mobv2_1x16_78k_comp1k.py"
	model = init_model(config, path+checkpoint, device)

	result = matting_inference(model, image, trimap) * 255
	print(f"result shape: {result.shape}; max value: {np.max(result)}") # result shape: (552, 800); max value: 255.0
	cv2.imwrite("../../data/result_matting_indexnet.jpg", result)
	cv2.imshow("show", result)
	cv2.waitKey(0)

def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	#print("device:", device)

	image_path = "../../src/mmediting/tests/data/"
	image_name = "merged/GT05.jpg"
	trimap_name = "trimap/GT05.png"

	matting_indexnet(device, image_path+image_name, image_path+trimap_name)

	print("test finish")

if __name__ == "__main__":
	main()

from mmedit.apis import init_model, generation_inference
from mmedit.core import tensor2img
import torch
from util import download_checkpoint
import cv2
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/126442129

def generation_pix2pix(device, image):
	path = "../../data/model/"
	checkpoint = "pix2pix_vanilla_unet_bn_1x1_80k_facades_20200524-6206de67.pth"
	url = "https://download.openmmlab.com/mmediting/synthesizers/pix2pix/pix2pix_facades/pix2pix_vanilla_unet_bn_1x1_80k_facades_20200524-6206de67.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmediting/configs/synthesizers/pix2pix/pix2pix_vanilla_unet_bn_1x1_80k_facades.py"
	model = init_model(config, path+checkpoint, device)

	result = generation_inference(model, image)
	print(f"result shape: {result.shape}; max value: {np.max(result)}") # result shape: (256, 768, 3); max value: 255

	cv2.imwrite("../../data/result_generation_pix2pix.jpg", result)
	cv2.imshow("show", result)
	cv2.waitKey(0)

def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	#print("device:", device)

	image_path = "../../data/image/"
	image_name = "11.png"

	generation_pix2pix(device, image_path+image_name)

	print("test finish")

if __name__ == "__main__":
	main()

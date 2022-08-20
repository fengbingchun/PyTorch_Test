from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img
import torch
from util import download_checkpoint
import cv2
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/126441016

def crop_save_image(srcimage, dstimage, name):
	crop_height, crop_width = int(srcimage.shape[1]/2), int(srcimage.shape[0]/2)
	print(f"crop height: {crop_height}; crop width: {crop_width}, data type: {type(crop_height)}")

	mat = cv2.resize(srcimage, (dstimage.shape[1], dstimage.shape[0]))
	srccrop = mat[0:crop_height, 0:crop_width]
	dstcrop = dstimage[0:crop_height, 0:crop_width]

	path = "../../data/"
	cv2.imwrite(path+"src_"+name, srccrop)
	cv2.imwrite(path+"result_"+name, dstcrop)

	cv2.imshow("show_src", srccrop)
	cv2.waitKey(0)
	cv2.imshow("show_result", dstcrop)
	cv2.waitKey(0)

def restoration_srcnn(device, image):
	path = "../../data/model/"
	checkpoint = "srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth"
	url = "https://download.openmmlab.com/mmediting/restorers/srcnn/srcnn_x4k915_1x16_1000k_div2k_20200608-4186f232.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmediting/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py"
	model = init_model(config, path+checkpoint, device)

	result = restoration_inference(model, image)
	print(f"result shape: {result.shape}; max value: {torch.max(result)}") # result shape: torch.Size([1, 3, 1920, 2000]); max value: 1.136252999305725

	result = tensor2img(result)
	#print(f"height: {result.shape[0]}, width: {result.shape[1]}, channel: {result.shape[2]}")
	srcimage = cv2.imread(image)
	crop_save_image(srcimage, result, "restoration_srcnn.png")

def restoration_liif(device, image):
	path = "../../data/model/"
	checkpoint = "liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth"
	url = "https://download.openmmlab.com/mmediting/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k_20210715-ab7ce3fc.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmediting/configs/restorers/liif/liif_edsr_norm_c64b16_g1_1000k_div2k.py"
	model = init_model(config, path+checkpoint, device)

	result = restoration_inference(model, image)
	print(f"result shape: {result.shape}; max value: {torch.max(result)}") # result shape: torch.Size([1, 3, 1920, 2000]); max value: 1.0

	result = tensor2img(result)
	srcimage = cv2.imread(image)
	crop_save_image(srcimage, result, "restoration_liif.png")

def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	#print("device:", device)

	image_path = "../../src/mmediting/tests/data/gt/"
	image_name = "baboon.png"

	#restoration_srcnn(device, image_path+image_name)
	restoration_liif(device, image_path+image_name)

	print("test finish")

if __name__ == "__main__":
	main()

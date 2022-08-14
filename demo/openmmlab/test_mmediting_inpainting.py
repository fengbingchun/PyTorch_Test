from mmedit.apis import init_model, inpainting_inference
from mmedit.core import tensor2img
import torch
from util import download_checkpoint
import cv2
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/126331541

def inpainting_global_local(device, image, mask):
	path = "../../data/model/"
	checkpoint = "gl_256x256_8x12_celeba_20200619-5af0493f.pth"
	url = "https://download.openmmlab.com/mmediting/inpainting/global_local/gl_256x256_8x12_celeba_20200619-5af0493f.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmediting/configs/inpainting/global_local/gl_256x256_8x12_celeba.py"
	model = init_model(config, path+checkpoint, device)

	result = inpainting_inference(model, image, mask)
	print(f"result shape: {result.shape}; max value: {torch.max(result)}") # result shape: torch.Size([1, 3, 256, 256]); max value: 1.0
	result = tensor2img(result, min_max=(-1, 1))[..., ::-1]
	print("shape:", np.shape(result)) # shape: (256, 256, 3)
	cv2.imwrite("../../data/result_inpainting_global_local_celeba.jpg", result)
	cv2.imshow("show", result)
	cv2.waitKey(0)

def inpainting_aot_gan(device, image, mask):
	path = "../../data/model/"
	checkpoint = "AOT-GAN_512x512_4x12_places_20220509-6641441b.pth"
	url = "https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmediting/inpainting/aot_gan/AOT-GAN_512x512_4x12_places_20220509-6641441b.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmediting/configs/inpainting/AOT-GAN/AOT-GAN_512x512_4x12_places.py"
	model = init_model(config, path+checkpoint, device)

	result = inpainting_inference(model, image, mask)
	print(f"result shape: {result.shape}; max value: {torch.max(result)}") # result shape: torch.Size([1, 3, 256, 256]); max value: 1.0
	result = tensor2img(result, min_max=(-1, 1))[..., ::-1]
	print("shape:", np.shape(result))
	cv2.imwrite("../../data/result_inpainting_aot_gan.jpg", result)
	cv2.imshow("show", result)
	cv2.waitKey(0)

def main():
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	#print("device:", device)

	image_path = "../../src/mmediting/tests/data/image/"
	image_name = "celeba_test.png"
	image_mask_name = "bbox_mask.png"

	inpainting_global_local(device, image_path+image_name, image_path+image_mask_name)
	#inpainting_aot_gan(device, image_path+image_name, image_path+image_mask_name)

	print("test finish")

if __name__ == "__main__":
	main()

import mmseg
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
import torch
from util import download_checkpoint
import cv2
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/126569920

def segmentor_pspnet(device, image):
	path = "../../data/model/"
	checkpoint = "pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
	url = "https://download.openmmlab.com/mmsegmentation/v0.5/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
	model = init_segmentor(config, path+checkpoint, device)

	result = inference_segmentor(model, image)
	print(f"result shape: {result[0].shape}, max value: {np.max(result[0])}, data type: {result[0].dtype}") # result shape: (667, 1400), max value: 18, data type: int64

	show_result_pyplot(model, image, result)

	dst = result[0].astype(np.uint8) * int(255/np.max(result[0]))
	print(f"type: {dst.dtype}, max value: {np.max(dst)}")
	cv2.imwrite("../../data/result_segmentor_pspnet.png", dst)
	cv2.imshow("show", dst)
	cv2.waitKey(0)

def main():
	print("mmseg version:", mmseg.__version__)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print("device:", device)

	image_path = "../../data/image/"
	image_name = "12.png"

	segmentor_pspnet(device, image_path+image_name)

	print("test finish")

if __name__ == "__main__":
	main()

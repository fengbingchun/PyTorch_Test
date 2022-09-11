import mmocr
from mmocr.utils.ocr import MMOCR
import tesserocr
import torch
from util import download_checkpoint
import cv2

# Blog: https://blog.csdn.net/fengbingchun/article/details/126805622

def mmocr_text_detection_recognition(image, device):
	# detection
	path = "../../data/model/"
	checkpoint = "textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth"
	url = "https://download.openmmlab.com/mmocr/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500-27f65b64.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmocr/configs/textdet/textsnake/textsnake_r50_fpn_unet_1200e_ctw1500.py"
	ocr = MMOCR(det="TextSnake", det_config=config, det_ckpt=path+checkpoint, recog=None, device=device)

	results = ocr.readtext(image, output="../../data/result_mmocr_text_detection.png", export="../../data/", export_format="json")

	# detection + recognition
	checkpoint2 = "seg_r31_1by16_fpnocr_academic-72235b11.pth"
	url = "https://download.openmmlab.com/mmocr/textrecog/seg/seg_r31_1by16_fpnocr_academic-72235b11.pth"
	download_checkpoint(path, checkpoint2, url)

	config2 = "../../src/mmocr/configs/textrecog/seg/seg_r31_1by16_fpnocr_toy_dataset.py"
	ocr2 = MMOCR(det="TextSnake", det_config=config, det_ckpt=path+checkpoint, recog="SEG", recog_config=config2, recog_ckpt=path+checkpoint2, device=device)

	results2 = ocr2.readtext(image, output="../../data/result_mmocr_text_recognition.png")
	print("recognition result:", results2)

def main():
	print(f"mmocr version: {mmocr.__version__}, tesserocr version: {tesserocr.__version__}")
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print("device:", device)

	image_path = "../../data/image/"
	image_name = "ocr_english.png"

	mmocr_text_detection_recognition(image_path+image_name, device)

	print("test finish")

if __name__ == "__main__":
	main()

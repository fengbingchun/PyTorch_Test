import mmgen
import torch
from util import download_checkpoint
import cv2
from torchvision import utils
from mmgen.apis import init_model, sample_conditional_model, sample_unconditional_model, sample_img2img_model

# Blog: https://blog.csdn.net/fengbingchun/article/details/126805945

def mmgeneration_conditional(device):
	path = "../../data/model/"
	checkpoint = "sagan_128_woReLUinplace_noaug_bigGAN_imagenet1k_b32x8_Glr1e-4_Dlr-4e-4_ndisc1_20210818_210232-3f5686af.pth"
	url = "https://download.openmmlab.com/mmgen/sagan/" + checkpoint
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmgeneration/configs/sagan/sagan_128_woReLUinplace_noaug_bigGAN_Glr-1e-4_Dlr-4e-4_ndisc1_imagenet1k_b32x8.py"
	model = init_model(config, path+checkpoint, device)

	results = sample_conditional_model(model, num_samples=2, num_batches=1, label=[8])
	print("results shape:", results.shape)
	results = (results[:, [2, 1, 0]] + 1.) / 2.

	utils.save_image(results, "../../data/result_mmgeneration_conditional.png")

def mmgeneration_unconditional(device):
	path = "../../data/model/"
	checkpoint = "stylegan2_c2_ffhq_1024_b4x8_20210407_150045-618c9024.pth"
	url = "https://download.openmmlab.com/mmgen/stylegan2/" + checkpoint
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmgeneration/configs/styleganv2/stylegan2_c2_ffhq_1024_b4x8.py"
	model = init_model(config, path+checkpoint, device)

	results = sample_unconditional_model(model, num_samples=2, num_batches=1)
	print("results shape:", results.shape)
	results = (results[:, [2, 1, 0]] + 1.) / 2.

	utils.save_image(results, "../../data/result_mmgeneration_unconditional.png")

def mmgeneration_image2image_translation(image, device):
	path = "../../data/model/"
	checkpoint = "pix2pix_vanilla_unet_bn_wo_jitter_flip_1x4_186840_edges2shoes_convert-bgr_20210902_170902-0c828552.pth"
	url = "https://download.openmmlab.com/mmgen/pix2pix/refactor/" + checkpoint
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmgeneration/configs/pix2pix/pix2pix_vanilla_unet_bn_wo_jitter_flip_edges2shoes_b1x4_190k.py"
	model = init_model(config, path+checkpoint, device)

	results = sample_img2img_model(model, image)
	print("results shape:", results.shape)
	results = (results[:, [2, 1, 0]] + 1.) / 2.

	utils.save_image(results, "../../data/result_mmgeneration_image2image_translation.png")

def main():
	print(f"mmgen version: {mmgen.__version__}")
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print("device:", device)

	image_path = "../../data/image/"
	image_name = "11.png"

	mmgeneration_conditional(device)
	mmgeneration_unconditional(device)
	mmgeneration_image2image_translation(image_path+image_name, device)

	print("test finish")

if __name__ == "__main__":
	main()

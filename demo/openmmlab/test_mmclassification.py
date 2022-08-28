import mmcls
from mmcls.apis import inference_model, init_model, show_result_pyplot
import torch
from util import download_checkpoint
import mmcv

# Blog: https://blog.csdn.net/fengbingchun/article/details/126570201

def classification_resnet(device, image):
	path = "../../data/model/"
	checkpoint = "resnet152_b16x8_cifar10_20210528-3e8e9178.pth"
	url = "https://download.openmmlab.com/mmclassification/v0/resnet/resnet152_b16x8_cifar10_20210528-3e8e9178.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmclassification/configs/resnet/resnet152_b16x8_cifar10.py"
	model = init_model(config, path+checkpoint, device)

	result = inference_model(model, image)
	print(mmcv.dump(result, file_format='json', indent=4))
	# show_result_pyplot(model, image, result)

def classification_vgg(device, image):
	path = "../../data/model/"
	checkpoint = "vgg19_batch256_imagenet_20210208-e6920e4a.pth"
	url = "https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.pth"
	download_checkpoint(path, checkpoint, url)

	config = "../../src/mmclassification/configs/vgg/vgg19_8xb32_in1k.py"
	model = init_model(config, path+checkpoint, device)

	result = inference_model(model, image)
	print(mmcv.dump(result, file_format='json', indent=4))
	# show_result_pyplot(model, image, result)

def main():
	print("mmcls version:", mmcls.__version__)
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	print("device:", device)

	image_path = "../../data/image/"
	image_name = "6.jpg"

	# classification_resnet(device, image_path+image_name)
	classification_vgg(device, image_path+image_name)

	print("test finish")

if __name__ == "__main__":
	main()

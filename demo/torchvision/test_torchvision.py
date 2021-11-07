from torchvision import datasets
from torchvision import io
from torchvision import models
from torchvision import ops
from torchvision import transforms

import torch

# Blog: https://blog.csdn.net/fengbingchun/article/details/121191194

# 下载MNIST数据集: torchvision.datasets包
test = datasets.MNIST("../../data", train=False, download=True)
train = datasets.MNIST("../../data", train=True, download=False)
print(f"raw_folder: test: {test.raw_folder}, train: {train.raw_folder}")
print(f"processed_folder: test: {test.processed_folder}, train: {train.processed_folder}")
print(f"extra_repr:\ntest: {test.extra_repr}\ntrain: {train.extra_repr}")
print(f"class to index: {test.class_to_idx}")

# 读写图像: torchvision.io包
tensor = io.read_image("../../data/image/1.jpg")
print("tensor shape:", tensor.shape)
io.write_png(tensor, "../../data/image/result.png")

tensor = io.read_image("../../data/image/lena.png")
print("tensor shape:", tensor.shape)
io.write_jpeg(tensor, "../../data/image/result.jpg")

# 下载pre-trained AlexNet模型: torchvision.models包
net = models.alexnet(pretrained=True)

# 计算机视觉操作: torchvision.ops包
boxes = torch.tensor([[1, 1, 101, 101], [3, 5, 13, 15], [2, 4, 22, 44]])
area = ops.box_area(boxes)
print(f"area: {area}")

index = ops.remove_small_boxes(boxes, min_size=20)
print(f"index: {index}")

# 图像变换: torchvision.transforms包
resize = transforms.Resize(size=[256, 128])
img = resize.forward(tensor)
io.write_jpeg(img, "../../data/image/resize.jpg")

grayscale = transforms.Grayscale()
img2 = grayscale.forward(img)
io.write_jpeg(img2, "../../data/image/gray.jpg")

affine = transforms.RandomAffine(degrees=35)
img3 = affine.forward(tensor)
io.write_jpeg(img3, "../../data/image/affine.jpg")

crop = transforms.CenterCrop(size=[128, 128])
img4 = crop.forward(tensor)
io.write_jpeg(img4, "../../data/image/crop.jpg")

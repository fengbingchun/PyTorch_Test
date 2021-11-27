import torch
from torchvision import models
from torchvision import transforms
import cv2
from PIL import Image
import math
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/121579039

#print(dir(models))

images_path = "../../data/image/"
images_name = ["5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg", "10.jpg"]
images_data = [] # opencv
tensor_data = [] # pytorch tensor

def images_stitch(images, cols=3, name="result.jpg"): # 图像简单拼接
    '''images: list, opencv image data; cols: number of images per line; name: save image result name'''
    width_total = 660
    width, height = width_total // cols, width_total // cols
    number = len(images)
    height_total = height * math.ceil(number / cols)

    mat1 = np.zeros((height_total, width_total, 3), dtype="uint8") # in Python images are represented as NumPy arrays

    for idx in range(number):
        height_, width_, _ = images[idx].shape
        if height_ != width_:
            if height_ > width_:
                width_ = math.floor(width_ / height_ * width)
                height_ = height
            else:
                height_ = math.floor(height_ / width_ * height)
                width_ = width
        else:
            height_, width_ = height, width

        mat2 = cv2.resize(images[idx], (width_, height_))
        offset_y, offset_x = (height - height_) // 2, (width - width_) // 2
        start_y, start_x = idx // cols * height, idx % cols * width
        mat1[start_y + offset_y:start_y + height_+offset_y, start_x + offset_x:start_x + width_+offset_x, :] = mat2

    cv2.imwrite(images_path+name, mat1)

for name in images_name:
    img = cv2.imread(images_path + name)
    print(f"name: {images_path+name}, opencv image shape: {img.shape}") # (h,w,c)
    images_data.append(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img_pil)
    print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (c,h,w)
    tensor = torch.unsqueeze(tensor, 0) # 返回一个新的tensor,对输入的既定位置插入维度1
    print(f"tensor shape: {tensor.shape}, max: {torch.max(tensor)}, min: {torch.min(tensor)}") # (1,c,h,w)
    tensor_data.append(tensor)

images_stitch(images_data)

model = models.alexnet(pretrained=True) # AlexNet网络
#print(model) # 可查看模型结构,与torchvision/models/alexnet.py中一致
model.eval() # AlexNet is required to be put in evaluation mode in order to do prediction/evaluation

with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()] # the line number specified the class number

for x in range(len(tensor_data)):
    prediction = model(tensor_data[x])
    #print(prediction.shape) # [1,1000]
    _, index = torch.max(prediction, 1)
    percentage = torch.nn.functional.softmax(prediction, dim=1)[0] * 100
    print(f"result: {classes[index[0]]}, {percentage[index[0]].item()}")

print("test finish")

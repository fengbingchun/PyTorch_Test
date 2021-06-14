import torch

# reference: https://pytorch.apachecn.org/docs/1.7/43.html

# 通道在最后的内存格式是在保留内存尺寸的顺序中对NCHW张量进行排序的另一种方法. 通道最后一个张量的排序方式使通道成为
# 最密集的维度(又称为每像素存储图像).

N, C, H, W = 10, 3, 32, 32

# 内存格式API
x = torch.empty(N, C, H, W)
print(x.stride()) # Ouputs: (3072, 1024, 32, 1)

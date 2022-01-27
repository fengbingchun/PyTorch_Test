import torch
import torch.nn as nn

# Blog: https://blog.csdn.net/fengbingchun/article/details/122724181

loss = nn.CrossEntropyLoss()

input = torch.tensor([[0.0418, 0.0801, -1.3888, -1.9604, 1.0712]])
target = torch.tensor([2]).long() # target为2,one-hot表示为[0,0,1,0,0]
output = loss(input, target)
print("output:", output)

data1 = [[ 0.0418,  0.0801, -1.3888, -1.9604,  1.0712],
         [ 0.3519, -0.6115, -0.0325,  0.4484, -0.1736],
         [ 0.1530,  0.0670, -0.3894, -1.0830, -0.4757],
         [ -1.3519, 0.2115, 1.2325,  -1.4484, 0.9736],
         [ 1.1230,  -0.5670, 1.0894, 1.9890, 0.03567]]
data2 = [4, 3, 2, 1, 0]

input = torch.tensor(data1)
target = torch.tensor(data2)
output = loss(input, target)
print("output:", output)

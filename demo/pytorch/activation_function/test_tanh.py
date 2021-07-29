import numpy as np
import torch

# blog: https://blog.csdn.net/fengbingchun/article/details/119202855

data = [1.1, -2.2, 3.3, 0.4, -0.5, -1.6]

# numpy impl
def tanh(x):
	lists = list()
	for i in range(len(x)):
		lists.append((np.exp(x[i]) - np.exp(-x[i])) / (np.exp(x[i]) + np.exp(-x[i])))
	return lists

def tanh_derivative(x):
	return 1 - np.power(tanh(x), 2)

output = [round(value, 4) for value in tanh(data)] # 通过round保留小数点后4位
print("numpy tanh:", output)
print("numpt tanh derivative:", [round(value, 4) for value in tanh_derivative(data)])
print("numpt tanh derivative2:", [round(1. - value*value, 4) for value in tanh(data)])

# call pytorch interface
input = torch.FloatTensor(data)
m = torch.nn.Tanh()
output2 = m(input)
print("pytorch tanh:", output2)
print("pytorch tanh derivative:", 1. - output2*output2)

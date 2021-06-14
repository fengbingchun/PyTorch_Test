import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# reference: https://pytorch.apachecn.org/docs/1.7/05.html

# 可以使用torch.nn包构建神经网络. nn依赖于autograd来定义模型并对其进行微分.nn.Module包含层,以及返回output的方法forward(input).

class Net(nn.Module):
    def __init__(self):
            #super(Net, self).__init__() # python 2.x
            super().__init__() # python 3.x
            # 1: input image channel; 6: output channels; 3*3 square convolution kernel
            self.conv1 = nn.Conv2d(1, 6, 3)
            self.conv2 = nn.Conv2d(6, 16, 3)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(16 * 6 * 6, 120) # 6 * 6: from image dimension
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x): # 你只需要定义forward函数,就可以使用autograd为你自动定义backward函数(计算梯度)
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size()) # conv1's .weight

input = torch.randn(1, 1, 32, 32) # 32*32随机输入
out = net(input)
print(out)

# 使用随机梯度将所有参数和反向传播的梯度缓冲区归零
net.zero_grad()
out.backward(torch.randn(1, 10))
print(out)

# torch.nn 仅支持小批量. 整个torch.nn包仅支持作为微型样本而不是单个样本的输入.
# 例如,nn.Conv2d将采用nSamples * nChannels * Height * Width的4D张量

# 损失函数采用一对(输出,目标)输入,并计算一个值,该值估计输出与目标之间的距离
output = net(input)
target = torch.randn(10) # a dummy target,for example
target = target.view(1, -1) # make it the same shape as output
criterion = nn.MSELoss() # 计算输入和目标之间的均方误差

loss = criterion(output, target)
print(loss)

print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# 反向传播误差 loss.backward(), 需要清除现有的梯度,否则梯度将累积到现有的梯度中
net.zero_grad() # zeroes the gradient buffers of all parameters

print("conv1.bias.grad before backward")
print(net.conv1.bias.grad)

loss.backward()

print("conv1.bias.grad after backward")
print(net.conv1.bias.grad)

# 更新权重:最简单的更新规则是随机梯度下降(SGD): weight = weight - learning_rate * gradient
# 可使用不同的更新规则,例如SGD、Nesterov-SGD、Adam、RMSProp等,为实现此目的,构建了一个小包装: torch.optim
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad() # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # Does the update

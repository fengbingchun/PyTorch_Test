from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import math
#from IPython.core.debugger import set_trace

import torch
import torch.nn.functional as F # 通常按照惯例将其导入到名称空间F中
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# reference: https://pytorch.apachecn.org/docs/1.7/16.html

# MNIST数据集
# 使用pathlib处理路径,并使用requests下载数据集
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

# 数据集为numpy数组格式,并已使用pickle存储
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# 每个图像为28x28,并存储为长度为784 = 28x28的扁平行
pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
print(x_train.shape)

# 转换数据到torch.tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
x_train, x_train.shape, y_train.min(), y_train.max()
print(x_train, y_train)
print(x_train.shape)
print(y_train.min(), y_train.max())

# 从零开始的神经网络(没有torch.nn)
# 对于权重,我们在初始化之后设置requires_grad,因为我们不希望该步骤包含在梯度中.(请注意,PyTorch中的尾随_表示该操作是原地执行的.)
weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

# 由于PyTorch具有自动计算梯度的功能,我们可以将任何标准的Python函数(或可调用对象)用作模型
# 尽管PyTorch提供了许多预写的损失函数,激活函数等,但是你可以使用纯Python轻松编写自己的函数
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias) # @代表点积运算

bs = 64  # batch size

xb = x_train[0:bs]  # a mini-batch from x
preds = model(xb)  # predictions
preds[0], preds.shape
print(preds[0], preds.shape)

def nll(input, target): # 实现负对数可能性作为损失函数
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb)) # 使用随机模型来检查损失

def accuracy(out, yb): # 实现一个函数来计算模型的准确率
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()
print(accuracy(preds, yb))

lr = 0.5  # learning rate
epochs = 2  # how many epochs to train for

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        # set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print(loss_func(model(xb), yb), accuracy(model(xb), yb)) # 检查损失和准确率


# 将重构代码,使其执行与以前相同的操作,只是我们将开始利用PyTorch的nn类使其更加简洁和灵活
# 使用torch.nn.functional
loss_func = F.cross_entropy # 如果您使用的是负对数似然损失和对数softmax激活,那么Pytorch会提供结合了两者的单一函数F.cross_entropy

def model(xb):
    return xb @ weights + bias

print(loss_func(model(xb), yb), accuracy(model(xb), yb))

# 使用nn.Module重构
# 我们将nn.Module子类化(它本身是一个类并且能够跟踪状态). nn.Module具有许多我们将要使用的属性和方法
# nn.Module(大写M)是PyTorch的特定概念,并且是我们将经常使用的一类.不要将nn.Module与模块(小写m)的Python概念混淆,该模块是可以导入的Python代码文件
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784))
        self.bias = nn.Parameter(torch.zeros(10))

    def forward(self, xb):
        return xb @ self.weights + self.bias

model = Mnist_Logistic() # 由于我们现在使用的是对象而不是仅使用函数,因此我们首先必须实例化模型
print(loss_func(model(xb), yb)) # nn.Module对象的使用就好像它们是函数一样(即,它们是可调用的),但是在后台Pytorch会自动调用我们的forward方法

# 我们可以利用model.parameters()和model.zero_grad()(它们都由PyTorch为nn.Module定义)来更新每个参数的值,并将每个参数的梯度分别归零
#with torch.no_grad():
#    for p in model.parameters(): p -= p.grad * lr
#    model.zero_grad()

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()

fit()
print(loss_func(model(xb), yb))

# 使用nn.Linear重构
# Pytorch具有许多类型的预定义层
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = Mnist_Logistic()
print(loss_func(model(xb), yb))

# 使用optim重构
# Pytorch还提供了一个包含各种优化算法的包torch.optim.我们可以使用优化器中的step方法采取向前的步骤,而不是手动更新每个参数
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()
print(loss_func(model(xb), yb))

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad() # optim.zero_grad()将梯度重置为0,我们需要在计算下一个小批量的梯度之前调用它

print(loss_func(model(xb), yb))

# 使用Dataset重构
# PyTorch有一个抽象的Dataset类.数据集可以是具有__len__函数(由Python的标准len函数调用)和具有__getitem__函数作为对其进行索引的一种方法
# PyTorch的TensorDataset是一个数据集包装张量.通过定义索引的长度和方式,这也为我们提供了沿张量的第一维进行迭代,索引和切片的方法
train_ds = TensorDataset(x_train, y_train) # x_train和y_train都可以合并为一个TensorDataset,这将更易于迭代和切片

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        xb, yb = train_ds[i * bs: i * bs + bs]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# 使用DataLoader重构
# Pytorch的DataLoader负责批量管理.您可以从任何Dataset创建一个DataLoader.DataLoader使迭代迭代变得更加容易.
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

for xb,yb in train_dl:
    pred = model(xb)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

# 添加验证: 验证集不需要反向传播
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

# 注意,我们总是在训练之前调用model.train(),并在推理之前调用model.eval(),
# 因为诸如nn.BatchNorm2d和nn.Dropout之类的层会使用它们,以确保这些不同阶段的行为正确
model, opt = get_model()

for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))

# 创建fit()和get_data()
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)

# fit运行必要的操作来训练我们的模型,并计算每个周期的训练和验证损失
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)

# get_data返回训练和验证集的数据加载器
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

# 获取数据加载器和拟合模型的整个过程可以在 3 行代码中运行
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


# 构建具有三个卷积层的神经网络
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1) # 使用Pytorch的预定义Conv2d类作为我们的卷积层
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb)) # 每个卷积后跟一个ReLU
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4) # 平均池化
        return xb.view(-1, xb.size(1)) # view是numpy的reshape的PyTorch版本

lr = 0.1
model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9) # 动量是随机梯度下降的一种变体,它也考虑了以前的更新,通常可以加快训练速度

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# nn.Sequential: Sequential对象以顺序方式运行其中包含的每个模块
class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def preprocess(x):
    return x.view(-1, 1, 28, 28)

model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# 包装DataLoader
def preprocess(x, y):
    return x.view(-1, 1, 28, 28), y

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)

# 检查您的GPU是否在Pytorch中正常工作
print(torch.cuda.is_available())


# 总结
# torch.nn
#  Module:创建一个行为类似于函数的可调用对象,但也可以包含状态(例如神经网络层权重).它知道其中包含的Parameter,
#    并且可以将其所有梯度归零,遍历它们以进行权重更新等.
#  Parameter:张量的包装器,用于告知Module具有在反向传播期间需要更新的权重.仅更新具有require_grad属性集的张量
#  functional:一个模块(通常按照惯例导入到F名称空间中),其中包含激活函数,损失函数等.以及卷积和线性层等层的无状态版本.
# torch.optim:包含诸如SGD的优化程序,这些优化程序在后退步骤
# Dataset中更新Parameter的权重.具有__len__和__getitem__的对象,包括Pytorch提供的类,例如TensorDataset
# DataLoader:获取任何Dataset并创建一个迭代器,该迭代器返回批量数据.


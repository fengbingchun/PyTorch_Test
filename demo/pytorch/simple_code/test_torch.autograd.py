import torch, torchvision

# reference: https://pytorch.apachecn.org/docs/1.7/04.html

# 从torchvision加载经过预训练的resnet18模型
# Linux: 下载到~/.cache/torch/hub/checkpoints/目录下, windows下载到: C:\Users\spring\.cache\torch\hub\checkpoints
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64) # 创建一个随机数据张量来表示具有3个通道的单个图像
labels = torch.rand(1, 1000) # 随机初始化label值

prediction = model(data) # 进行预测,正向传播

loss = (prediction - labels).sum() # 使用模型的预测和相应的标签来计算误差(loss)
loss.backward() # 反向传播, 然后,Autograd会为每个模型参数计算梯度并将其存储在参数的.grad属性中

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) # 加载一个优化器,为SGD,学习率为0.01,动量为0.9. 在优化器中注册模型的所有参数

optim.step() # 调用.step()启动梯度下降.优化器通过.grad中存储的梯度来调整每个参数


# 用rquires_grad=True创建两个张量a和b. 这向autograd发出信号,应跟踪对它们的所有操作
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3 * a**3 - b**2 # Q = 3a^3-b^2,假设a和b是神经网络的参数,Q是误差

external_grad = torch.tensor([1., 1.])
# 在Q上调用.backward()时,Autograd将计算这些梯度并将其存储在各个张量的.grad属性中.需要在Q.backward()中显示传递gradient参数,因为它是向量
Q.backward(gradient=external_grad)
# check if collected gradients are correct
print(9*a**2 == a.grad)
print(-2*b == b.grad)

# 从概念上讲,Autograd在由函数对象组成的有向无环图(DAG)中记录数据(张量)和所有已执行的操作(以及由此产生的新张量).在此DAG中,叶子是输入张量,根是输出张量.
# 通过从根到跟踪此图,可以使用链式规则自动计算梯度
# DAG在PyTorch中是动态的.要注意的重要一点是,图是从头开始重新创建的

# torch.autograd跟踪所有将其requires_grad标志设置为True的张量的操作. 对于不需要梯度的张量,将此属性设置为False会将其从梯度计算DAG中排除
# 即使只有一个输入张量具有requires_grad=True, 操作的输出张量也将需要梯度
x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does a require gradients?: {a.requires_grad}")
b = x + z
print(f"Dose b require gradients?: {b.requires_grad}")

# 在NN中,不计算梯度的参数通常称为冻结参数.在微调中,我们冻结了大部分模型,通常仅修改分类器层以对新标签进行预测
# 加载一个预训练的resnet18模型,并冻结所有参数
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the network
for param in model.parameters():
    param.requires_grad = False
# 假设我们要在10个标签的新数据集中微调模型.在resnet中,分类器是最后一个线性层model.fc.
# 我们可以简单地将其替换为充当我们的分类器的新线性层(默认情况下未冻结)
model.fc = torch.nn.Linear(512, 10) # 现在,除了model.fc的参数外,模型中的所有参数都将冻结.计算梯度的唯一参数是model.fc的权重和偏差
# Optimize only the classifier
# 尽管我们在优化器中注册了所有参数,但唯一可计算梯度的参数是分类器的权重和偏差
optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)
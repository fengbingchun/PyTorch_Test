import torch
import numpy as np

# reference: https://pytorch.apachecn.org/docs/1.7/03.html

# 张量初始化
# 1.直接生成张量
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print("x_data:\n", x_data)

# 2.通过Numpy数组来生成张量,反过来也可以由张量生成Numpy数组
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print("x_np:\n", x_np)

# 3.通过已有的张量来生成新的张量: 新的张量将继承已有张量的属性(结构、类型),也可以重新指定新的数据类型
x_ones = torch.ones_like(x_data) # 保留x_data的属性
print("x_ones:\n", x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float) # 重写x_data的数据类型: int -> float
print("x_rand:\n", x_rand)

# 4.通过指定数据维度来生成张量
shape = (2,3,) # shape是元组类型,用来描述张量的维数
rand_tensor = torch.rand(shape)
print("rand_tensor:\n", rand_tensor)
ones_tensor = torch.ones(shape)
print("ones_tensor:\n", ones_tensor)
zeros_tensor = torch.zeros(shape)
print("zeros_tensor:\n", zeros_tensor)

# 张量属性: 从张量属性我们可以得到张量的维数、数据类型以及它们所存储的设备(CPU或GPU)
tensor = torch.rand(3, 4)
print(f"shape of tensor: {tensor.shape}")
print(f"datatype of tensor: {tensor.dtype}")
print(f"device tensor is stored on: {tensor.device}")

# 张量运算: 有超过100种张量相关的运算操作,例如转置、索引、切片、数学运算、线性代数、随机采样等.
# 所有这些运算都可以在GPU上运行(相对于CPU来说可以达到更高的运算速度)
# 判断当前环境GPU是否可用,然后将tensor导入GPU内运行
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

# 1.张量的索引和切片
tensor = torch.ones(4, 4)
tensor[:, 1] = 0 # 将第1列(从0开始)的数据全部赋值为0
print(f"tensor:\n {tensor}")

# 2.张量的拼接: 可以通过torch.cat方法将一组张量按照指定的维度进行拼接,也可以参考torch.stack方法,但与torch.cat稍微有点不同
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(f"t1:\n {t1}")

# 3.张量的乘积和矩阵乘法
print(f"tensor.mul(tensor):\n {tensor.mul(tensor)}") # 逐个元素相乘结果
print(f"tensor * tensor:\n {tensor * tensor}") # 等价写法

print(f"tensor.matmul(tensor.T):\n {tensor.matmul(tensor.T)}") # 张量与张量的矩阵乘法
print(f"tensor @ tensor.T:\n {tensor @ tensor.T}") # 等价写法

# 4.自动赋值运算: 通常在方法后有"_"作为后缀,例如:x.copy_(y), x.t_()操作会改变x的取值
print(f"tensor:\n {tensor}")
tensor.add_(5)
print(f"tensor:\n {tensor}")

# Tensor与Numpy的转化: 张量和Numpy array数组在CPU上可以共用一块内存区域,改变其中一个另一个也会随之改变
# 1.由张量变换为Numpy array数组
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1) # 修改张量的值,则Numpy array数组值也会随之改变
print(f"t: {t}")
print(f"n: {n}")

# 2.由Numpy array数组转为张量
n = np.ones(5)
print(f"n: {n}")
t = torch.from_numpy(n)
print(f"t: {t}")

np.add(n, 1, out=n) # 修改Numpy array数组的值,则张量值也会随之改变
print(f"n: {n}")
print(f"t: {t}")

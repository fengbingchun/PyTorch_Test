from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# reference: https://pytorch.apachecn.org/docs/1.7/22.html

# 什么是GAN？ GAN由Ian Goodfellow于2014年发明，并在论文《生成对抗网络》中首次进行了描述。它们由两个不同的模型
# 组成：生成器和判别器。生成器的工作是生成看起来像训练图像的"假"图像。判别器的工作是查看图像并从生成器输出它是真
# 实的训练图像还是伪图像。在训练过程中，生成器不断尝试通过生成越来越好的伪造品而使判别器的表现超过智者，而判别器
# 正在努力成为更好的侦探并正确地对真实和伪造图像进行分类。博弈的平衡点是当生成器生成的伪造品看起来像直接来自训练
# 数据时，而判别器则总是猜测生成器输出是真实还是伪造品的50%置信度。

# 什么是DCGAN？ DCGAN是上述GAN的直接扩展，不同之处在于，DCGAN分别在判别器和生成器中分别使用卷积和卷积转置层。
# 它最早由Radford等人，在论文《使用深度卷积生成对抗网络的无监督表示学习》中描述。判别器由分层的卷积层，批量
# 规范层和LeakyReLU激活组成。

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


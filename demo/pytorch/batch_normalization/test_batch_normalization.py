import torch
from torch import nn
import numpy as np

# Blog: https://blog.csdn.net/fengbingchun/article/details/118959997

# reference: https://github.com/Johann-Huber/batchnorm_pytorch/blob/main/batch_normalization_in_pytorch.ipynb

# BatchNorm reimplementation
class myBatchNorm2d(nn.Module):
    def __init__(self, input_size = None , epsilon = 1e-5, momentum = 0.99):
        super(myBatchNorm2d, self).__init__()
        assert input_size, print('Missing input_size parameter.')

        # Batch mean & var must be defined during training
        self.mu = torch.zeros(1, input_size)
        self.var = torch.ones(1, input_size)

        # For numerical stability
        self.epsilon = epsilon

        # Exponential moving average for mu & var update
        self.it_call = 0  # training iterations
        self.momentum = momentum # EMA smoothing

        # Trainable parameters
        self.beta = torch.nn.Parameter(torch.zeros(1, input_size))
        self.gamma = torch.nn.Parameter(torch.ones(1, input_size))

        # Batch size on which the normalization is computed
        self.batch_size = 0

    def forward(self, x):
        # [batch_size, input_size]

        self.it_call += 1

        if self.training:
            print("Info: training ...")
            if( self.batch_size == 0 ):
                # First iteration : save batch_size
                self.batch_size = x.shape[0]

            # Training : compute BN pass
            #batch_mu = (x.sum(dim=0)/x.shape[0]).unsqueeze(0) # [1, input_size]
            batch_mu = torch.mean(x, dim=0)
            #batch_var = (x.var(dim=0)/x.shape[0]).unsqueeze(0)*2 # [1, input_size]
            batch_var = torch.var(x, unbiased=False, dim=0)
            #print("batch_mu:", batch_mu)
            #print("batch_var:", batch_var)

            x_normalized = (x-batch_mu)/torch.sqrt(batch_var + self.epsilon) # [batch_size, input_size]
            x_bn = self.gamma * x_normalized + self.beta # [batch_size, input_size]

            # Update mu & std
            if(x.shape[0] == self.batch_size):
                running_mu = batch_mu
                running_var = batch_var
            else:
                running_mu = batch_mu*self.batch_size/x.shape[0]
                running_var = batch_var*self.batch_size/x.shape[0]

            self.mu = running_mu * (self.momentum/self.it_call) + \
                            self.mu * (1 - (self.momentum/self.it_call))
            self.var = running_var * (self.momentum/self.it_call) + \
                        self.var * (1 - (self.momentum/self.it_call))

        else:
            print("Info: inference ...")
            # Inference: compute BN pass using estimated mu & var
            if (x.shape[0] == self.batch_size):
                estimated_mu = self.mu
                estimated_var = self.var
            else :
                estimated_mu = self.mu*x.shape[0]/self.batch_size
                estimated_var = self.var*x.shape[0]/self.batch_size

            x_normalized = (x-estimated_mu)/torch.sqrt(estimated_var + self.epsilon) # [batch_size, input_size]
            x_bn = self.gamma * x_normalized + self.beta # [batch_size, input_size]

        return x_bn # [batch_size, output_size=input_size]

# N = 3, C = 1, H = 1, W = 6
input_size = 1 # channel
bn = myBatchNorm2d(input_size)
data =  [[[[11.1, -2.2, 23.3, 54.4, 58.5, -16.6]]],
		[[[-97.7, -28.8, 49.9, -61.3, 52.6, -33.9]]],
		[[[-2.45, -15.7, 72.4, 9.1, 47.2, 21.7]]]]
input = torch.FloatTensor(data) # [N, C, H, W]
print("input:", input)
output = bn.forward(input)
print("output:", output)

'''
print("######################")
a = np.array(data)
print(np.mean(a, axis=0))
print(np.var(a, axis=0))
'''

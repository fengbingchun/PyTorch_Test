import sys
sys.path.append("..") # 为了导入pytorch目录中的内容
from pytorch.lenet5.test_lenet5_mnist import LeNet5
import torch

# Blog: https://blog.csdn.net/fengbingchun/article/details/125706670

def print_state_dict(model_state_dict, optimizer_state_dict):
    print("print model's state_dict:")
    for param_tensor in model_state_dict:
        print("  ", param_tensor, "\t", model_state_dict[param_tensor].size())

    print("print optimizer's state_dict:")
    for var_name in optimizer_state_dict:
        print("  ", var_name, "\t", optimizer_state_dict[var_name])

def save_load_model(model):
    '''saving and loading models'''
    model.load_state_dict(torch.load("../../data/Lenet-5.pth")) # 加载模型
    model.eval() # 将网络设置为评估模式

    # state_dict:返回一个字典,保存着module的所有状态,参数和persistent buffers都会包含在字典中,字典的key就是参数和buffer的names
    print("model state dict keys:", model.state_dict().keys())
    print("model type:", type(model)) # model type: <class 'pytorch.lenet5.test_lenet5_mnist.LeNet5'>
    print("model state dict type:", type(model.state_dict())) # model state dict type: <class 'collections.OrderedDict'>

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    print_state_dict(model.state_dict(), optimizer.state_dict())

    torch.save(model, "../../data/Lenet-5_all.pth") # 保存整个模型
    torch.save(model.state_dict(), "../../data/Lenet-5_parameters.pth") # 推荐:仅保存训练模型的参数,为以后恢复模型提供最大的灵活性

def save_load_checkpoint(model):
    '''saving & loading a general checkpoint for inference and/or resuming training'''
    path = "../../data/Lenet-5_parameters.tar"
    model.load_state_dict(torch.load("../../data/Lenet-5.pth")) # 加载模型
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    torch.save({
                'epoch': 5,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, path)

    checkpoint = torch.load(path)
    model2 = LeNet5(n_classes=10).to('cpu') # 实例化一个LeNet5网络对象
    model2.load_state_dict(checkpoint['model_state_dict'])
    optimizer2 = torch.optim.SGD(params=model2.parameters(), lr=0.1)
    optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print("epoch:", epoch)
    model.eval() # 将网络设置为评估模式
    #model.train() # 恢复训练,将网络设置为训练模式

    print_state_dict(model2.state_dict(), optimizer2.state_dict())

def save_load_multiple_models():
    '''saving multiple models in one file'''
    path1 = "../../data/Lenet-5.pth"
    path2 = "../../data/Lenet-5_parameters_mul.tar"
    model1 = LeNet5(n_classes=10).to('cpu') # 实例化一个LeNet5网络对象
    model1.load_state_dict(torch.load(path1)) # 加载模型
    optimizer1 = torch.optim.Adam(params=model1.parameters(), lr=0.001)

    model2 = LeNet5(n_classes=10).to('cpu') # 实例化一个LeNet5网络对象
    model2.load_state_dict(torch.load(path1)) # 加载模型
    optimizer2 = torch.optim.SGD(params=model2.parameters(), lr=0.1)

    torch.save({
            'epoch': 100,
            'model1_state_dict': model1.state_dict(),
            'model2_state_dict': model2.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            }, path2)

    checkpoint = torch.load(path2)
    modelA = LeNet5(n_classes=10).to('cpu') # 实例化一个LeNet5网络对象
    modelA.load_state_dict(checkpoint['model1_state_dict'])
    optimizerA = torch.optim.SGD(params=modelA.parameters(), lr=0.1)
    optimizerA.load_state_dict(checkpoint['optimizer1_state_dict'])

    modelB = LeNet5(n_classes=10).to('cpu') # 实例化一个LeNet5网络对象
    modelB.load_state_dict(checkpoint['model2_state_dict'])
    optimizerB = torch.optim.Adam(params=modelB.parameters(), lr=0.01)
    optimizerB.load_state_dict(checkpoint['optimizer2_state_dict'])

    epoch = checkpoint['epoch']
    print("epoch:", epoch)
    modelA.eval() # 将网络设置为评估模式
    #modelA.train() # 恢复训练,将网络设置为训练模式

    #modelB.eval() # 将网络设置为评估模式
    modelB.train() # 恢复训练,将网络设置为训练模式

    print_state_dict(modelA.state_dict(), optimizerA.state_dict())
    print_state_dict(modelB.state_dict(), optimizerB.state_dict())

def main():
    model = LeNet5(n_classes=10).to('cpu') # 实例化一个LeNet5网络对象

    #save_load_model(model)
    #save_load_checkpoint(model)
    save_load_multiple_models()

    print("test finish")

if __name__ == "__main__":
    main()

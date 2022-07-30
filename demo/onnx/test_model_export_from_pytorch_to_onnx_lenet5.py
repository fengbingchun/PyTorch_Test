import sys
sys.path.append("..") # 为了导入pytorch目录中的内容
from pytorch.lenet5.test_lenet5_mnist import LeNet5, list_files, get_image_label
import torch
import onnx
import cv2
import numpy as np
import onnxruntime

# Blog: https://blog.csdn.net/fengbingchun/article/details/126072998

def load_pytorch_model(model_name):
    model = LeNet5(n_classes=10).to('cpu') # 实例化一个LeNet5网络对象
    model.load_state_dict(torch.load(model_name)) # 加载pytorch模型
    model.eval() # 将网络设置为评估模式

    return model

def export_model_from_pytorch_to_onnx(pytorch_model, onnx_model_name):
    batch_size = 1
    # input to the model
    x = torch.randn(batch_size, 1, 32, 32)
    out = pytorch_model(x)
    #print("out:", out)

    # export the model
    torch.onnx.export(pytorch_model,             # model being run
                      x,                         # model input (or a tuple for multiple inputs)
                      onnx_model_name,           # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=9,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})

def verify_onnx_model(onnx_model_name):
    # model is an in-memory ModelProto
    model = onnx.load(onnx_model_name)
    #print("the model is:\n{}".format(model))

    # check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("  the model is invalid: %s" % e)
        exit(1)
    else:
        print("  the model is valid")

def image_preprocess(image_names, image_name_suffix):
    input_data = []
    labels = []

    for image_name in image_names:
        label = get_image_label(image_name, image_name_suffix)
        labels.append(label)

        img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        # MNIST图像背景为黑色,而测试图像的背景色为白色,识别前需要做转换
        img = cv2.bitwise_not(img)
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = norm_img.reshape(1, 1, 32, 32).astype('float32')
        #print(f"img type: {type(norm_img)}, shape: {norm_img.shape}")
        input_data.append(norm_img)

    return input_data, labels

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

def inference(model_name, image_names, input_data, labels):
    session = onnxruntime.InferenceSession(model_name, None)
    # get the name of the first input of the model
    input_name = session.get_inputs()[0].name
    count = 0

    for data in input_data:
        raw_result = session.run([], {input_name: data})

        res = postprocess(raw_result)
        idx = np.argmax(res)

        image_name = image_names[count][image_names[count].rfind("/")+1:]
        print(f"  image name: {image_name}, actual value: {labels[count]}, predict value: {idx}, percentage: {round(res[idx]*100, 4)}%")
        count += 1

def main():
    print("1.load pytorch model")
    pytorch_model_name = "../../data/Lenet-5.pth"
    pytorch_model = load_pytorch_model(pytorch_model_name)

    print("2.export onnx model")
    onnx_model_name = "../../data/Lenet-5.onnx"
    export_model_from_pytorch_to_onnx(pytorch_model, onnx_model_name)
    verify_onnx_model(onnx_model_name)

    print("3.prepare test images")
    image_path = "../../data/image/handwritten_digits/"
    image_name_suffix = ".png"
    image_names = list_files(image_path, image_name_suffix)
    input_data, labels = image_preprocess(image_names, image_name_suffix)

    print("4.inference")
    inference(onnx_model_name, image_names, input_data, labels)

    print("test finish")

if __name__ == "__main__":
    main()
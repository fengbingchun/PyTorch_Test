import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference, version_converter
from onnx import AttributeProto, TensorProto, GraphProto

# Blog: https://blog.csdn.net/fengbingchun/article/details/125947000

# reference: https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md

def manipulate_tensorproto_and_numpy_array():
    # create a Numpy array
    numpy_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    print("Original Numpy array:\n{}\n".format(numpy_array))

    # convert the Numpy array to a TensorProto
    tensor = numpy_helper.from_array(numpy_array)
    print("TensorProto:\n{}".format(tensor))

    # convert the TensorProto to a Numpy array
    new_array = numpy_helper.to_array(tensor)
    print("After round trip, Numpy array:\n{}\n".format(new_array))

    tensorproto_name = "../../data/tensor.pb"
    # save the TensorProto
    with open(tensorproto_name, 'wb') as f:
        f.write(tensor.SerializeToString())
    
    # load a TensorProto
    new_tensor = onnx.TensorProto()
    with open(tensorproto_name, 'rb') as f:
        new_tensor.ParseFromString(f.read())
    print("After saving and loading, new TensorProto:\n{}".format(new_tensor))

def run_shape_inference():
    # preprocessing: create a model with two nodes, Y's shape is unknown
    node1 = helper.make_node('Transpose', ['X'], ['Y'], perm=[1, 0, 2])
    node2 = helper.make_node('Transpose', ['Y'], ['Z'], perm=[1, 0, 2])

    graph = helper.make_graph(
        [node1, node2],
        'two-transposes',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 3, 4))],
        [helper.make_tensor_value_info('Z', TensorProto.FLOAT, (2, 3, 4))],
    )

    original_model = helper.make_model(graph, producer_name='onnx-examples-2')

    # check the model and print Y's shape information
    onnx.checker.check_model(original_model)
    print("before shape inference, the shape info of Y is:\n{}".format(original_model.graph.value_info))

    # apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(original_model)

    # check the model and print Y's shape information
    onnx.checker.check_model(inferred_model)
    print("after shape inference, the shape info of Y is:\n{}".format(inferred_model.graph.value_info))

def create_save_onnx_model(model_name):
    # create one input(ValueInfoProto)
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
    pads = helper.make_tensor_value_info('pads', TensorProto.FLOAT, [1, 4])
    value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, [1])

    # create one output(ValueInfoProto)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

    # create a node(NodeProto): this is based on Pad-11
    node_def = helper.make_node(
        'Pad',                  # name
        ['X', 'pads', 'value'], # inputs
        ['Y'],                  # outputs
        mode='constant',        # attributes
    )

    # create the graph(GraphProto)
    graph_def = helper.make_graph(
        [node_def],        # nodes
        'test-model',      # name
        [X, pads, value],  # inputs
        [Y],               # outputs
    )

    # create the model(ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnx-example')

    # save the ONNX model
    onnx.save(model_def, model_name)

def load_check_onnx_model(model_name):
    # model is an in-memory ModelProto
    model = onnx.load(model_name)
    print("the model is:\n{}".format(model))

    # check the model
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print("the model is invalid: %s" % e)
    else:
        print("the model is valid!")

def main():
    manipulate_tensorproto_and_numpy_array()
    run_shape_inference()

    model_name = "../../data/example.onnx"
    create_save_onnx_model(model_name)
    load_check_onnx_model(model_name)

    print("test finish")

if __name__ == "__main__":
    main()
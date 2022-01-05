from typing import Callable, Any, List

# Blog: https://blog.csdn.net/fengbingchun/article/details/122331018

def _forward_unimplemented(self, *input: Any) -> None:
    "Should be overridden by all subclasses"
    print("_forward_unimplemented")
    raise NotImplementedError

class Module:
    def __init__(self):
        print("Module.__init__")

    forward: Callable[..., Any] = _forward_unimplemented

    def _call_impl(self, *input, **kwargs):
        print("Module._call_impl")
        result = self.forward(*input, **kwargs)
        return result

    __call__: Callable[..., Any] = _call_impl

    def cpu(self):
        print("Module.cpu")

class AlexNet(Module):
    def __init__(self):
        print("AlexNet.__init__")
        super(AlexNet, self).__init__()

    # def forward(self, x):
    #     print("AlexNet.forward")
    #     return x

model = AlexNet()
x: List[int] = [1, 2, 3, 4]
print("result:", model(x))

model.cpu()

print("test finish")

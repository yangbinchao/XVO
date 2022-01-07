import torch
import torchvision
import torchvision.models as models
from torch.autograd import Variable
import onnx
import onnxruntime as ort
import numpy as np

def torch_onnx(torch_model_path = "test.pth", onnx_model_path = "test.onnx", model_parameters = False, pretrain_model = True):
    print('=> start transform pytorch to onnx')
    print('=> torch version is {}'.format(torch.__version__))

    if model_parameters:  # save model is model parameters, not whole model
        torch_model = torch.load(torch_model_path)
        model = models.resnet50()  # model construct
        model.load_state_dict(torch_model).cuda()
    elif pretrain_model:
        model = torchvision.models.resnet50(pretrained=True).cuda()
    else:
        torch_model = torch.load(torch_model_path)
        model = torch_model.cuda()
    
    model.eval()  # set the model to inference mode

    batch_size = 1  
    input_shape = (3, 224, 224)   
    input = Variable(torch.randn(batch_size, *input_shape)).cuda()
    
    torch.onnx.export(model,
                        input,
                        onnx_model_path,
                        opset_version=10,
                        do_constant_folding=True,	# 是否执行常量折叠优化
                        input_names=["input"],	
                        output_names=["output"],	
                        dynamic_axes={"input":{0:"batch_size", 1: "channel", 2: "height", 3: "width"},  \
                            "output":{0:"batch_size"}},
                        verbose=False)  # display detailed information when running
    print('=> onnx file is here {}'.format(onnx_model_path))
    print('\ncompleted')

def check_onnx(onnx_model_path = 'resnet50.onnx'):
    print("=> start check onnx model")
    model = onnx.load(onnx_model_path)
    onnx.checker.check_model(model)
    if False: # Print a human readable representation of the graph
        print(onnx.helper.printable_graph(model.graph))  
    print("\n=> passed")

def run_onnx(onnx_model_path = 'resnet50.onnx'):
    print("=> Start run and test onnx model")
    ort_session = ort.InferenceSession(onnx_model_path)
    outputs = ort_session.run(None,{"input": np.random.randn(1, 3, 224, 224).astype(np.float32)},)
    print(outputs[0])
    print('\ncompleted')


if __name__ == "__main__":
    torch_model_path = "test.pth"
    onnx_model_path = "test.onnx"
    torch_onnx(torch_model_path,onnx_model_path)
    check_onnx(onnx_model_path)
    run_onnx(onnx_model_path)



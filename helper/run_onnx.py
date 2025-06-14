import os
import time
import cv2
import torch
import torchvision
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from PIL import Image


TRT_LOGGER = trt.Logger()  # This logger is required to build an engine
dymatic_size = True
single_size = False

def get_img_np_nchw(filename):
    image = cv2.imread(filename)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_cv = cv2.resize(image_cv, (224, 224))
    miu = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(image_cv, dtype=float) / 255.
    r = (img_np[:, :, 0] - miu[0]) / std[0]
    g = (img_np[:, :, 1] - miu[1]) / std[1]
    b = (img_np[:, :, 2] - miu[2]) / std[2]
    img_np_t = np.array([r, g, b])
    img_np_nchw = np.expand_dims(img_np_t, axis=0)
    return img_np_nchw

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        Within this context, host_mom means the cpu memory and device means the GPU memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_engine(max_batch_size=1, onnx_file_path="", engine_file_path="", \
               fp16_mode=False, int8_mode=False, save_engine=False):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine(max_batch_size, save_engine):
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, \
                builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            if single_size:
                builder.max_workspace_size = 1 << 30 # Your workspace size
            elif dymatic_size:
                config = builder.create_builder_config()
                config.max_workspace_size = 1 << 30 
            builder.max_batch_size = max_batch_size
            # pdb.set_trace()
            builder.fp16_mode = fp16_mode  # Default: False
            builder.int8_mode = int8_mode  # Default: False
            if int8_mode:
                # To be updated
                raise NotImplementedError

            # Parse model file
            if not os.path.exists(onnx_file_path):
                quit('ONNX file {} not found'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())

            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))

            if single_size:
                engine = builder.build_cuda_engine(network)  # single batchsize
            elif dymatic_size:
                profile = builder.create_optimization_profile()
                profile.set_shape("input", (1,3,224,224),(1,3,224,224), (1,3,224,224))
                config.add_optimization_profile(profile)
                engine = builder.build_engine(network, config)

            print("Completed creating Engine")

            if save_engine:
                with open(engine_file_path, "wb") as f:
                    f.write(engine.serialize())
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine(max_batch_size, save_engine)


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def postprocess_the_outputs(h_outputs, shape_of_output):
    h_outputs = h_outputs.reshape(*shape_of_output)
    return h_outputs

if __name__ == "__main__":
    filename = 'test.jpg'
    max_batch_size = 1
    onnx_model_path = "test.onnx" # "test.onnx"  'resnet50.onnx'
    # These two modes are dependent on hardwares
    fp16_mode = False
    int8_mode = False
    trt_engine_path = './model_fp16_{}_int8_{}.trt'.format(fp16_mode, int8_mode)

    img_np_nchw = get_img_np_nchw(filename)
    img_np_nchw = img_np_nchw.astype(dtype=np.float32)

    # Build an engine
    engine = get_engine(max_batch_size, onnx_model_path, trt_engine_path, fp16_mode, int8_mode)
    # Create the context for this engine
    context = engine.create_execution_context()
    # Allocate buffers for input and output
    inputs, outputs, bindings, stream = allocate_buffers(engine) # input, output: host # bindings

    # Do inference
    shape_of_output = (max_batch_size, 1000)
    # Load data to the buffer
    inputs[0].host = img_np_nchw.reshape(-1)

    # inputs[1].host = ... for multiple input
    t1 = time.time()
    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream) # numpy data
    t2 = time.time()
    feat = postprocess_the_outputs(trt_outputs[0], shape_of_output)

    print('TensorRT ok')

    model = torchvision.models.resnet50(pretrained=True).cuda()
    resnet_model = model.eval()

    input_for_torch = torch.from_numpy(img_np_nchw).cuda()
    t3 = time.time()
    feat_2= resnet_model(input_for_torch)
    t4 = time.time()
    feat_2 = feat_2.cpu().data.numpy()

    print('Pytorch ok')


    mse = np.mean((feat - feat_2)**2)
    print("Inference time with the TensorRT engine: {}".format(t2-t1))
    print("Inference time with the PyTorch model: {}".format(t4-t3))
    print('MSE Error = {}'.format(mse))

    print('All completed')
# trt_engine.py
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import tensorrt as trt

def init():   # 1. 子进程开始初始化cuda driver
    cuda.init()

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
TRT_LOGGER = trt.Logger()

class TensorRTEngine(object):
    def __init__(self, file, batch_size=1):
        self.cfx = cuda.Device(0).make_context()  #2. trt engine创建前首先初始化cuda上下文
        # self.engine, self.network = self.load_engine_from_onnx(file, batch_size)
        self.engine = self.load_engine_from_trt(file, batch_size)
        self.input_shape, self.output_shape = self.infer_shape()
        print(self.input_shape)
        print(self.output_shape)
        
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    def __del__(self):
        del self.inputs
        del self.outputs
        del self.stream
        self.cfx.detach() # 2. 实例释放时需要detech cuda上下文
    
    def load_engine_from_trt(self, trt_file, batch_size=1):
        f = open(trt_file, "rb")                     # 读取trt模型
        runtime = trt.Runtime(TRT_LOGGER)   # 创建一个Runtime(传入记录器Logger)
        engine = runtime.deserialize_cuda_engine(f.read())      # 从文件中加载trt引擎
        return engine
        
    def load_engine_from_onnx(self, onnx_file, batch_size=1):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_batch_size = batch_size
            builder.max_workspace_size = 1 << 30
            with open(onnx_file, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
            engine = builder.build_cuda_engine(network)
        print("Load onnx sucessful!")
        
        return engine, network

    def infer_shape(self):
        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                input_shape = self.engine.get_binding_shape(binding)
            else:
                output_shape = self.engine.get_binding_shape(binding)
        
        return input_shape, output_shape
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self.engine:
            # size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            size = trt.volume(self.engine.get_binding_shape(binding))  #采用固定bs的trt文件时
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def do_inference(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def preprocess(self, data):
        data = data.reshape(-1)
        return data

    def postprocess(self, data):
        data = data.reshape(self.output_shape)
        return data

    def inference(self, data):
        self.inputs[0].host = self.preprocess(data)
        self.cfx.push()  # 3. 推理前执行cfx.push()
        # print(self.outputs[0].host.shape)
        trt_outputs = self.do_inference(self.context, bindings=self.bindings, 
                                            inputs=self.inputs, 
                                            outputs=self.outputs, 
                                            stream=self.stream)
        
        output = self.postprocess(trt_outputs[0])
        self.cfx.pop()  # 3. 推理后执行cfx.pop()

        return output
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import onnxruntime as ort

# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

from spcl import datasets
from spcl import models
from spcl.models.dsbn import convert_dsbn, convert_bn
from spcl.evaluators import Evaluator, EvaluatorONNX, EvaluatorTRT
from spcl.utils.data import transforms as T
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.trt_engine import TensorRTEngine, init

def get_data(name, data_dir, height, width, batch_size, workers):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, test_loader


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

# class HostDeviceMem(object):
#     def __init__(self, host_mem, device_mem):
#         """
#         host_mem: cpu memory
#         device_mem: gpu memory
#         """
#         self.host = host_mem
#         self.device = device_mem

#     def __str__(self):
#         return "Host:\n" + str(self.host)+"\nDevice:\n"+str(self.device)

#     def __repr__(self):
#         return self.__str__()

def main_worker(args):
    cudnn.benchmark = True

    log_dir = osp.dirname(args.resume)
    sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    dataset, test_loader = get_data(args.dataset, args.data_dir, args.height,
                                    args.width, args.batch_size, args.workers)

    # Create model
    if args.model_type == 'onnx':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']

        ort_sess = ort.InferenceSession(args.resume, providers=providers)
        # print(ort_sess.get_providers())
        in_name = [input.name for input in ort_sess.get_inputs()][0]
        out_name = [output.name for output in ort_sess.get_outputs()]

        # Evaluator
        evaluator = EvaluatorONNX(ort_sess, in_name, out_name)
        print("Test on {}:".format(args.dataset))
        evaluator.evaluate(test_loader, dataset.query,
                           dataset.gallery, cmc_flag=True, rerank=args.rerank)

    elif args.model_type == 'trt':
        # cuda.init()
        # cfx = cuda.Device(0).make_context()

        # 1. 确定batch size大小，与导出的trt模型保持一致
        # BATCH_SIZE = 32

        # 2. 选择是否采用FP16精度，与导出的trt模型保持一致
        # USE_FP16 = True                                         
        # target_dtype = np.float16 if USE_FP16 else np.float32

        # 3. 创建Runtime，加载TRT引擎
        # f = open(args.resume, "rb")                     # 读取trt模型
        # runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))   # 创建一个Runtime(传入记录器Logger)
        # engine = runtime.deserialize_cuda_engine(f.read())      # 从文件中加载trt引擎
        # context = engine.create_execution_context()             # 创建context

        # 4. 分配input和output内存
        # input_batch = np.random.randn(args.batch_size, 3, args.height, args.width).astype(target_dtype)
        # output = np.empty([args.batch_size, 1024], dtype = target_dtype)

        # d_input = cuda.mem_alloc(1 * input_batch.nbytes)
        # d_output = cuda.mem_alloc(1 * output.nbytes)

        # bindings = [int(d_input), int(d_output)]

        # stream = cuda.Stream()


        # inputs, outputs, bindings = [], [], []

        # for binding in engine:
        #     # print(binding) # 绑定的输入输出
        #     # print(engine.get_binding_shape(binding)) # get_binding_shape 是变量的大小
        #     # size = trt.volume(engine.get_binding_shape(binding))*engine.max_batch_size
        #     # volume 计算可迭代变量的空间，指元素个数
        #     size = trt.volume(engine.get_binding_shape(binding)) # 如果采用固定bs的onnx，则采用该句
        #     dtype = trt.nptype(engine.get_binding_dtype(binding))
        #     # get_binding_dtype  获得binding的数据类型
        #     # nptype等价于numpy中的dtype，即数据类型
        #     # allocate host and device buffers
        #     host_mem = cuda.pagelocked_empty(size, dtype)  # 创建锁业内存
        #     device_mem = cuda.mem_alloc(host_mem.nbytes)    # cuda分配空间
        #     # print(int(device_mem)) # binding在计算图中的缓冲地址
        #     bindings.append(int(device_mem))
        #     #append to the appropriate list
        #     if engine.binding_is_input(binding):
        #         inputs.append(HostDeviceMem(host_mem, device_mem))
        #     else:
        #         outputs.append(HostDeviceMem(host_mem, device_mem))

        init()
        USE_FP16 = False                                       
        target_dtype = np.float16 if USE_FP16 else np.float32
        trt_engine = TensorRTEngine(args.resume, batch_size=256)
        # Evaluator
        evaluator = EvaluatorTRT(trt_engine, target_dtype)
        print("Test on {}:".format(args.dataset))
        evaluator.evaluate(test_loader, dataset.query,
                           dataset.gallery, cmc_flag=True, rerank=args.rerank)
        
    else:
        model = models.create(args.arch, pretrained=False,
                              num_features=args.features, dropout=args.dropout, num_classes=0)
        if args.dsbn:
            print("==> Load the model with domain-specific BNs")
            convert_dsbn(model)

        # Load from checkpoint
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model, strip='module.')

        if args.dsbn:
            print(
                "==> Test with {}-domain BNs".format("source" if args.test_source else "target"))
            convert_bn(model, use_target=(not args.test_source))

        model.cuda()
        model = nn.DataParallel(model)

        # Evaluator
        model.eval()
        evaluator = Evaluator(model)
        print("Test on {}:".format(args.dataset))
        evaluator.evaluate(test_loader, dataset.query,
                           dataset.gallery, cmc_flag=True, rerank=args.rerank)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Testing the model")
    # data
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--resume', type=str, required=True, metavar='PATH')
    parser.add_argument('--model-type', type=str, default='pth', required=True)
    # testing configs
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--dsbn', action='store_true',
                        help="test on the model with domain-specific BN")
    parser.add_argument('--test-source', action='store_true',
                        help="test on the source domain")
    parser.add_argument('--seed', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    main()

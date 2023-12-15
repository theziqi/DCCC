from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
# import onnxruntime as ort

# import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit

from .evaluation_metrics import cmc, mean_ap
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch, to_numpy


def extract_cnn_feature(model, inputs):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50, with_path=False):
    model.eval()
    batch_time = AverageMeter()
    # data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    images = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            # data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs)
            for fname, output, pid, img in zip(fnames, outputs, pids, imgs):
                features[fname] = output
                labels[fname] = pid
                images[fname] = img

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      #   'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              #   data_time.val, data_time.avg
                              ))
    if with_path:
        return features, labels, images
    else:
        return features, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # dist_m.addmm_(1, -2, x, y.t())
    dist_m.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist_m, x.numpy(), y.numpy()


def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), cmc_flag=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    if (not cmc_flag):
        return mAP

    cmc_configs = {
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True), }
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores:')
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}'.format(k, cmc_scores['market1501'][k-1]))
    return cmc_scores['market1501'], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features(self.model, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(
            features, query, gallery)
        results = evaluate_all(query_features, gallery_features,
                               distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(
            distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)


def extract_features_onnx(sess, in_name, out_name, data_loader, print_freq=50, with_path=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    images = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        inputs = to_numpy(imgs)
        # outputs = model(inputs)
        outputs = sess.run(out_name, {in_name: inputs})[0]
        # outputs = extract_cnn_feature(model, imgs)
        outputs = to_torch(outputs)
        # print(outputs.size())
        for fname, output, pid, img in zip(fnames, outputs, pids, imgs):
            features[fname] = output
            labels[fname] = pid
            images[fname] = img

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                #   'Data {:.3f} ({:.3f})\t'
                    .format(i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                        #   data_time.val, data_time.avg
                            ))
    if with_path:
        return features, labels, images
    else:
        return features, labels


class EvaluatorONNX(object):
    def __init__(self, sess, inname, outname):
        super(EvaluatorONNX, self).__init__()
        self.sess = sess
        self.inname = inname
        self.outname = outname

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features_onnx(
            self.sess, self.inname, self.outname, data_loader)
        distmat, query_features, gallery_features = pairwise_distance(
            features, query, gallery)
        results = evaluate_all(query_features, gallery_features,
                               distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(
            distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

class EvaluatorTRT(object):
    def __init__(self, trt_engine, target_dtype):
        super(EvaluatorTRT, self).__init__()
        # self.inputs = inputs
        # self.outputs = outputs
        # self.stream = stream
        # self.bindings = bindings
        # self.context = context
        self.trt_engine = trt_engine
        self.target_dtype = target_dtype

    def evaluate(self, data_loader, query, gallery, cmc_flag=False, rerank=False):
        features, _ = extract_features_trt(
            self.trt_engine, self.target_dtype, data_loader)

        distmat, query_features, gallery_features = pairwise_distance(
            features, query, gallery)
        results = evaluate_all(query_features, gallery_features,
                               distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat = re_ranking(
            distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
        return evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery, cmc_flag=cmc_flag)

def extract_features_trt(trt_egnine, target_dtype, data_loader, print_freq=50, with_path=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()
    images = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end)
        batch = to_numpy(imgs)
        batch = batch.astype(target_dtype)
        # print(trt_egnine.input_shape)
        # print(batch.shape)

        # def do_inference(inputs, outputs, context, bindings, stream): # result gets copied into output
            
        #     # Transfer data from CPU to the GPU.
        #     [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        #     # Run inference.
        #     context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        #     # Transfer predictions back from the GPU.
        #     [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        #     # gpu to cpu
        #     # Synchronize the stream
        #     stream.synchronize()
        #     # Return only the host outputs.
        #     return [out.host for out in outputs]
        
        # inputs[0].host = batch
        # cfx.push()
        # os = do_inference(inputs, outputs, context, bindings, stream)
        os = trt_egnine.inference(batch)
        # print(os)
        outs = to_torch(os)
        # print(outs.size())
        # print(trt_egnine.output_shape)
        # print(outs)
        # cfx.pop()

        for fname, output, pid, img in zip(fnames, outs, pids, imgs):
            features[fname] = output
            labels[fname] = pid
            images[fname] = img

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                #   'Data {:.3f} ({:.3f})\t'
                    .format(i + 1, len(data_loader),
                            batch_time.val, batch_time.avg,
                        #   data_time.val, data_time.avg
                            ))
    if with_path:
        return features, labels, images
    else:
        return features, labels
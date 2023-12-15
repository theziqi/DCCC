import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import init
from torch import nn, autograd
import collections

from .losses import CrossEntropyLabelSmooth, SoftEntropy, SoftEntropySmooth, MMDLoss, SoftmaxMSELoss


class HM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, indexes, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indexes)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indexes = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, indexes):
            ctx.features[y] = ctx.momentum * \
                ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def hm(inputs, indexes, features, momentum=0.5):
    return HM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * \
                ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hard(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(
                    ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * \
                ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_hard(inputs, indexes, features, momentum=0.5):
    return CM_Hard.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Avg(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(
                    ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * \
                ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None


def cm_avg(inputs, indexes, features, momentum=0.5):
    return CM_Avg.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(
                    ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            median = np.argmin(np.array(distances))
            ctx.features[index] = ctx.features[index] * \
                ctx.momentum + (1 - ctx.momentum) * features[median]
            ctx.features[index] /= ctx.features[index].norm()

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index+nums] = ctx.features[index+nums] * \
                ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index+nums] /= ctx.features[index+nums].norm()

        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_WgtMean(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum, tau_w):
        ctx.features = features
        ctx.momentum = momentum
        ctx.tau_w = tau_w
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(
                    ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance)

            # distances = F.normalize(torch.stack(distances, dim=0), dim=0)
            distances = torch.stack(distances, dim=0)
            w = F.softmax(- distances / ctx.tau_w, dim=0)
            features = torch.stack(features, dim=0)
            w_mean = w.unsqueeze(1).expand_as(features) * features
            w_mean = w_mean.sum(dim=0)
            # mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * \
                ctx.momentum + (1 - ctx.momentum) * w_mean
            ctx.features[index] /= ctx.features[index].norm()

        return grad_inputs, None, None, None, None


def cm_wgtmean(inputs, indexes, features, momentum=0.5, tau_w=0.09):
    return CM_WgtMean.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device), torch.Tensor([tau_w]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer(
            'features', torch.zeros(num_samples, num_features))
        self.register_buffer('labels', torch.zeros(num_samples).long())

    def forward(self, inputs, indexes):
        # inputs: B*2048, features: L*2048
        inputs = hm(inputs, indexes, self.features, self.momentum)
        inputs /= self.temp
        B = inputs.size(0)

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec)
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
            return (masked_exps/masked_sums)

        targets = self.labels[indexes].clone()
        labels = self.labels.clone()

        sim = torch.zeros(labels.max()+1, B).float().cuda()
        sim.index_add_(0, labels, inputs.t().contiguous())
        nums = torch.zeros(labels.max()+1, 1).float().cuda()
        nums.index_add_(0, labels, torch.ones(
            self.num_samples, 1).float().cuda())
        mask = (nums > 0).float()
        sim /= (mask*nums+(1-mask)).clone().expand_as(sim)
        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(
            sim.t().contiguous(), mask.t().contiguous())
        return F.nll_loss(torch.log(masked_sim+1e-6), targets)


class ClusterMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, smooth=0.1, mode='wgm', tau_w=0.09):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.smooth = smooth
        self.tau_w = tau_w

        self.ce_loss = nn.CrossEntropyLoss().cuda()
        # self.ce_loss = CrossEntropyLabelSmooth(self.num_samples, self.smooth, True).cuda()
        if smooth > 0:
            self.soft_ce_loss = SoftEntropySmooth(epsilon=self.smooth).cuda()
        else:
            self.soft_ce_loss = SoftEntropy().cuda()
        # self.sim_loss = SoftmaxMSELoss().cuda()
        self.register_buffer(
            'features', torch.zeros(num_samples, num_features))

    def forward(self, inputs1, inputs2, targets):
        inputs1 = F.normalize(inputs1, dim=1).cuda()
        inputs2 = F.normalize(inputs2, dim=1).cuda()

        inputs = inputs1
        if self.cm_type == 'wgm':
            outputs = cm_wgtmean(
                inputs, targets, self.features, self.momentum, self.tau_w)
        elif self.cm_type == 'hard':
            outputs = cm_hard(inputs, targets, self.features, self.momentum)
        elif self.cm_type == 'avg':
            outputs = cm_avg(inputs, targets, self.features, self.momentum)
        elif self.cm_type == 'cc':
            outputs = cm(inputs, targets, self.features, self.momentum)
        outputs /= self.temp

        regression = inputs2.mm(self.features.t())
        regression /= self.temp

        loss_c = self.ce_loss(outputs, targets)
        # loss_s = self.sim_loss(outputs.t().contiguous(),
        #                        regression.t().contiguous())
        # loss_s = self.soft_ce_loss(
        #     outputs.t().contiguous(), regression.t().contiguous())
        loss_s = self.soft_ce_loss(
            outputs, regression, targets)
        return loss_c, loss_s

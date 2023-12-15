from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils.meters import AverageMeter


class Trainer_UDA(object):
    def __init__(self, encoder, memory, source_classes):
        super(Trainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes

    def train(self, epoch, data_loader_source, data_loader_target,
              optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)
            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # forward
            f_out = self._forward(inputs)

            # de-arrange batch
            f_out = f_out.view(device_num, -1, f_out.size(-1))
            f_out_s, f_out_t = f_out.split(f_out.size(1)//2, dim=1)
            f_out_s, f_out_t = f_out_s.contiguous().view(-1, f_out.size(-1)
                                                         ), f_out_t.contiguous().view(-1, f_out.size(-1))

            # compute loss with the hybrid memory
            loss_s = self.memory(f_out_s, s_targets)
            loss_t = self.memory(f_out_t, t_indexes + self.source_classes)
            loss = loss_s + loss_t

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)


class Trainer_USL(object):
    def __init__(self, encoder, encoder_ema, memory=None, use_cm=False, alpha=0.999, soft_ce_weight=0.5):
        super(Trainer_USL, self).__init__()
        self.encoder = encoder
        self.encoder_ema = encoder_ema
        self.memory = memory
        self.use_cm = use_cm
        self.alpha = alpha
        self.soft_ce_weight = soft_ce_weight

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()
        self.encoder_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_c = AverageMeter()
        losses_s = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs1, inputs2, labels, indexes = self._parse_data(inputs)

            # forward
            f_out1 = self.encoder(inputs1)
            with torch.no_grad():
                f_out2 = self.encoder_ema(inputs2)

            if self.use_cm:
                # compute loss with the cluster memory
                loss_c, loss_s = self.memory(
                    f_out1, f_out2, labels)
            else:
                # compute loss with the hybrid memory
                loss = self.memory(f_out1, indexes)

            loss = (1 - self.soft_ce_weight) * \
                loss_c + self.soft_ce_weight * loss_s

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.soft_ce_weight != 0.0:
                self._update_ema_variables(
                    self.encoder, self.encoder_ema, self.alpha, epoch * len(data_loader) + i)

            losses_c.update(loss_c.item())
            losses_s.update(loss_s.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_c {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_c.val, losses_c.avg,
                              losses_s.val, losses_s.avg,))

    def _parse_data(self, inputs):
        imgs1, imgs2, _, pids, _, indexes = inputs
        return imgs1.cuda(), imgs2.cuda(), pids.cuda(), indexes.cuda()

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        # alpha = min(1 - 1 / (global_step + 1), alpha)
        if global_step >= 99:
            alpha = min(1 - 1 / (global_step - 98), alpha)
        # else:
        #     alpha = 0

        for (ema_name, ema_param), (model_name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from copy import deepcopy
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)


class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam) in enumerate(data_source):
            if (pid<0): continue
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)

class GroupSampler(Sampler):
    def __init__(self, dataset_labels, group_n=1, batch_size=None):
        label2data_idx = defaultdict(list)
        for i, (_, label, _) in enumerate(dataset_labels):
            label2data_idx[label].append(i)

        label2data = defaultdict(list)
        for label, data_idx in label2data_idx.items():
            if len(data_idx) > 1:
                label2data[label].extend(data_idx)
            else:
                label2data[-1].extend(data_idx)

        self.label2data_idx = label2data
        self.dataset_labels = dataset_labels
        self.group_n = group_n
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset_labels)

    def __iter__(self):
        data_idxes = []
        for label, data_idx in self.label2data_idx.items():
            if label != -1:
                data_idx = deepcopy(data_idx)
                random.shuffle(data_idx)
                data_idxes.extend([data_idx[i: i + self.group_n] for i in range(0, len(data_idx), self.group_n)])
                # data_idxes.append(data_idx)
        random.shuffle(data_idxes)
        ret = []
        for data_idx in data_idxes:
            ret.extend(data_idx)
        data_idx = deepcopy(self.label2data_idx[-1])
        random.shuffle(data_idx)
        ret.extend(data_idx)

        if self.batch_size is not None:
            batch_shuffle_ret = []
            tmp = [ret[i: i + self.batch_size] for i in range(0, len(ret), self.batch_size)]
            random.shuffle(tmp)
            for batch in tmp:
                batch_shuffle_ret.extend(batch)
            return iter(batch_shuffle_ret)
        else:
            return iter(ret)

    def __str__(self):
        return f"GroupSampler(num_instances={self.group_n}, batch_size={self.batch_size})"

    def __repr__(self):
        return self.__str__()
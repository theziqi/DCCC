from __future__ import print_function, absolute_import
import argparse
from ast import arg
import os.path as osp
import random
import numpy as np
import sys
import collections
import copy
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.hm import HybridMemory, ClusterMemory
from spcl.trainers import Trainer_USL
from spcl.evaluators import Evaluator, extract_features
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler, GroupSampler
from spcl.utils.data.preprocessor import Preprocessor
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance
from spcl.utils.lr_scheduler import WarmupMultiStepLR

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)

    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None

    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer, mutual=True),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True,
                          dropout=args.dropout, num_classes=0, pooling_type=args.pooling_type)
    model_ema = models.create(args.arch, num_features=args.features,
                              dropout=args.dropout, num_classes=0, pooling_type=args.pooling_type)

    # Load from checkpoint
    if args.resume:
        global start_epoch
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model, strip='module.')
        copy_state_dict(checkpoint['state_dict'], model_ema, strip='module.')
        start_epoch = checkpoint['epoch']

    # use CUDA
    model.cuda()
    model_ema.cuda()
    model = nn.DataParallel(model)
    model_ema = nn.DataParallel(model_ema)
    return model, model_ema


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(
        dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model, model_ema = create_model(args)

    use_cm = args.use_cm

    if not use_cm:
        # Create hybrid memory
        memory = HybridMemory(model.module.num_features, len(dataset.train),
                              temp=args.temp, momentum=args.momentum).cuda()

        # Initialize target-domain instance features
        print("==> Initialize instance features in the hybrid memory")
        cluster_loader = get_test_loader(dataset, args.height, args.width,
                                         args.batch_size, args.workers, testset=sorted(dataset.train))
        features, _ = extract_features(model, cluster_loader, print_freq=50)
        features = torch.cat([features[f].unsqueeze(0)
                              for f, _, _ in sorted(dataset.train)], 0)
        memory.features = F.normalize(features, dim=1).cuda()
        del cluster_loader, features

    # Evaluator
    evaluator = Evaluator(model)
    evaluator_ema = Evaluator(model_ema)

    # Optimizer
    params = [{"params": [value]}
              for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(
        params, lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == 'warmup':
        lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=1, warmup_factor=0.1,
                                         warmup_iters=args.warmup_step)
    elif args.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=0.1)

    # Trainer
    trainer = Trainer_USL(model, model_ema, use_cm=use_cm,
                          soft_ce_weight=args.soft_ce_weight)

    for epoch in range(start_epoch, args.epochs):
        # Calculate distance
        print('==> Create pseudo labels for unlabeled data with self-paced policy')
        if use_cm:
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            features, _ = extract_features(
                model, cluster_loader, print_freq=50)
            features = torch.cat([features[f].unsqueeze(0)
                                  for f, _, _ in sorted(dataset.train)], 0)
            rerank_dist = compute_jaccard_distance(
                features, k1=args.k1, k2=args.k2)
        else:
            features = memory.features.clone()
            rerank_dist = compute_jaccard_distance(
                features, k1=args.k1, k2=args.k2)
            del features

        if epoch == start_epoch:
            # started eps value
            eps = args.eps

        if args.eps_scheduler == 'fix':
            eps = eps
        elif args.eps_scheduler == 'step' and epoch % args.eps_step == 0 and eps > 0.399:
            eps = eps - 0.006 * args.eps_step
        elif args.eps_scheduler == 'linear' and eps > 0.399:
            eps = eps - 0.005
        elif args.eps_scheduler == 'expo' and eps > 0.399:
            eps = eps * 0.99
        elif args.eps_scheduler == 'log':
            eps = args.eps - args.eps * \
                np.log(epoch * (np.e - 1) / args.epochs + 1) / 2

        cluster = DBSCAN(eps=eps, min_samples=4,
                         metric='precomputed', n_jobs=-1)

        print('Clustering criterion: eps: {:.3f}'.format(eps))

        # select & cluster images as training set of this epochs
        pseudo_labels = cluster.fit_predict(rerank_dist)
        num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
        num_cluster = num_ids
        # recluster = KMeans(n_clusters=num_cluster)
        # re_pseudo_labels = recluster.fit_predict(features)

        # if (epoch==start_epoch or (epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
        #     tsne = TSNE(n_components=2, init='pca')
        #     features_tsne = tsne.fit_transform(features)
        #     plt.figure()
        #     plt.rcParams['figure.figsize'] = 20, 20

        #     plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=pseudo_labels)
        #     plt.title('DBSCAN cluster result')
        #     plt.savefig(osp.join(args.logs_dir, 'dbscan_result_%s.png' % epoch))
        #     plt.cla()

        #     plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=re_pseudo_labels)
        #     plt.title('KMeans cluster result')
        #     plt.savefig(osp.join(args.logs_dir, 'kmeans_result_%s.png' % epoch))

        if use_cm:
            @torch.no_grad()
            def generate_cluster_features(labels, features):
                centers = collections.defaultdict(list)
                outliers = []
                for i, label in enumerate(labels):
                    if label == -1:
                        outliers.append(features[i])
                    else:
                        centers[labels[i]].append(features[i])
                print("outliers: ", len(outliers))
                centers = [
                    torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
                ]
                print("centers: ", len(centers))
                # centers.extend(outliers)
                # print("centers: ", len(centers))
                centers = torch.stack(centers, dim=0)
                return centers

            cluster_features = generate_cluster_features(
                pseudo_labels, features)
            del cluster_loader, features

            # Create memory bank
            memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                                   momentum=args.momentum, smooth=args.smooth, tau_w=args.tau_w, mode=args.cm_type).cuda()
            memory.features = F.normalize(cluster_features, dim=1).cuda()
            # memory.features = F.normalize(cluster_features.repeat(2, 1), dim=1).cuda()
        else:
            # generate new dataset and calculate cluster centers
            def generate_pseudo_labels(cluster_id, num):
                labels = []
                outliers = 0
                for i, ((fname, _, cid), id) in enumerate(zip(sorted(dataset.train), cluster_id)):
                    if id != -1:
                        labels.append(id)
                    else:
                        labels.append(num + outliers)
                        outliers += 1
                return torch.Tensor(labels).long()

            pseudo_labels = generate_pseudo_labels(pseudo_labels, num_ids)

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if use_cm:
                if label != -1:
                    pseudo_labeled_dataset.append((fname, label.item(), cid))
            else:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        if use_cm:
            print('==> Statistics for epoch {}: {} clusters'.format(
                epoch, num_cluster))
        else:
            # statistics of clusters and un-clustered instances
            index2label = collections.defaultdict(int)
            for label in pseudo_labels:
                index2label[label.item()] += 1
            index2label = np.fromiter(index2label.values(), dtype=float)
            num_clusters = (index2label > 1).sum()
            num_unclusters = (index2label == 1).sum()
            print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(
                epoch, num_clusters, num_unclusters))

            memory.labels = pseudo_labels.cuda()

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset)

        train_loader.new_epoch()

        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader))

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP1 = evaluator.evaluate(
                test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            mAP2 = evaluator_ema.evaluate(
                test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP1 > best_mAP) or (mAP2 > best_mAP)
            best_mAP = max(mAP1, mAP2, best_mAP)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, (is_best and (mAP1 <= mAP2)), fpath=osp.join(args.logs_dir, 'checkpoint_ema.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  model_ema mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP1, mAP2, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query,
                       dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--use-cm', action="store_true",
                        help="use cluster memory or not")
    parser.add_argument('--cm-type', type=str, default="wgm")
    # cluster
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-scheduler', type=str, default="fix")
    parser.add_argument('--eps-step', type=int, default=5)
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--smooth', type=float,
                        default=0.1, help="label smoothing")
    parser.add_argument('--momentum', type=float, default=0.2,
                        help="update momentum for memory")
    parser.add_argument('--pooling-type', type=str, default='avg')
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--lr-scheduler', type=str, default='warmup')
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[],
                        help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    parser.add_argument('--soft-ce-weight', type=float, default=0.3,
                        help="weight for soft cross entropy loss")
    parser.add_argument('--tau-w', type=float, default=0.09,
                        help="hyperparameter for DyCL")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--resume', type=str, metavar='PATH',
                        default='')
    main()

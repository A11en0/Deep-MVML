# -*- coding: UTF-8 -*-
import os
import random
from functools import reduce

import torch
import numpy as np
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from config import *
from layers.net_H import FusionNet, UncertaintyNet
from model import Network
from models_bak import TMC
from train import test, Trainer
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data, init_random_seed


def run(device, args, save_dir, file_name):
    print('*' * 30)
    print('dataset:\t', args.DATA_SET_NAME)
    print('optimizer:\t Adam')
    print('*' * 30)

    save_name = save_dir + file_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(save_name):
        return

    features, labels, idx_list = load_mat_data(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), True)

    # writer = SummaryWriter()
    fold_list, metrics_results = [], []
    rets = np.zeros((Fold_numbers, 11))  # 11 metrics
    for fold in range(Fold_numbers):
        TEST_SPLIT_INDEX = fold
        print('-' * 50 + '\n' + 'Fold: %s' % fold)
        train_features, train_labels, train_partial_labels, test_features, test_labels = split_data_set_by_idx(
            features, labels, idx_list, TEST_SPLIT_INDEX, args)
        
        # load views features and labels
        views_dataset = ViewsDataset(train_features, train_partial_labels, device)
        views_data_loader = DataLoader(views_dataset, batch_size=args.batch_size, shuffle=True,
                                       num_workers=args.workers, pin_memory=True,
                                       drop_last=True)

        view_code_list = list(train_features.keys())
        view_feature_dim_list = [train_features[code].shape[1] for code in view_code_list]
        # feature_dim = reduce(lambda x, y: x + y, view_feature_dim_list)

        num_view = len(view_code_list)
        class_num = train_labels.shape[1]
        input_size = view_feature_dim_list
        latent_dim = args.latent_dim
        high_feature_dim = args.high_feature_dim
        embedding_dim = args.embedding_dim
        common_embedding_dim = args.common_embedding_dim

        # load model
        # model = Network(view_num, input_size, features_dim, high_feature_dim, class_num, device).to(device)

        model = Network(num_view, input_size, latent_dim, high_feature_dim,
                 embedding_dim, common_embedding_dim, class_num, device).to(device)

        # training
        trainer = Trainer(model, args, device)
        loss_list = trainer.fit(views_data_loader, train_features, train_partial_labels, test_features, test_labels,
                                class_num, fold)

        fold_list.append(loss_list)

        metrics_results, _ = test(model, test_features, test_labels, device, is_eval=True)

        for i, key in enumerate(metrics_results):
            rets[fold][i] = metrics_results[key]

    rets = np.zeros((Fold_numbers, args.epochs, 11))  # 11 metrics
    for fold, li_fold in enumerate(fold_list):
        for i, epoch in enumerate(li_fold):
            for j, key in enumerate(li_fold[i]):
                rets[fold][i][j] = li_fold[i][key]

    means = np.mean(rets, axis=0)
    stds = np.std(rets, axis=0)
    _index = np.argsort(means[:, 0])
    means = means[_index]
    stds = stds[_index]

    print("\n------------summary--------------")
    print("Best Epoch: ", _index[0])
    metrics_list = list(metrics_results.keys())

    with open(save_name, "w") as f:
        for i, _ in enumerate(means[0, :]):
            print("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics_list[i], means=means[0, :][i],
                                                           std=stds[0, :][i]))
            f.write("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics_list[i], means=means[0, :][i],
                                                           std=stds[0, :][i]))
            f.write("\n")

    # print("\n------------summary--------------")
    # for i in range(means.shape[0]):
    #     for j in range(means.shape[1]):
    #         mu = means[i][j]
    #         std = stds[i][j]
    #         print(f"{mu:.4f} +- {std:.4f} \t")
    #     print()


if __name__ == '__main__':
    args = Args()

    # setting random seeds
    init_random_seed(args.seed)

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # device = 'cpu'

    # lrs = [1e-2, 5e-2, 2e-3, 6e-3, 5e-3, 1e-4, 5e-4, 1e-5, 1e-6]
    lrs = [1e-3]

    # noise_rates = [0.3, 0.5, 0.7]
    noise_rates = [0.0]

    datanames = ['Emotions', 'Scene', 'Yeast']
    # datanames = ['Yeast']
    # datanames = ['Scene']
    # datanames = ['Pascal']
    # datanames = ['Iaprtc12']

    # datanames = ['Corel5k']
    # datanames = ['Mirflickr']
    # datanames = ['Espgame']

    param_grid = {
        'latent_dim': [64, 128, 256, 512],
        'high_feature_dim': [64, 128, 256, 512],
        'embedding_dim': [64, 128, 256, 512],
        'common_embedding_dim': [64, 128, 256, 512],
    }

    MAX_EVALS = 15
    best_score = 0
    best_hyperparams = {}
    
    for dataname in datanames:
        for p in noise_rates:
            for lr in lrs:
                for i in range(MAX_EVALS):
                    random.seed(i)  # 设置随机种子，每次搜索设置不同的种子，若种子固定，那每次选取的超参都是一样的
                    args.DATA_SET_NAME = dataname
                    args.noise_rate = p
                    args.lr = lr

                    # 随机搜索
                    random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}

                    args.latent_dim = random_params['latent_dim']
                    args.high_feature_dim = random_params['high_feature_dim']
                    args.embedding_dim = random_params['embedding_dim']
                    args.common_embedding_dim = random_params['common_embedding_dim']

                    save_dir = f'results/{dataname}/'
                    save_name = f'{args.DATA_SET_NAME}-lr{args.lr}-p{args.noise_rate}-r{args.noise_num}-lat{args.latent_dim}-hdim{args.high_feature_dim}-emd{args.embedding_dim}-comm{args.embedding_dim}.txt'
                    run(device, args, save_dir, save_name)




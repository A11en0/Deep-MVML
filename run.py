# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from config import *
from train import train, test
from layer.view_block import ViewBlock
from models import Model, ModelEmbedding
from train import test, Trainer
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data, init_random_seed


def run(device, args, save_dir, file_name):
    print('*' * 30)
    print('seed:\t', args.seed)
    print('dataset:\t', args.DATA_SET_NAME)
    # print('latent dim:\t', args.latent_dim)
    # print('high feature dim:\t', args.high_feature_dim)
    # print('embedding dim:\t', args.embedding_dim)
    # print('coef_cl:\t', args.coef_cl)
    # print('coef_ml:\t', args.coef_ml)
    print('optimizer:\t Adam')
    print('*' * 30)

    # setting random seeds
    init_random_seed(args.seed)

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
        views_data_loader = DataLoader(views_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        # instantiation View Model
        view_code_list = list(train_features.keys())
        view_feature_nums_list = [train_features[code].shape[1] for code in view_code_list]
        view_blocks = [ViewBlock(view_code_list[i], view_feature_nums_list[i], args.common_feature_dim)
                       for i in range(len(view_code_list))]

        label_nums = train_labels.shape[1]

        # load model
        if args.le:
            model = ModelEmbedding(view_blocks, args.common_feature_dim, label_nums, device, args).to(device)
        else:
            model = Model(view_blocks, args.common_feature_dim, label_nums, device, args).to(device)

        print(model)

        # training
        loss_list = train(model, device, views_data_loader, args, loss_coefficient,
                     train_features, train_partial_labels, test_features, test_labels, fold=1)
        fold_list.append(loss_list)

        metrics_results, _ = test(model, test_features, test_labels, device, is_eval=True, args=args)

        for i, key in enumerate(metrics_results):
            rets[fold][i] = metrics_results[key]

    rets = np.zeros((Fold_numbers, args.epoch, 11))  # 11 metrics
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
        f.write(str(_index[0]))


if __name__ == '__main__':
    args = Args()

    # setting random seeds
    init_random_seed(args.seed)

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # device = 'cpu'

    # tune parameters
    # datanames = ['yeast.mat', 'scene.mat', 'Pascal.mat',  'emotions.mat', 'Corel5k.mat', 'Mirflickr.mat', 'Espgame.mat']
    # datanames = ['yeast', 'scene', 'emotions', ]  # epoch25 / latent64 / lr 1e-3
        # self.common_feature_dim = 256
        # self.latent_dim = 6  # 小数据集 64
        # self.embedding_dim = 512  # 极其重要
        # self.keep_prob = 0.5
        # self.scale_coeff = 1.0

    datanames = ['Emotions']
    # datanames = ['Pascal']
    # datanames = ['Corel5k']  # bug
    # datanames = ['Espgame']
    # datanames = ['Mirflickr']
    # datanames = ['Espgame']
    # datanames = ['Iaprtc12']

    # lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    lrs = [1e-3]
    etas = [5e-3]

    # noise_rates = [0.3, 0.5, 0.7]
    noise_rates = [0.0]

    # for kl_coef in kl_coef_lists:
    for eta in etas:
        for dataname in datanames:
            for lr in lrs:
                for p in noise_rates:
                    args.lr = lr
                    args.DATA_SET_NAME = dataname
                    args.eta = eta
                    args.noise_rate = p
                    # args.coef_kl = kl_coef
                    save_dir = f'results/{args.DATA_SET_NAME}/'
                    file_name = f'{args.DATA_SET_NAME}_bs{args.batch_size}_ml{args.coef_ml}_' \
                                f'kl{args.coef_kl}_epoch{args.epoch}_lr{args.lr}_com{args.common_feature_dim}_' \
                                f'lat{args.latent_dim}_p{args.noise_rate}.txt'
                    run(device, args, save_dir, file_name)




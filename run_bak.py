# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from config import *
from layer.view_block import ViewBlock
from models import Model
from train import train, test
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data, init_random_seed


def run(args, save_dir, file_name):
    print('*' * 30)
    print('ML Loss coefficient:\t', args.coef_ml)
    print('KL loss coefficient:\t', args.coef_kl)
    print('dataset:\t', args.DATA_SET_NAME)
    print('common feature dims:\t', args.common_feature_dim)
    print('latent dims:\t', args.latent_dim)
    print('optimizer:\t Adam')
    print('*' * 30)

    save_name = save_dir + file_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    features, labels, idx_list = load_mat_data(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), True)

    fold_list = []
    rets = np.zeros((Fold_numbers, 11))
    for fold in range(Fold_numbers):
        # if fold == 1:
        #     break

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

        # load model
        label_nums = train_labels.shape[1]
        num_view = len(view_code_list)

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
    _index = np.argsort(means[:, 6])

    # times 5 fold
    # means = means[_index]*5
    # stds = stds[_index]*5

    mean_choose = means[_index][-1, :]
    std_choose = means[_index][-1, :]

    print("\n------------summary--------------")
    print("Best Epoch: ", _index[0])
    metrics_list = list(metrics_results.keys())

    with open(save_name, "w") as f:
        for i, _ in enumerate(mean_choose):
            print("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics_list[i], means=mean_choose[i],
                                                           std=std_choose[i]))
            f.write("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics_list[i], means=mean_choose[i],
                                                           std=std_choose[i]))
            f.write("\n")
        f.write(str(_index[0]))

    # Draw figures
    # if args.is_test_in_train:
    #     metrics_keys = ["ML_loss", "Hamming", "Average", "Ranking", "Coverage", "MacroF1", "MicroF1", "OneError"]
    #     up_keys = ["Average", "MacroF1", "MicroF1"]
    #     for epoch in range(len(fold_list[0])):
    #         for key in metrics_keys:
    #             matrics_vals = 0.0
    #             for fold in range(len(fold_list)):
    #                 matrics_vals += fold_list[fold][epoch][key]
    #             matrics_vals = matrics_vals / len(fold_list)
    #             if key in up_keys:
    #                 writer.add_scalar(f"UP/{key}", matrics_vals, epoch)  # log
    #             else:
    #                 writer.add_scalar(f"Down/{key}", matrics_vals, epoch)  # log

    # print("\n------------summary--------------")
    # means = np.mean(rets, axis=0)*5
    # stds = np.std(rets, axis=0)*5
    #
    # metrics_list = list(metrics_results.keys())
    # with open(save_name, "w") as f:
    #     for i, _ in enumerate(means):
    #         print("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics_list[i], means=means[i], std=stds[i]))
    #         f.write("{metric}\t{means:.4f}±{std:.4f}".format(metric=metrics_list[i], means=means[i], std=stds[i]))
    #         f.write("\n")


if __name__ == '__main__':
    args = Args()

    # setting random seeds
    init_random_seed(args.seed)

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # device = 'cpu'

    # tune parameters
    datanames = ['Yeast', 'Scene', 'Pascal',  'Emotions', 'Iaprtc12', 'Corel5k', 'Mirflickr', 'Espgame']
    # datanames = ['yeast', 'scene', 'emotions', ]  # epoch25 / latent64 / lr 1e-3

    datanames = ['Emotions']
    # datanames = ['Scene']
    # datanames = ['Yeast']
    # datanames = ['Pascal']
    # datanames = ['Corel5k']  # bug
    # datanames = ['Mirflickr']
    # datanames = ['Espgame']
    # datanames = ['Iaprtc12']

    # lrs = [1e-2, 1e-3, 1e-4, 1e-5]
    # lrs = [1e-3]
    lrs = [1e-3]

    # etas = [1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 5e-5]
    etas = [1e-3]
    # zetas = [1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 5e-5]
    # alphas = [10, 5, 2, 1, 1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 5e-5]

    noise_rates = [0.3, 0.5, 0.7]
    # noise_rates = [0.7]

    # for kl_coef in kl_coef_lists:
    for eta in etas:
        for dataname in datanames:
            for lr in lrs:
                for p in noise_rates:
                    args.lr = lr
                    args.DATA_SET_NAME = dataname
                    args.noise_rate = p

                    save_dir = f'results/{args.DATA_SET_NAME}/'
                    file_name = f'{args.DATA_SET_NAME}_bs{args.batch_size}_ml{args.coef_ml}_' \
                                f'kl{args.coef_kl}_epoch{args.epoch}_lr{args.lr}_com{args.common_feature_dim}_' \
                                f'lat{args.latent_dim}_p{args.noise_rate}-eta{args.eta}-zeta{args.zeta}-alpha{args.alpha}.txt'
                    run(args, save_dir, file_name)




# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *
from layer.view_block import ViewBlock, DecoderBlock
from models import Model, Model_AE, ModelEmbedding
from train import train, test
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data_v1, init_random_seed


def run(args, save_name):
    features, labels, idx_list = load_mat_data_v1(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME), True)

    writer = SummaryWriter()

    label_num = np.size(labels, 1)
    metrics_result_list = []
    avg_metrics = {}
    fold_list = []
    for fold in range(Fold_numbers):
        TEST_SPLIT_INDEX = fold
        print('-' * 50 + '\n' + 'Fold: %s' % fold)
        train_features, train_labels, train_partial_labels, test_features, test_labels = split_data_set_by_idx(
            features, labels, idx_list, TEST_SPLIT_INDEX, partial_rate=args.partial_rate)

        # load views features and labels
        views_dataset = ViewsDataset(train_features, train_partial_labels, device)
        views_data_loader = DataLoader(views_dataset, batch_size=256, shuffle=True, num_workers=0)

        # instantiation View Model
        view_code_list = list(train_features.keys())
        view_feature_nums_list = [train_features[code].shape[1] for code in view_code_list]
        view_blocks = [ViewBlock(view_code_list[i], view_feature_nums_list[i], args.common_feature_dim)
                       for i in range(len(view_code_list))]

        decoder_blocks = [DecoderBlock(view_code_list[i], 128, view_feature_nums_list[i])
                          for i in range(len(view_code_list))]

        # load model
        label_nums = train_labels.shape[1]
        if args.ae:
            model = Model_AE(view_blocks, decoder_blocks, args.common_feature_dim, label_nums, device, args).to(device)
        elif args.le:
            model = ModelEmbedding(view_blocks, decoder_blocks, args.common_feature_dim, label_nums, device, args).to(device)
        else:
            model = Model(view_blocks, args.common_feature_dim, label_nums, device, args).to(device)

        # training
        loss_list = train(model, device, views_data_loader, args, loss_coefficient,
                     train_features, train_partial_labels, test_features, test_labels, WEIGHT_DECAY, fold=1)
        fold_list.append(loss_list)

        metrics_results, _ = test(model, test_features, test_labels, device, is_eval=True, args=args)
        # pprint(metrics_results)

        # show results
        for m in metrics_results:
            if m[0] in avg_metrics:
                avg_metrics[m[0]] += m[1]
            else:
                avg_metrics[m[0]] = m[1]

        metrics_result_list.append(metrics_results)

    # Draw figures
    metrics_keys = ["ML_loss", "Hamming", "Average", "Ranking", "Coverage", "MacroF1", "MicroF1", "OneError"]
    up_keys = ["Average", "MacroF1", "MicroF1"]
    for epoch in range(len(fold_list[0])):
        for key in metrics_keys:
            matrics_vals = 0.0
            for fold in range(len(fold_list)):
                matrics_vals += fold_list[fold][epoch][key]
            matrics_vals = matrics_vals / len(fold_list)
            if key in up_keys:
                writer.add_scalar(f"UP/{key}", matrics_vals, epoch)  # log
            else:
                writer.add_scalar(f"Down/{key}", matrics_vals, epoch)  # log

    print("\n------------summary--------------")
    if not os.path.exists('results'):
        os.mkdir('results')

    with open(save_name, "w") as f:
        for k, v in avg_metrics.items():
            print("{metric}:\t{value}".format(metric=k, value=v / Fold_numbers))
            f.write("{metric}:\t{value}".format(metric=k, value=v / Fold_numbers))
            f.write("\n")

    writer.flush()
    writer.close()

def boot(args, save_name):
    print('*' * 30)
    print('ML Loss coefficient:\t', args.coef_ml)
    print('KL loss coefficient:\t', args.coef_kl)
    print('dataset:\t', args.DATA_SET_NAME)
    print('common feature dims:\t', args.common_feature_dim)
    print('optimizer:\t Adam')
    print('*' * 30)

    if not os.path.exists(save_name):
        run(args, save_name)


if __name__ == '__main__':
    args = Args()

    # setting random seeds
    init_random_seed(args.seed)

    device = torch.device("cuda") if args.cuda else torch.device("cpu")

    kl_coef_lists = [0.0]
    # kl_coef_lists += np.arange(1, 2, 0.1).tolist()
    # kl_coef_lists += np.arange(2, 5, 0.5).tolist()
    # datanames = ['Pascal.mat', 'scene.mat', 'yeast.mat',  'Corel5k.mat', 'emotions.mat', 'Mirflickr.mat', 'Espgame.mat',]

    datanames = ['Corel5k.mat']
    # kl_coef_lists = [0.0]
    # for kl_coef in kl_coef_lists:
    #     for ml_coef in ml_coef_lists:
    #         args.coef_kl = kl_coef
    #         args.coef_ml = ml_coef
    #         boot(args)

    # common_feature_dims = [512]
    # # common_feature_dims = [64, 128, 256, 512]
    # latents_dims = [64]
    # # latents_dims = [64, 128, 256, 512]
    # for lr in [1e-3, ]:
    #     # for kl_coef in kl_coef_lists:
    #     for latents_dim in latents_dims:
    #         for common_feature_dim in common_feature_dims:
    #             args.lr = lr
    #             args.common_feature_dim = common_feature_dim
    #             args.latent_dim = latents_dim
    #             args.coef_kl = kl_coef
    #             save_name = f'results/{args.DATA_SET_NAME}_{args.coef_ml}_{args.coef_kl}_{args.lr}_{args.common_feature_dim}_{args.latent_dim}.txt'
    #             boot(args, save_name)

    for kl_coef in kl_coef_lists:
        for dataname in datanames:
            args.DATA_SET_NAME = dataname
            args.coef_kl = kl_coef
            save_name = f'results/{args.DATA_SET_NAME}_{args.coef_ml}_{args.epoch}_{args.coef_kl}_{args.lr}_{args.common_feature_dim}_{args.latent_dim}.txt'
            boot(args, save_name)
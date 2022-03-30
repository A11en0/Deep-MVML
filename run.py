# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import *
from layer.view_block import ViewBlock, DecoderBlock
from models import Model, ModelEmbedding
from train import train, test
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data_v1, init_random_seed


def run(args, save_name):
    features, labels, idx_list = load_mat_data_v1(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), True)

    writer = SummaryWriter()

    label_num = np.size(labels, 1)
    metrics_result_list = []
    avg_metrics = {}
    fold_list = []
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
    with open(save_name, "w") as f:
        for k, v in avg_metrics.items():
            print("{metric}:\t{value}".format(metric=k, value=v / Fold_numbers))
            f.write("{metric}:\t{value}".format(metric=k, value=v / Fold_numbers))
            f.write("\n")

    writer.flush()
    writer.close()

def boot(args, save_dir, file_name):
    print('*' * 30)
    print('ML Loss coefficient:\t', args.coef_ml)
    print('KL loss coefficient:\t', args.coef_kl)
    print('dataset:\t', args.DATA_SET_NAME)
    print('common feature dims:\t', args.common_feature_dim)
    print('optimizer:\t Adam')
    print('*' * 30)

    save_name = save_dir + file_name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_name):
        run(args, save_name)


if __name__ == '__main__':
    args = Args()

    # setting random seeds
    init_random_seed(args.seed)

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # device = 'cpu'

    # tune parameters
    # datanames = ['yeast.mat', 'scene.mat', 'Pascal.mat',  'emotions.mat', 'Corel5k.mat', 'Mirflickr.mat', 'Espgame.mat']
    datanames = ['yeast', 'scene', 'emotions', ]  # 'Pascal']
    lrs = [1e-3]
    etas = [5e-3]
    noise_rates = [0.3, 0.5, 0.7]
    # noise_rates = [0.0]

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
                    boot(args, save_dir, file_name)

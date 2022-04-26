# -*- coding: UTF-8 -*-
import os
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from config import *
from model import Network
from train import test, Trainer
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data, init_random_seed


def run(device, args, save_dir, file_name):
    print('*' * 30)
    print('seed:\t', args.seed)
    print('dataset:\t', args.DATA_SET_NAME)
    print('latent dim:\t', args.latent_dim)
    print('high feature dim:\t', args.high_feature_dim)
    print('embedding dim:\t', args.embedding_dim)
    print('coef_cl:\t', args.coef_cl)
    print('coef_ml:\t', args.coef_ml)
    print('optimizer:\t Adam')
    print('*' * 30)

    # setting random seeds
    init_random_seed(args.seed)

    features, labels, idx_list = load_mat_data(os.path.join(args.DATA_ROOT, args.DATA_SET_NAME + '.mat'), True)

    # writer = SummaryWriter()
    fold_list, metrics_results = [], []

    train_features, train_labels, train_partial_labels, test_features, test_labels = split_data_set_by_idx(
        features, labels, idx_list, TEST_SPLIT_INDEX, args)

    for v in train_features:
        train_features[v] = torch.cat((train_features[v], test_features[v]), dim=0)

    # load views features and labels
    views_dataset = ViewsDataset(train_features, train_partial_labels, device)
    views_data_loader = DataLoader(views_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True,
                                   drop_last=True)

    view_code_list = list(train_features.keys())
    view_feature_dim_list = [train_features[code].shape[1] for code in view_code_list]

    num_view = len(view_code_list)
    class_num = train_labels.shape[1]
    input_size = view_feature_dim_list

    model = Network(num_view, input_size, args.latent_dim, args.high_feature_dim,
                    args.embedding_dim, class_num, device).to(device)

    print(model)

    # training
    trainer = Trainer(model, None, args, device)
    loss_list = trainer.pretrain(views_data_loader, class_num)


if __name__ == '__main__':
    args = Args_Pre()

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # device = 'cpu'

    # lrs = [1e-2, 5e-2, 2e-3, 6e-3, 5e-3, 1e-4, 5e-4, 1e-5, 1e-6]
    # lrs = [5e-4, 5e-6, 6e-7, 3e-8, 6e-4, 6e-4, 3e-4, 5e-3, 5e-5]
    # lrs = [6e-4]

    # noise_rates = [0.0, 0.3, 0.5, 0.7]
    # noise_rates = [0.3]

    # datanames = ['Emotions', 'Scene', 'Yeast', 'Pascal', 'Iaprtc12', 'Corel5k', 'Mirflickr', 'Espgame']
    # label_nums = [6, 6, 14, 20, 291, 260, 38, 268]

    # datanames = ['Mirflickr', 'Emotions', 'Yeast', 'Scene', 'Pascal']
    # label_nums = [6]

    # datanames = ['Iaprtc12', 'Corel5k', 'Espgame']
    # label_nums = [300]

    # datanames = ['Emotions']
    # datanames = ['Scene']
    # datanames = ['Yeast']
    # datanames = ['Pascal']
    datanames = ['Iaprtc12']
    # datanames = ['Corel5k']
    # datanames = ['Mirflickr']
    # datanames = ['Espgame']

    # param_grid = {
    #     'latent_dim': [6],
    #     'high_feature_dim': [256],
    #     'embedding_dim': [256],
    # }

    # MAX_EVALS = 1
    # best_score = 0
    # best_hyperparams = {}

    for i, dataname in enumerate(datanames):
        args.DATA_SET_NAME = dataname
        run(device, args, None, None)




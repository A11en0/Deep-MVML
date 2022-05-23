# -*- coding: UTF-8 -*-
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import *
from models.Networks import Network
from train import test, Trainer
from utils.common_tools import split_data_set_by_idx, ViewsDataset, load_mat_data, init_random_seed, init_network
from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir="runs/")


def run(device, args, save_dir, file_name):
    print('*' * 30)
    print('seed:\t', args.seed)
    print('dataset:\t', args.DATA_SET_NAME)
    # print('latent dim:\t', args.latent_dim)
    print('high feature dim:\t', args.high_feature_dim)
    print('embedding dim:\t', args.embedding_dim)
    print('coef_cl:\t', args.coef_cl)
    print('coef_ml:\t', args.coef_ml)
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

    fold_list, metrics_results = [], []
    rets = np.zeros((Fold_numbers, 11))  # 11 metrics
    for fold in range(Fold_numbers):

        writer = SummaryWriter(log_dir=f"runs/{file_name}/{fold}")

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

        model = Network(num_view, input_size, args.high_feature_dim, args.latent_dim,
                        args.embedding_dim, class_num, args, device).to(device)

        # init_network(model)
        # print(model)

        # init_graph = {}
        # for i in range(len(view_code_list)):
        #     init_graph[i] = torch.zeros(args.batch_size, view_feature_dim_list[i]).to(device)

        # init_labels = torch.zeros(args.batch_size, class_num)
        # writer.add_graph(model, [init_graph, init_labels])

        # Parallel
        # model = torch.nn.DataParallel(model)

        # training
        trainer = Trainer(model, writer, args, device)
        # trainer.pretrain(views_data_loader, class_num)
        loss_list = trainer.fit(views_data_loader, train_features, train_labels, test_features, test_labels,
                                class_num, fold)

        fold_list.append(loss_list)

        metrics_results, _ = test(model, test_features, test_labels, device, is_eval=True, args=args)

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
    # means = means[_index]
    # stds = stds[_index]

    mean_choose = means[_index][0, :]
    std_choose = stds[_index][0, :]

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


if __name__ == '__main__':
    args = Args()

    device = torch.device("cuda") if args.cuda else torch.device("cpu")
    # device = 'cpu'

    # lrs = [1e-2, 5e-2, 2e-3, 6e-3, 5e-3, 1e-4, 5e-4, 1e-5, 1e-6]
    # lrs = [5e-4, 5e-6, 6e-7, 3e-8, 6e-4, 6e-4, 3e-4, 5e-3, 5e-5]
    # lrs = [1e-3]

    # noise_rates = [0.3, 0.5, 0.7]
    noise_rates = [0.0]

    # datanames = ['Emotions', 'Scene', 'Yeast', 'Pascal', 'Iaprtc12', 'Corel5k', 'Mirflickr', 'Espgame']
    # label_nums = [6, 6, 14, 20, 291, 260, 38, 268]

    # datanames = ['Yeast', 'Corel5k']
    # datanames = ['Pascal', 'Mirflickr']
    # datanames = ['Iaprtc12', 'Emotions']
    # datanames = ['Scene', 'Espgame']

    # datanames = ['Emotions']
    datanames = ['Scene']
    # datanames = ['Yeast']
    # datanames = ['Pascal']
    # datanames = ['Iaprtc12']
    # datanames = ['Corel5k']
    # datanames = ['Mirflickr']
    # datanames = ['Espgame']

    param_grid = {
        # 'latent_dim': [64, 128, 256, 512],
        'high_feature_dim': [64, 128, 256, 512],
        'embedding_dim': [64, 128, 256, 512],
    }

    # etas = [0.1, 1e-2, 1e-3, 1e-4]
    # zetas = [0.1, 1e-2, 1e-3]
    # alphas = [0.1, 1, 10]

    for i, dataname in enumerate(datanames):
        for p in noise_rates:
            # for lr in lrs:

                # for eta in etas:
                #     for zeta in zetas:
                #         for alpha in alphas:
                # for coef_kl in np.arange(0, 1, 0.2):
                #     for coef_cl in np.arange(0, 1, 0.2):
                #         for coef_mi in np.arange(0, 1, 0.2):

                        # for emb in param_grid['embedding_dim']:
                        #     for hi in param_grid['high_feature_dim']:

                            args.DATA_SET_NAME = dataname
                            args.noise_rate = p
                            # args.lr = lr
                            # args.coef_cl = coef_cl
                            # args.coef_mi = coef_mi

                            save_dir = f'results/{args.DATA_SET_NAME}/'
                            save_name = f'{args.DATA_SET_NAME}-lr{args.lr}-epochs{args.epochs}-' \
                                        f'hdim{args.high_feature_dim}-emd{args.embedding_dim}-' \
                                        f'coef_ml-{args.coef_ml}-coef_cl{args.coef_cl}-coef_rec{args.coef_rec}-marginloss.txt'

                            run(device, args, save_dir, save_name)

                        # args.embedding_dim = emb
                        # args.high_feature_dim = hi
                        # args.coef_cl = coef_cl

                    # args.latent_dim = label_nums[i]
                    #     # Grid Search
                    #     for lat in param_grid['latent_dim']:
                    #         for hi in param_grid['high_feature_dim']:
                    #             for emd in param_grid['embedding_dim']:
                    #                 args.latent_dim = lat
                    #                 args.high_feature_dim = hi
                    #                 args.embedding_dim = emd
                    #
                    #                 save_dir = f'results/{dataname}/'
                    #                 save_name = f'{args.DATA_SET_NAME}-lr{args.lr}-p{args.noise_rate}-r{args.noise_num}-' \
                    #                             f'lat{args.latent_dim}-hdim{args.high_feature_dim}-emd{args.embedding_dim}' \
                    #                             f'.txt-cl-ml'
                    #                 run(device, args, save_dir, save_name)


                # 随机搜索
                # random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
                # args.latent_dim = random_params['latent_dim']
                # args.high_feature_dim = random_params['high_feature_dim']
                # args.embedding_dim = random_params['embedding_dim']
                # args.common_embedding_dim = random_params['common_embedding_dim']


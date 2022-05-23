# -*- coding: UTF-8 -*-

Fold_numbers = 5
TEST_SPLIT_INDEX = 1


class Args:
    def __init__(self):
        self.DATA_ROOT = './Datasets'
        self.DATA_SET_NAME = 'Emotions'
        self.epochs = 30
        self.pretrain_epochs = 200
        self.show_epoch = 1
        # self.epoch_used_for_final_result = 4
        self.model_save_epoch = 10
        self.model_save_dir = 'model_save_dir'
        self.no_verbose = True
        self.momentumae = 0.9

        self.is_test_in_train = True
        self.batch_size = 512
        self.seed = 43
        self.cuda = True
        self.workers = 0
        self.opt = 'adam'
        self.lr = 1e-3  #
        self.lr_pre = 1e-3  #
        self.weight_decay = 1e-5  #

        self.temperature_f = 0.5
        self.temperature_l = 1.0

        self.latent_dim = 64  # 512
        self.high_feature_dim = 128  # 256
        self.embedding_dim = 64  # 256
        self.class_emb_dim = 64
        # self.cluster_dim = 256  # 256

        self.coef_ml = 1.0
        self.coef_cl = 0.5
        self.coef_rec = 0.5

        self.noise_rate = 0.0
        self.noise_num = 0

        self.in_layers = 2

# class Args:
#     def __init__(self):
#         self.DATA_ROOT = './Datasets'
#         self.DATA_SET_NAME = 'Emotions'
#         self.epochs = 200
#         self.pretrain_epochs = 200
#         self.show_epoch = 1
#         # self.epoch_used_for_final_result = 4
#         self.model_save_epoch = 10
#         self.model_save_dir = 'model_save_dir'
#         self.no_verbose = True
#         self.momentumae = 0.9
#
#         self.using_lp = False
#         self.mu = 1  # [0.001, 0.01, 0.1] 1
#         self.alpha = 1  # [0.1, 1, 10] 1
#         self.zeta = 0.01  # [0.001, 0.01, 0.1] 1
#         self.eta = 1e-2  # lr
#         self.maxiter = 200
#         self.neighbors_num = 10
#
#         self.is_test_in_train = True
#         self.batch_size = 512
#         self.seed = 43
#         self.cuda = True
#         self.workers = 0
#         self.opt = 'adam'
#         self.lr = 1e-3  #
#         self.lr_pre = 1e-3  #
#         self.weight_decay = 1e-5  #
#         self.noise_rate = 0.7
#         self.noise_num = 3
#
#         self.temperature_f = 0.5
#         self.temperature_l = 1.0
#
#         self.latent_dim = 512  # 512 related-to label dim
#         self.high_feature_dim = 128  # 256
#         self.embedding_dim = 64  # 256
#         # self.cluster_dim = 256  # 256
#
#         self.coef_ml = 1.0
#         self.coef_cl = 0.5
#         self.coef_mi = 0.0
#
#         self.tau = 1.0
#
#         # self.using_weight = False
#         # self.gamma = 0.5
#
#         # self.keep_prob = 0.5
#         # self.scale_coeff = 1.0

# Emotions
# class Args:
#     def __init__(self):
#         self.DATA_ROOT = '/home/allen/Datasets/MVML'
#         self.DATA_SET_NAME = 'Emotions'
#         self.epochs = 50
#         self.show_epoch = 1
#         # self.epoch_used_for_final_result = 4
#         self.model_save_epoch = 10
#         self.model_save_dir = 'model_save_dir'
#
#         self.is_test_in_train = True
#         self.batch_size = 512
#         self.seed = 43
#         self.cuda = True
#         self.workers = 0
#         self.opt = 'adam'
#         self.lr = 1e-3  # 1e-3 5e-3
#         self.weight_decay = 1e-5  # 1e-5
#         self.noise_rate = 0.7
#         self.noise_num = 3
#
#         self.temperature_f = 0.5
#         self.temperature_l = 1.0
#
#         self.latent_dim = 64  # 512 related-to label dim
#         self.high_feature_dim = 1024  # 256
#         self.embedding_dim = 1024  # 256
#
#         self.coef_cl = 2.0
#         self.coef_ml = 1.0
#
#         # self.coef_kl = 3.0
#         # self.gamma = 2.0
#
#         self.gamma = 0.5
#
#         self.keep_prob = 0.5
#         self.scale_coeff = 1.0
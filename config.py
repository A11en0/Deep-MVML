# -*- coding: UTF-8 -*-

Fold_numbers = 5
TEST_SPLIT_INDEX = 1


class Args:
    def __init__(self):
        self.DATA_ROOT = '/home/allen/Datasets/MVML'
        self.DATA_SET_NAME = 'Emotions'
        self.epochs = 100
        self.show_epoch = 1
        # self.epoch_used_for_final_result = 4
        self.model_save_epoch = 20
        self.model_save_dir = 'model_save_dir'
        self.resume = False

        self.is_test_in_train = True
        self.batch_size = 512
        self.seed = 43
        self.cuda = True
        self.workers = 0
        self.opt = 'adam'
        self.lr = 1e-3  # 1e-3 5e-3
        self.weight_decay = 1e-5  # 1e-5
        self.noise_rate = 0.7
        self.noise_num = 3

        self.temperature_f = 0.5
        self.temperature_l = 1.0

        self.latent_dim = 256  # 512 related-to label dim
        self.high_feature_dim = 512  # 256
        self.embedding_dim = 512  # 256

        self.coef_cl = 0.0
        self.coef_ml = 1.0

        # self.coef_kl = 3.0

        # self.gamma = 2.0

        self.gamma = 0.5

        self.keep_prob = 0.5
        self.scale_coeff = 1.0


class Args_Pre:
    def __init__(self):
        self.DATA_ROOT = '/home/allen/Datasets/MVML'
        self.DATA_SET_NAME = 'Emotions'
        self.epochs = 200
        self.show_epoch = 1
        # self.epoch_used_for_final_result = 4
        self.model_save_epoch = 30
        self.model_save_dir = 'model_save_dir'

        self.is_test_in_train = True
        self.batch_size = 512
        self.seed = 43
        self.cuda = True
        self.workers = 0
        self.opt = 'adam'
        self.lr = 1e-3  # 1e-3 5e-3
        self.weight_decay = 1e-5  # 1e-5
        self.noise_rate = 0.7
        self.noise_num = 3

        self.temperature_f = 0.5
        self.temperature_l = 1.0

        self.latent_dim = 256  # 512 related-to label dim
        self.high_feature_dim = 512  # 256
        self.embedding_dim = 512  # 256

        self.coef_cl = 0.0
        self.coef_ml = 1.0

        # self.coef_kl = 3.0

        # self.gamma = 2.0

        self.gamma = 0.5

        self.keep_prob = 0.5
        self.scale_coeff = 1.0
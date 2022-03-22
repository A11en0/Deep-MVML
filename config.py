# -*- coding: UTF-8 -*-

Fold_numbers = 5
TEST_SPLIT_INDEX = 1

class Args:
    def __init__(self):
        self.DATA_ROOT = './datasets'
        self.DATA_SET_NAME = 'Pascal.mat'
        self.epoch = 200
        self.show_epoch = 1
        self.epoch_used_for_final_result = 4
        self.model_save_epoch = 5
        self.model_save_dir = 'model_save_dir'
        self.seed = 8
        self.cuda = False

        self.gamma = 0.1  # [0.001, 0.01, 0.1] 1
        self.alpha = 1  # [0.1, 1, 10] 1
        self.zeta = 0.01  # [0.001, 0.01, 0.1] 1
        self.eta = 1e-2  # lr
        self.maxiter = 200

        self.neighbors_num = 10
        self.no_verbose = True
        self.using_lp = False
        self.le = True

        self.ae = False

        self.lr = 1e-3  # 5e-3
        self.partial_rate = 0

        self.common_feature_dim = 512
        self.latent_dim = 64
        self.keep_prob = 0.5
        self.scale_coeff = 1.0

        self.coef_ml = 1.0
        self.coef_kl = 0.0

WEIGHT_DECAY = 1e-5

loss_coefficient = {}

# loss_coefficient['ML_loss'] = 1.0
# loss_coefficient['kl_loss'] = 1.1



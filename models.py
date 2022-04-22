import time
import faiss
import scipy
import torch
import numpy as np
from torch import nn
from sklearn import preprocessing
import torch.nn.functional as F

from layer.VAE import Feature_VAE, Label_VAE, Feature_VAE_Fusing
from layer.view_block import EncoderBlock, DecoderBlock


class Model(nn.Module):
    def __init__(self, view_blocks, common_feature_dim, label_num, device, model_args=None):
        super(Model, self).__init__()
        self.view_blocks = nn.ModuleDict()
        self.view_blocks_codes = []

        for view_block in view_blocks:
            self.view_blocks[str(view_block.code)] = view_block
            self.view_blocks_codes.append(str(view_block.code))

        self.model_args = model_args
        self.common_feature_dim = common_feature_dim
        view_count = len(self.view_blocks)
        self.final_feature_num = (view_count + 1) * common_feature_dim
        self.fc_comm_extract = nn.Linear(common_feature_dim, common_feature_dim)
        self.fc_predictor = nn.Linear(self.final_feature_num, label_num)
        self.device = device

    def forward(self, x):
        view_features_dict = self._extract_view_features(x)
        final_features = torch.zeros(x[0].shape[0], self.final_feature_num).to(self.device)
        comm_feature = torch.zeros(x[0].shape[0], self.common_feature_dim).to(self.device)
        view_count = len(self.view_blocks)

        for view_code, view_feature in view_features_dict.items():
            view_code = int(view_code)
            final_features[:, view_code * self.common_feature_dim: (view_code + 1) *
            self.common_feature_dim] = view_feature[0]  # view-specific feature
            view_comm_feature = self.fc_comm_extract(view_feature[1])   # common feature
            comm_feature += view_comm_feature

        comm_feature /= view_count

        final_features[:, -self.common_feature_dim:] = comm_feature

        outputs = self.fc_predictor(final_features)

        outputs = torch.sigmoid(outputs)
        return outputs

    def _extract_view_features(self, x):
        view_features_dict = {}
        for view_block_code in self.view_blocks_codes:
            view_x = x[int(view_block_code)]
            view_block = self.view_blocks[view_block_code]
            view_features = view_block(view_x)
            view_features_dict[view_block_code] = view_features
        return view_features_dict


def label_propagation(args, Wn, L, Y_pred, Y_P_train, Z_current, gamma, alpha, zeta, maxiter):
    beta = 1  # set beta as the pivot
    eta = args.eta  # learning rate
    Z = Y_P_train

    Z_g = torch.from_numpy(Z).float().detach().cuda()
    Y_P_train_g = torch.from_numpy(Y_P_train).float().detach().cuda()
    Y_pred_g = torch.from_numpy(Y_pred).float().detach().cuda()
    L_g = torch.from_numpy(L).float().detach().cuda()

    with torch.no_grad():
        for i in range(maxiter):
            W_matmul_Z_g = torch.from_numpy(Wn.dot(Z_g.cpu().numpy())).detach().cuda()
            grad = gamma * (Z_g - W_matmul_Z_g) + alpha * (Z_g - Y_P_train_g) + \
                   beta * (Z_g - Y_pred_g) + zeta * (Z_g - Z_g @ L_g)
            Z_g = Z_g - eta * grad

    Z = Z_g.detach().cpu().numpy()

    min_max_scaler = preprocessing.MinMaxScaler()
    Z = min_max_scaler.fit_transform(Z)

    torch.cuda.empty_cache()

    return Z

def estimating_label_correlation_matrix(Y_P):
    num_class = Y_P.shape[1]
    n = Y_P.shape[0]

    R = np.zeros((num_class, num_class))
    for i in range(num_class):
        for j in range(num_class):
            if i == j:
                R[i][j] = 0
            else:
                if np.sum(Y_P[:, i]) == 0 and np.sum(Y_P[:, j]) == 0:
                    R[i][j] = 1e-5  # avoid divide zero error
                else:
                    R[i][j] = Y_P[:, i].dot(Y_P[:, j]) / (Y_P[:, i].sum() + Y_P[:, j].sum())
    D_1_2 = np.diag(1. / np.sqrt(np.sum(R, axis=1)))
    L = D_1_2.dot(R).dot(D_1_2)
    L = np.nan_to_num(L)

    return L

def build_graph(X, k=10, args=None):
    if not args.no_verbose:
        print('Building Graph - V1')
    # kNN search for the graph
    X = X.astype('float32')
    d = X.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index
    
    faiss.normalize_L2(X)
    index.add(X)
    N = X.shape[0]
    Nidx = index.ntotal

    c = time.time()
    D, I = index.search(X, k + 1)
    elapsed = time.time() - c
    if not args.no_verbose:
        print('kNN Search Time: %.4f s' % elapsed)

    # Create the graph
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    return Wn


if __name__ == '__main__':
    f1 = torch.randn(1000, 20)
    f2 = torch.randn(1000, 20)
    features = {0: f1, 1: f2}
    model = Model(features, common_feature_dim=64, label_num=20, model_args=None)
    logit = model(features)
    print(logit)
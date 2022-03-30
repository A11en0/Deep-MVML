import time
import faiss
import scipy
import torch
import numpy as np
from torch import nn
from sklearn import preprocessing
import torch.nn.functional as F


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

class Model_AE(nn.Module):
    def __init__(self, view_blocks, decoder_blocks, common_feature_dim, label_num, device, model_args=None):
        super(Model_AE, self).__init__()
        self.view_blocks = nn.ModuleDict()
        self.view_blocks_codes = []

        self.decoder_blocks = nn.ModuleDict()
        self.decoder_blocks_codes = []

        for view_block in view_blocks:
            self.view_blocks[str(view_block.code)] = view_block
            self.view_blocks_codes.append(str(view_block.code))

        for decoder_block in decoder_blocks:
            self.decoder_blocks[str(decoder_block.code)] = decoder_block
            self.decoder_blocks_codes.append(str(decoder_block.code))

        self.model_args = model_args
        self.common_feature_dim = common_feature_dim
        view_count = len(self.view_blocks)
        self.final_feature_num = (view_count + 1) * common_feature_dim
        self.fc_comm_extract = nn.Linear(common_feature_dim, common_feature_dim)
        self.fc_predictor = nn.Linear(self.final_feature_num, label_num)
        self.device = device

        # initialization

    def forward(self, x, labels):
        view_features_dict = self._extract_view_features(x)
        recons = []
        final_features = torch.zeros(x[0].shape[0], self.final_feature_num).to(self.device)
        comm_feature = torch.zeros(x[0].shape[0], self.common_feature_dim).to(self.device)
        view_count = len(self.view_blocks)

        # common feature
        for view_code, view_feature in view_features_dict.items():
            view_code = int(view_code)
            final_features[:, view_code * self.common_feature_dim: (view_code + 1) *
            self.common_feature_dim] = view_feature[0]  # view-specific feature
            view_comm_feature = self.fc_comm_extract(view_feature[1])  # common feature
            comm_feature += view_comm_feature
        comm_feature /= view_count

        # concat view-specific and common -> Decoder
        for view_code, view_feature in view_features_dict.items():
            concat_features = torch.cat((comm_feature.clone(), view_feature[0].clone()), dim=1)
            decoder_block = self.decoder_blocks[view_code]
            recon = decoder_block(concat_features)
            recons.append(recon)

        # predict
        final_features[:, -self.common_feature_dim:] = comm_feature
        outputs = self.fc_predictor(final_features)
        outputs = torch.sigmoid(outputs)

        return recons, outputs

    def _extract_view_features(self, x):
        view_features_dict = {}
        for view_block_code in self.view_blocks_codes:
            view_x = x[int(view_block_code)]
            view_block = self.view_blocks[view_block_code]
            view_features = view_block(view_x)
            view_features_dict[view_block_code] = view_features
        return view_features_dict

class ModelEmbedding(nn.Module):
    def __init__(self, view_blocks, decoder_blocks, common_feature_dim, label_dim, device, args=None):
        super(ModelEmbedding, self).__init__()
        self.view_blocks = nn.ModuleDict()
        self.view_blocks_codes = []

        self.decoder_blocks = nn.ModuleDict()
        self.decoder_blocks_codes = []

        for view_block in view_blocks:
            self.view_blocks[str(view_block.code)] = view_block
            self.view_blocks_codes.append(str(view_block.code))

        for decoder_block in decoder_blocks:
            self.decoder_blocks[str(decoder_block.code)] = decoder_block
            self.decoder_blocks_codes.append(str(decoder_block.code))

        self.common_feature_dim = args.common_feature_dim

        view_count = len(self.view_blocks)
        self.final_feature_num = (view_count + 1) * common_feature_dim
        self.fc_comm_extract = nn.Linear(common_feature_dim, common_feature_dim)
        # self.fc_predictor = nn.Linear(common_feature_dim, label_dim)
        self.device = device

        # self.label_encoder = LabelEmbedding(label_dim, 64, common_feature_dim)
        # initialization

        # feature layers
        # self.fx1 = nn.Linear(self.final_feature_num, 256)
        self.fx1 = nn.Linear(512+64, 256)
        self.fx2 = nn.Linear(256, 512)
        self.fx3 = nn.Linear(512, 256)
        self.fx_mu = nn.Linear(256, args.latent_dim)
        self.fx_logvar = nn.Linear(256, args.latent_dim)
        self.fx_mu_batchnorm = nn.BatchNorm1d(args.latent_dim)
        self.fx_sigma_batchnorm = nn.BatchNorm1d(args.latent_dim)

        # self.fd_x1 = nn.Linear(self.final_feature_num + args.latent_dim, 512)
        self.fd_x1 = nn.Linear(512+2*args.latent_dim, 512)
        self.fd_x2 = nn.Linear(512, args.embedding_dim)
        self.feat_mp_mu = nn.Linear(args.embedding_dim, label_dim)

        # label layers
        self.fe0 = nn.Linear(label_dim, args.embedding_dim)

        self.fe1 = nn.Linear(args.embedding_dim, 512)
        self.fe2 = nn.Linear(512, 256)
        self.fe_mu = nn.Linear(256, args.latent_dim)
        self.fe_logvar = nn.Linear(256, args.latent_dim)
        self.fe_mu_batchnorm = nn.BatchNorm1d(args.latent_dim)
        self.fe_sigma_batchnorm = nn.BatchNorm1d(args.latent_dim)

        # label layers
        # self.fd1 = nn.Linear(self.final_feature_num + args.latent_dim, 512)
        self.fd1 = nn.Linear(64, 512)
        self.fd2 = nn.Linear(512, args.embedding_dim)
        # self.fd1 = self.fd_x1
        # self.fd2 = self.fd_x2
        self.label_mp_mu = self.feat_mp_mu

        # things they share
        self.dropout = nn.Dropout(p=args.keep_prob)
        self.scale_coeff = args.scale_coeff

        self.attention = nn.Sequential(
            nn.Linear(512+64, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def label_encode(self, x):
        h0 = self.dropout(F.relu(self.fe0(x)))
        h1 = self.dropout(F.relu(self.fe1(h0)))
        h2 = self.dropout(F.relu(self.fe2(h1)))
        mu = self.fe_mu(h2) * self.scale_coeff
        logvar = self.fe_logvar(h2) * self.scale_coeff
        return mu, logvar

    def feat_encode(self, x):
        h1 = self.dropout(F.relu(self.fx1(x)))
        h2 = self.dropout(F.relu(self.fx2(h1)))
        h3 = self.dropout(F.relu(self.fx3(h2)))
        mu = self.fx_mu(h3) * self.scale_coeff
        logvar = self.fx_logvar(h3) * self.scale_coeff
        return mu, logvar

    def label_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def feat_reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def label_decode(self, z):
        h3 = F.relu(self.fd1(z))
        h4 = F.relu(self.fd2(h3))
        h5 = F.normalize(h4, dim=1)
        return h5
        # return torch.sigmoid(self.label_mp_mu(h4))

    def feat_decode(self, z):
        h4 = F.relu(self.fd_x1(z))
        h5 = F.relu(self.fd_x2(h4))
        h6 = F.normalize(h5, dim=1)
        return h6
        # return torch.sigmoid(self.feat_mp_mu(h5))

    def label_forward_(self, label, feat):
        # x = torch.cat((feat, label), 1)
        mu, logvar = self.label_encode(label)
        mu = self.fe_mu_batchnorm(mu)
        logvar = self.fe_sigma_batchnorm(mu)
        z = self.label_reparameterize(mu, logvar)
        return self.label_decode(torch.cat((feat, z), 1)), mu, logvar

    def label_forward(self, label):
        # x = torch.cat((feat, label), 1)
        mu, logvar = self.label_encode(label)
        mu = self.fe_mu_batchnorm(mu)
        logvar = self.fe_sigma_batchnorm(mu)
        z = self.label_reparameterize(mu, logvar)
        return self.label_decode(z), z, mu, logvar

    def feat_forward(self, x):
        # feature encoder
        mu, logvar = self.feat_encode(x)
        mu = self.fx_mu_batchnorm(mu)
        logvar = self.fx_sigma_batchnorm(mu)
        z = self.feat_reparameterize(mu, logvar)
        return self.feat_decode(torch.cat((x, z), 1)), mu, logvar

    def forward(self, feature, label):
        label_emb, label_mu, label_logvar, z = self.label_forward(label)

        # view-specific feature extracts
        view_features_dict = self._extract_view_features(feature)
        final_embed = torch.zeros(feature[0].shape[0], 3, 512).to(self.device)
        comm_feature = torch.zeros(feature[0].shape[0], self.common_feature_dim).to(self.device)
        view_count = len(self.view_blocks)

        # common feature
        for view_code, view_feature in view_features_dict.items():
            view_code = int(view_code)
            final_embed[:, view_code, :] = view_feature[0]
            view_comm_feature = self.fc_comm_extract(view_feature[1])  # common feature
            comm_feature += view_comm_feature

        comm_feature /= view_count
        final_embed[:, -1, :] = comm_feature
        z = z.unsqueeze(1).expand(-1, 3, -1)
        final_embed = torch.cat((final_embed, z), dim=-1)
        A = self.attention(final_embed)  # NxK
        A = A.permute(0, 2, 1)
        A = F.softmax(A, dim=-1)  # softmax over N
        M = torch.matmul(A, final_embed)  # KxL
        feature_embedding = torch.squeeze(M)

        # label_emb, label_mu, label_logvar = self.label_forward(label, feature_embedding)
        feat_emb, feat_mu, feat_logvar = self.feat_forward(feature_embedding)
        embs = self.fe0.weight
        
        label_out = torch.matmul(label_emb, embs).sigmoid()
        feat_out = torch.matmul(feat_emb, embs).sigmoid()

        return label_out, label_mu, label_logvar, feat_out, feat_mu, feat_logvar

    def _extract_view_features(self, x):
        view_features_dict = {}
        for view_block_code in self.view_blocks_codes:
            view_x = x[int(view_block_code)]
            view_block = self.view_blocks[view_block_code]
            view_features = view_block(view_x)
            view_features_dict[view_block_code] = view_features
        return view_features_dict


    # def forward_t(self, feature):


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

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


if __name__ == '__main__':
    f1 = torch.randn(1000, 20)
    f2 = torch.randn(1000, 20)
    features = {0: f1, 1: f2}
    model = Model(features, common_feature_dim=64, label_num=20, model_args=None)
    logit = model(features)
    print(logit)
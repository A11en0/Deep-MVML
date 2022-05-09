import time
import faiss
import scipy
import torch
import numpy as np
from torch import nn
from sklearn import preprocessing
from torch.nn.functional import normalize


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            # nn.ReLU(),
            # nn.Linear(256, feature_dim),
            # nn.ReLU(),
            # nn.Linear(512, feature_dim),
        )

        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 2000),
        #     nn.ReLU(),
        #     nn.Linear(2000, feature_dim),
        # )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, feature_dim, input_dim):
        super(Decoder, self).__init__()
        # self.decoder = nn.Sequential(
        #     nn.Linear(feature_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, input_dim)
        # )

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, num_view, input_size, latent_dim, high_feature_dim,
                 embedding_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []

        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], latent_dim).to(device))
            self.decoders.append(Decoder(latent_dim, embedding_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # self.feature_contrastive_module = nn.Sequential(
        #     nn.Linear(latent_dim, high_feature_dim),
        # )

        # label embedding
        # self.label_emb_layer = nn.Linear(class_num, embedding_dim)

        # label AE
        # self.label_encoder = Encoder(embedding_dim, latent_dim).to(device)
        # self.label_decoder = Decoder(latent_dim, embedding_dim).to(device)

        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, class_num),
        #     nn.Softmax(dim=1)
        # )

        # self.classifier = nn.Sequential(
        # nn.Linear(num_view * embedding_dim, common_embedding_dim),
        # nn.Linear(embedding_dim, class_num),
        # nn.BatchNorm1d(128),
        # nn.ReLU(inplace=True),
        # nn.Dropout(),
        # nn.Linear(128, class_num),
        # nn.BatchNorm1d(class_num),
        # nn.Sigmoid()
        # )

        # self.lat_bn = nn.BatchNorm1d(latent_dim)
        # self.emb_bn = nn.BatchNorm1d(latent_dim)

        self.view = num_view

    def forward(self, xs, labels):
        feat_embs = []
        feat_outs = []
        hs = []
        zs = []
        # cls = []
        embs = self.label_emb_layer.weight  # label embedding

        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)

            zs.append(z)
            # hs.append(h)
            feat_embs.append(xr)
            # cls.append(self.classifier(xr))

        _label_emb = self.label_emb_layer(labels)
        z_label = self.label_encoder(_label_emb)
        # x_z_label = torch.cat([_label_emb, z_label], dim=1)
        label_emb = self.label_decoder(z_label)
        label_out = torch.sigmoid(torch.matmul(label_emb, embs))

        for v in range(self.view):
            feat_outs.append(torch.sigmoid(torch.matmul(feat_embs[v], embs)))

        return feat_outs, label_out, hs, zs
        # return cls, feat_embs, hs, zs

    def forward_old(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds


class Network(nn.Module):
    def __init__(self, num_view, input_size, latent_dim, high_feature_dim,
                 embedding_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []

        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], latent_dim).to(device))
            self.decoders.append(Decoder(latent_dim, embedding_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # self.feature_contrastive_module = nn.Sequential(
        #     nn.Linear(latent_dim, high_feature_dim),
        # )

        # label embedding
        # self.label_emb_layer = nn.Linear(class_num, embedding_dim)

        # label AE
        # self.label_encoder = Encoder(embedding_dim, latent_dim).to(device)
        # self.label_decoder = Decoder(latent_dim, embedding_dim).to(device)

        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, class_num),
        #     nn.Softmax(dim=1)
        # )

        # self.classifier = nn.Sequential(
        # nn.Linear(num_view * embedding_dim, common_embedding_dim),
        # nn.Linear(embedding_dim, class_num),
        # nn.BatchNorm1d(128),
        # nn.ReLU(inplace=True),
        # nn.Dropout(),
        # nn.Linear(128, class_num),
        # nn.BatchNorm1d(class_num),
        # nn.Sigmoid()
        # )

        # self.lat_bn = nn.BatchNorm1d(latent_dim)
        # self.emb_bn = nn.BatchNorm1d(latent_dim)

        self.view = num_view

    def forward(self, xs, labels):
        feat_embs = []
        feat_outs = []
        hs = []
        zs = []
        # cls = []
        embs = self.label_emb_layer.weight  # label embedding

        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)

            zs.append(z)
            # hs.append(h)
            feat_embs.append(xr)
            # cls.append(self.classifier(xr))

        _label_emb = self.label_emb_layer(labels)
        z_label = self.label_encoder(_label_emb)
        # x_z_label = torch.cat([_label_emb, z_label], dim=1)
        label_emb = self.label_decoder(z_label)
        label_out = torch.sigmoid(torch.matmul(label_emb, embs))

        for v in range(self.view):
            feat_outs.append(torch.sigmoid(torch.matmul(feat_embs[v], embs)))

        return feat_outs, label_out, hs, zs
        # return cls, feat_embs, hs, zs

    def forward_old(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds


def label_propagation(args, Wn, L, Y_pred, Y_P_train, Z_current, mu, alpha, zeta, maxiter):
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
            grad = mu * (Z_g - W_matmul_Z_g) + alpha * (Z_g - Y_P_train_g) + \
                   beta * (Z_g - Y_pred_g) + zeta * (Z_g - Z_g @ L_g)
            # grad = alpha * (Z_g - Y_P_train_g) + beta * (Z_g - Y_pred_g) + zeta * (Z_g - Z_g @ L_g)
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
    # model = Model(features, common_feature_dim=64, label_num=20, model_args=None)
    # logit = model(features)
    # print(logit)



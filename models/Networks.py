import torch
import numpy as np
from torch import nn
from torch.nn.functional import normalize

from models.Cluster import ClusterAssignment
from models.Discriminator import Discriminator


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            # nn.BatchNorm1d(128),
            # nn.ReLU(),
            # nn.Linear(128, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

# class Encoder(nn.Module):
#     def __init__(self, input_dim, feature_dim):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, feature_dim),
#             # nn.BatchNorm1d(256),
#             # nn.ReLU(),
#             # nn.Linear(256, feature_dim),
#             # nn.BatchNorm1d(128),
#             # nn.ReLU(),
#             # nn.Linear(128, feature_dim),
#         )
#
#     def forward(self, x):
#         return self.encoder(x)


# class Decoder(nn.Module):
#     def __init__(self, latent_dim, embedding_dim):
#         super(Decoder, self).__init__()
#
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 2000),
#             nn.ReLU(),
#             nn.Linear(2000, 500),
#             nn.ReLU(),
#             # nn.Linear(500, 500),
#             # nn.ReLU(),
#             nn.Linear(500, 200),
#             nn.ReLU(),
#             nn.Linear(200, embedding_dim)
#         )
#
#     def forward(self, x):
#         return self.decoder(x)

# class Encoder(nn.Module):
#     def __init__(self, input_dim, feature_dim):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 500),
#             nn.ReLU(),
#             nn.Linear(500, 500),
#             nn.ReLU(),
#             nn.Linear(500, 2000),
#             nn.ReLU(),
#             nn.Linear(2000, feature_dim),
#         )
#
#     def forward(self, x):
#         return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, feature_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, embedding_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class Network(nn.Module):
    def __init__(self, num_view, input_size, high_feature_dim, latent_dim,
                 embedding_dim, class_num, args, device):
        super(Network, self).__init__()
        self.args = args
        self.encoders = []
        # self.decoders = []
        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], latent_dim).to(device))
            # self.decoders.append(Decoder(latent_dim, embedding_dim).to(device))
            # self.decoders.append(Decoder(latent_dim, embedding_dim).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        # self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(latent_dim, high_feature_dim),
        )

        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, class_num),
        #     nn.Softmax(dim=1)
        # )

        self.classifier = nn.Sequential(
            nn.Linear(num_view * latent_dim, class_num),
            # nn.Linear(256, 128),
            # nn.Linear(128, class_num),
            # nn.Sigmoid()
        )

        self.view = num_view

        # self.assignment = ClusterAssignment(
        #     cluster_number=class_num, embedding_dimension=cluster_dim
        # )

        # self.disc = Discriminator(embedding_dim*2)

    def forward(self, xs, labels):
        hs = []
        qs = []
        xrs = []
        zs = []

        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            # q = self.label_contrastive_module(z)
            # xr = self.decoders[v](z)
            hs.append(h)
            # xrs.append(xr)
            zs.append(z)
            # qs.append(q)

        outputs = self.classifier(torch.concat(zs, dim=1))
        return outputs, hs, xrs, zs

    def forward_(self, xs, labels):
        view_feats = []
        comm_feats = []
        hs = []
        info_scores = []

        for v in range(self.view):
            view_feat = self.encoders[v](xs[v])                        # view-specific feature
            view_comm = self.common_extract(view_feat)                 # common feature
            hs.append(normalize(self.projector(view_feat), dim=1))     # CL projector
            # view_feat = normalize(view_feat, dim=1)                  # normalize
            view_feats.append(view_feat)
            comm_feats.append(view_comm)

        batch_size = view_feats[0].shape[1]
        idx = np.random.permutation(batch_size)
        common = torch.mean(torch.stack(comm_feats), dim=0)  # mean

        for v in range(self.view):
            view_feat = view_feats[v]
            shuf_feat = view_feats[v][:, idx]
            z_f_1 = torch.cat((common, view_feat), dim=1)
            z_f_2 = torch.cat((common, shuf_feat), dim=1)
            z_f_1_score = self.disc(z_f_1)
            z_f_2_score = self.disc(z_f_2)
            score = [z_f_1_score, z_f_2_score]
            info_scores.append(score)

        # lbl_1 = torch.ones(batch_size)
        # lbl_2 = torch.zeros(batch_size)
        # lbl = torch.cat((lbl_1, lbl_2), 1)

        feats_concat = torch.cat(view_feats, dim=1)
        final_feat = torch.cat((feats_concat, common), dim=1)
        feat_out = self.classifier(final_feat)

        return feat_out, view_feats, info_scores, hs


if __name__ == '__main__':
    f1 = torch.randn(1000, 20)
    f2 = torch.randn(1000, 20)
    features = {0: f1, 1: f2}




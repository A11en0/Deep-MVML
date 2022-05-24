import math

import torch
import numpy as np
from torch import nn
from torch.nn.functional import normalize

from models.GNNs import GIN, FDModel


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
        self.view = num_view
        self.encoders = []

        # View-Specific Encoder
        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], latent_dim).to(device))

        # Encoder lists
        self.encoders = nn.ModuleList(self.encoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(latent_dim, high_feature_dim),
        )

        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, class_num),
        #     nn.Softmax(dim=1)
        # )

        # Label semantic encoding module
        self.label_embedding = nn.Parameter(torch.eye(class_num),
                                            requires_grad=False)
        self.label_adj = nn.Parameter(torch.eye(class_num),
                                      requires_grad=False)
        self.GIN_encoder = GIN(2, class_num, args.class_emb_dim,
                              [math.ceil(args.class_emb_dim / 2)])

        # Semantic-guided feature-disentangling module
        self.FD_model = FDModel(latent_dim, args.class_emb_dim,
                                512, embedding_dim, args.in_layers, 1,
                                False, 'leaky_relu', 0.1)

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(num_view * latent_dim, class_num),
            # nn.Linear(256, 128),
            # nn.Linear(128, class_num),
            # nn.Sigmoid()
        )

        # Classifier
        self.cls_conv = nn.Conv1d(class_num, class_num, embedding_dim * num_view, groups=class_num)

    def forward(self, xs, labels):
        xrs = []
        hs = []

        # Generating semantic label embeddings via label semantic encoding module
        label_embedding = self.GIN_encoder(self.label_embedding, self.label_adj)

        for v in range(self.view):
            # Generating label-specific features via semantic-guided feature-disentangling module
            x = xs[v]
            z = self.encoders[v](x)
            z_label = self.FD_model(z, label_embedding)
            h = normalize(self.feature_contrastive_module(z_label), dim=1)
            h = torch.mean(h, dim=1)
            hs.append(h)
            xrs.append(z_label)

        # concat all view-specific features
        X = torch.cat(xrs, dim=2)

        # Classification
        output = self.cls_conv(X).squeeze(2)

        return output, label_embedding, hs


if __name__ == '__main__':
    f1 = torch.randn(1000, 20)
    f2 = torch.randn(1000, 20)
    features = {0: f1, 1: f2}




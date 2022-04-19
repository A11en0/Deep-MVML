import torch.nn as nn
from torch.nn.functional import normalize
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),
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
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        # self.decoder = nn.Sequential(
        #     nn.Linear(feature_dim, 2000),
        #     nn.ReLU(),
        #     nn.Linear(2000, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, 500),
        #     nn.ReLU(),
        #     nn.Linear(500, input_dim)
        # )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, num_view, input_size, latent_dim, high_feature_dim,
                 embedding_dim, common_embedding_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], latent_dim).to(device))
            self.decoders.append(Decoder(embedding_dim, latent_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(latent_dim, high_feature_dim),
        )

        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, class_num),
        #     nn.Softmax(dim=1)
        # )

        self.classifier = nn.Sequential(
            # nn.Linear(num_view * embedding_dim, common_embedding_dim),
            nn.Linear(embedding_dim, common_embedding_dim),
            nn.BatchNorm1d(common_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(common_embedding_dim, class_num),
            nn.BatchNorm1d(class_num),
            nn.Sigmoid()
        )

        self.view = num_view

    def forward(self, xs):
        xrs = []
        hs = []
        zs = []
        cls = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            xr = self.decoders[v](z)
            zs.append(z)
            hs.append(h)
            xrs.append(xr)
            cls.append(self.classifier(xr))
        return cls, xrs, hs, zs

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
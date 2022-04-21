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
                 embedding_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], latent_dim).to(device))
            self.decoders.append(Decoder(embedding_dim, latent_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        # label embedding
        self.label_emb_layer = nn.Linear(class_num, embedding_dim)

        # label AE
        self.label_encoder = Encoder(embedding_dim, latent_dim).to(device)
        self.label_decoder = Decoder(embedding_dim, latent_dim).to(device)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(latent_dim, high_feature_dim),
        )

        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, class_num),
        #     nn.Softmax(dim=1)
        # )

        self.classifier = nn.Sequential(
            # nn.Linear(num_view * embedding_dim, common_embedding_dim),
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, class_num),
            nn.BatchNorm1d(class_num),
            nn.Sigmoid()
        )

        self.view = num_view

    def forward(self, xs, labels):
        feat_embs = []
        feat_outs = []
        hs = []
        zs = []
        cls = []
        embs = self.label_emb_layer.weight  # label embedding

        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            # x_z = torch.cat([x, z], dim=1)
            xr = self.decoders[v](z)
            zs.append(z)
            hs.append(h)
            feat_embs.append(xr)
            cls.append(self.classifier(xr))

        _label_emb = self.label_emb_layer(labels)
        z_label = self.label_encoder(_label_emb)
        # x_z_label = torch.cat([_label_emb, z_label], dim=1)
        label_emb = self.label_decoder(z_label)
        label_out = torch.sigmoid(torch.matmul(label_emb, embs))

        for v in range(self.view):
            feat_outs.append(torch.sigmoid(torch.matmul(feat_embs[v], embs)))

        # return cls, feat_embs, hs, zs
        return feat_outs, label_out, hs, zs

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
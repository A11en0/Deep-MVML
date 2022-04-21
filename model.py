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
        output_dim = 256

        self.encoders = []
        self.decoders = []
        self.feature_mus = []
        self.feature_logvars = []
        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], output_dim).to(device))
            self.decoders.append(Decoder(embedding_dim, latent_dim).to(device))
            self.feature_mus.append(nn.Linear(output_dim, latent_dim))
            self.feature_logvars.append(nn.Linear(output_dim, latent_dim))
            
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_mus = nn.ModuleList(self.feature_mus)
        self.feature_logvars = nn.ModuleList(self.feature_logvars)

        # self.feature_mu = nn.Linear(output_dim, latent_dim)
        # self.feature_logvar = nn.Linear(output_dim, latent_dim)

        # label embedding
        self.label_emb_layer = nn.Linear(class_num, embedding_dim)

        # label AE
        self.label_encoder = Encoder(class_num, output_dim).to(device)
        self.label_decoder = Decoder(embedding_dim, latent_dim).to(device)

        # mu / logvar
        self.label_mu = nn.Linear(output_dim, latent_dim)
        self.label_logvar = nn.Linear(output_dim, latent_dim)
        self.label_mu_batchnorm = nn.BatchNorm1d(latent_dim)
        self.label_sigma_batchnorm = nn.BatchNorm1d(latent_dim)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(latent_dim, high_feature_dim),
        )

        # self.label_contrastive_module = nn.Sequential(
        #     nn.Linear(feature_dim, class_num),
        #     nn.Softmax(dim=1)
        # )

        self.classifier = nn.Sequential(
            # nn.Linear(num_view * embedding_dim, common_embedding_dim),
            nn.Linear(embedding_dim, class_num),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(128, class_num),
            # nn.BatchNorm1d(class_num),
            nn.Sigmoid()
        )

        self.view = num_view

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def feature_forward(self, v, feature):
        output = self.encoders[v](feature)
        mu = self.feature_mus[v](output)
        logvar = self.feature_logvars[v](output)
        # x_z_label = torch.cat(label_emb, z_label], dim=1)
        # mu = self.label_mu_batchnorm(mu)
        # logvar = self.label_sigma_batchnorm(logvar)
        z = self.reparameterize(mu, logvar)
        return self.decoders[v](z), z, mu, logvar

    def label_forward(self, labels):
        # _label_emb = self.label_emb_layer(labels)
        output = self.label_encoder(labels)
        mu = self.label_mu(output)
        logvar = self.label_logvar(output)
        # x_z_label = torch.cat(label_emb, z_label], dim=1)
        # mu = self.label_mu_batchnorm(mu)
        # logvar = self.label_sigma_batchnorm(logvar)
        z = self.reparameterize(mu, logvar)
        return self.label_decoder(z), z, mu, logvar

    def forward(self, xs, labels):
        feat_embs = []
        feat_mus = []
        feat_logvars = []
        feat_outs = []
        hs = []
        zs = []
        cls = []
        embs = self.label_emb_layer.weight  # label embedding

        # for v in range(self.view):
        #     x = xs[v]
        #     z = self.encoders[v](x)
        #     h = normalize(self.feature_contrastive_module(z), dim=1)
        #     # x_z = torch.cat([x, z], dim=1)
        #     xr = self.decoders[v](z)
        #     zs.append(z)
        #     hs.append(h)
        #     feat_embs.append(xr)
            # cls.append(self.classifier(xr))

        for v in range(self.view):
            x = xs[v]
            feat_emb, feat_z, feat_mu, feat_logvar = self.feature_forward(v, x)
            h = normalize(self.feature_contrastive_module(feat_z), dim=1)
            # x_z = torch.cat([x, z], dim=1)
            # xr = self.decoders[v](z)
            zs.append(feat_z)
            hs.append(h)
            feat_mus.append(feat_mu)
            feat_logvars.append(feat_logvar)
            feat_embs.append(feat_emb)

        label_emb, label_z, label_mu, label_logvar = self.label_forward(labels)
        label_out = self.classifier(label_emb)
        # label_out = torch.sigmoid(torch.matmul(label_emb, embs))

        for v in range(self.view):
            feat_outs.append(self.classifier(feat_embs[v]))
            # feat_outs.append(torch.sigmoid(torch.matmul(feat_embs[v], embs)))

        # return cls, feat_embs, hs, zs
        return feat_outs, feat_mus, feat_logvars, label_out, label_mu, label_logvar, hs

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
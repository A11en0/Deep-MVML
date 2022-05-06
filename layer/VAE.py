import torch
from torch import nn
import torch.nn.functional as F

from config import Args


class MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1,
                 with_output_nonlineartity=True):
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        self.fcs = nn.ModuleList()
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_features):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            if with_output_nonlineartity:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_features, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

        self.reset_parameters()

    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()

    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input


class VAE_basic(nn.Module):
    def __init__(self, args):
        super(VAE_basic, self).__init__()
        self.scale_coeff = args.scale_coeff

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

class Label_VAE(VAE_basic):
    def __init__(self, label_dim, label_embedding_layer, out_features, args):
        super(Label_VAE, self).__init__(args)

        # self.fe0 = nn.Linear(label_dim, args.embedding_dim)
        self.label_embedding_layer = label_embedding_layer
        self.encoder = MLP(in_features=args.embedding_dim, out_features=out_features, hidden_features=[256],
                              batchNorm=True, nonlinearity='leaky_relu')

        self.mu = nn.Linear(out_features, args.latent_dim)
        self.logvar = nn.Linear(out_features, args.latent_dim)
        self.mu_batchnorm = nn.BatchNorm1d(args.latent_dim)
        self.sigma_batchnorm = nn.BatchNorm1d(args.latent_dim)

        self.decoder = MLP(in_features=args.latent_dim, out_features=args.embedding_dim, hidden_features=[256],
                           batchNorm=True, nonlinearity='leaky_relu')

    def encode(self, x):
        output = self.label_embedding_layer(x)
        output = self.encoder(output)
        mu = self.mu(output) * self.scale_coeff
        logvar = self.logvar(output) * self.scale_coeff
        return mu, logvar

    def decode(self, z):
        output = self.decoder(z)
        output = F.normalize(output, dim=1)
        return output

    def forward(self, label):
        # x = torch.cat((feat, label), 1)
        mu, logvar = self.encode(label)
        mu = self.mu_batchnorm(mu)
        logvar = self.sigma_batchnorm(logvar)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), z, mu, logvar

class Feature_VAE(VAE_basic):
    def __init__(self, in_features, out_features, hidden_features, label_dim, view_count, args):
        super(Feature_VAE, self).__init__(args)
        # encoder
        self.encoder = MLP(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                           batchNorm=True, nonlinearity='leaky_relu')
        self.mu = nn.Linear(out_features, args.latent_dim)
        self.logvar = nn.Linear(out_features, args.latent_dim)
        self.mu_batchnorm = nn.BatchNorm1d(args.latent_dim)
        self.sigma_batchnorm = nn.BatchNorm1d(args.latent_dim)

        # decoder
        self.decoder = MLP(in_features=in_features + args.latent_dim, out_features=args.embedding_dim, hidden_features=[512],
                           batchNorm=True, nonlinearity='leaky_relu')

        # self.feat_mp_mu = nn.Linear(args.embedding_dim, label_dim)

    def encode(self, x):
        output = self.encoder(x)
        mu = self.mu(output) * self.scale_coeff
        logvar = self.logvar(output) * self.scale_coeff
        return mu, logvar

    def decode(self, z):
        output = self.decoder(z)
        output = F.normalize(output, dim=1)
        return output

    def forward(self, x):
        # feature encoder
        mu, logvar = self.encode(x)

        # batchnorm
        mu = self.mu_batchnorm(mu)
        logvar = self.sigma_batchnorm(logvar)
        z = self.reparameterize(mu, logvar)

        # feature decoder
        output = self.decode(torch.cat((x, z), dim=1))
        return output, mu, logvar


class Feature_VAE_Fusing(VAE_basic):
    def __init__(self, in_features, out_features, hidden_features, label_dim, view_count, args):
        super(Feature_VAE_Fusing, self).__init__(args)
        # encoder
        self.encoder = MLP(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                           batchNorm=True, nonlinearity='leaky_relu')
        self.mu = nn.Linear(out_features, args.latent_dim)
        self.logvar = nn.Linear(out_features, args.latent_dim)
        self.mu_batchnorm = nn.BatchNorm1d(args.latent_dim)
        self.sigma_batchnorm = nn.BatchNorm1d(args.latent_dim)

        # decoder
        self.decoder = MLP(in_features=args.common_feature_dim + args.latent_dim, out_features=args.embedding_dim, hidden_features=[512],
                           batchNorm=True, nonlinearity='leaky_relu')

        # self.feat_mp_mu = nn.Linear(args.embedding_dim, label_dim)

    def encode(self, x):
        output = self.encoder(x)
        mu = self.mu(output) * self.scale_coeff
        logvar = self.logvar(output) * self.scale_coeff
        return mu, logvar

    def decode(self, z):
        output = self.decoder(z)
        output = F.normalize(output, dim=1)
        return output

    def forward(self, view_features, comm_features):
        feat_embeddings = []
        num_view = len(view_features)

        # feature encoder
        view_feature = torch.cat(view_features, dim=1)
        comm_feature = torch.stack(comm_features).mean(dim=0)
        fusing_features = torch.cat((view_feature, comm_feature), dim=1)

        mu, logvar = self.encode(fusing_features)

        # batch norm
        mu = self.mu_batchnorm(mu)
        logvar = self.sigma_batchnorm(logvar)
        z = self.reparameterize(mu, logvar)

        # feature decoder
        for v in range(num_view):
            x_z = torch.cat((view_features[v], z), dim=1)
            feat_embeddings.append(self.decode(x_z))

        return feat_embeddings, mu, logvar


if __name__ == '__main__':
    data = torch.randn(256, 1536)
    target = torch.randn(256, 14)

    args = Args()
    fe0 = nn.Linear(14, args.embedding_dim)
    # feature_vae = Feature_VAE(in_features=512*3, out_features=256, hidden_features=[256, 512], label_dim=14, view_count=2, args=args)
    label_vae = Label_VAE(label_dim=14, label_embedding_layer=fe0, out_features=256, args=args)
    # print(feature_vae)

    print(label_vae)
    rets = label_vae(target)
    print(rets)



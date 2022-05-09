import torch
from torch import nn


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

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, feature_dim, input_dim):
        super(Decoder, self).__init__()

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

        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], embedding_dim).to(device))

        self.encoders = nn.ModuleList(self.encoders)

        # self.classifier = nn.Linear(num_view * embedding_dim, class_num)

        self.classifier = nn.Sequential(
            nn.Linear(num_view * embedding_dim + embedding_dim, class_num),
            # nn.Linear(embedding_dim, class_num),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(128, class_num),
            # nn.BatchNorm1d(class_num),
            nn.Sigmoid()
        )

        self.fc_comm_extract = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.view = num_view

    def forward(self, xs, labels):
        feat_embs = []
        view_comm_feats = []

        for v in range(self.view):
            view_feat = self.encoders[v](xs[v])
            view_comm_feats.append(self.fc_comm_extract(view_feat))
            feat_embs.append(view_feat)

        comm_embs = torch.stack(view_comm_feats).mean(dim=0)
        feat_embs = torch.cat(feat_embs, dim=1)
        feat_embs = torch.cat([feat_embs, comm_embs], dim=1)

        feat_outs = self.classifier(feat_embs)

        return feat_outs


if __name__ == '__main__':
    f1 = torch.randn(1000, 20)
    f2 = torch.randn(1000, 20)
    features = {0: f1, 1: f2}



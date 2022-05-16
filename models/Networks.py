import torch
from torch import nn
from torch.nn.functional import normalize

from models.Cluster import ClusterAssignment


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
    def __init__(self, num_view, input_size, high_feature_dim,
                 embedding_dim, cluster_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []

        for v in range(num_view):
            self.encoders.append(Encoder(input_size[v], embedding_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)

        # self.ln = nn.Sequential(
        #     nn.Linear(embedding_dim, 256),
        #     # nn.BatchNorm1d(256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, embedding_dim),
        #     # nn.BatchNorm1d(embedding_dim),
        #     nn.ReLU(inplace=True),
        # )

        self.classifier = nn.Sequential(
            nn.Linear(num_view * embedding_dim, class_num),
            # nn.Linear(embedding_dim, class_num),
            # nn.BatchNorm1d(128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(128, class_num),
            # nn.BatchNorm1d(class_num),
            # nn.Sigmoid()
        )

        self.fc_comm_extract = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim)
        )

        self.comm_encoder = nn.Linear(embedding_dim*num_view, cluster_dim)

        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, class_num),
        )

        # self.projector = self.classifier

        self.assignment = ClusterAssignment(
            cluster_number=class_num, embedding_dimension=cluster_dim
        )

        self.view = num_view

    def forward(self, xs, labels):
        feat_embs = []
        comm_feats = []
        hs = []

        for v in range(self.view):
            view_feat = self.encoders[v](xs[v])
            view_comm = self.fc_comm_extract(view_feat)
            hs.append(normalize(self.projector(view_feat), dim=1))
            # view_feat = normalize(view_feat, dim=1)
            feat_embs.append(view_feat)
            comm_feats.append(view_comm)

        cat_view_comm = torch.cat(comm_feats, dim=1)
        cluster_out = self.assignment(self.comm_encoder(cat_view_comm))
        # target = target_distribution(output).detach()
        # loss = loss_function(output.log(), target) / output.shape[0]

        feat_emb_concat = torch.cat(feat_embs, dim=1)
        feat_out = self.classifier(feat_emb_concat)

        return feat_out, feat_embs, cluster_out, hs


if __name__ == '__main__':
    f1 = torch.randn(1000, 20)
    f2 = torch.randn(1000, 20)
    features = {0: f1, 1: f2}



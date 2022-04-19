import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_f, temperature_l, args, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.args = args
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def info_nce_loss(self, h_i, h_j):
        N = 2 * self.batch_size
        features = torch.cat((h_i, h_j), dim=0)

        labels = torch.cat([torch.arange(self.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.temperature_f
        loss = self.criterion(logits, labels)
        loss = loss / N
        return loss

    def forward_feature(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f

        # sim_i_j = torch.diag(sim, h_i.shape[0])
        # sim_j_i = torch.diag(sim, -h_i.shape[0])

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy
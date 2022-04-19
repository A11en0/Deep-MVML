import torch.nn as nn


class FusionNet(nn.Module):
    """N (the number of views) FusionNet help h reconstruct x"""
    def __init__(self, h_dim, n_latents):
        super(FusionNet, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(h_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_latents),)
        self.n_latents = n_latents

    def forward(self, h):
        x_re = self.linears(h)
        return x_re

class UncertaintyNet(nn.Module):
    """N (the number of views) FusionNet help h reconstruct x"""
    def __init__(self, h_dim, data_dim):
        super(UncertaintyNet, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(h_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, h):
        sigma = self.linears(h)
        return sigma
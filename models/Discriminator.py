import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Linear(input_dim, 1),
            # nn.Linear(256, 128),
            # nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, c):
        logits = self.disc(c)

        return logits


# class Discriminator_(nn.Module):
#     def __init__(self, n_h):
#         super(Discriminator, self).__init__()
#         self.f_k = nn.Bilinear(n_h, n_h, 1)
#
#         for m in self.modules():
#             self.weights_init(m)
#
#     def weights_init(self, m):
#         if isinstance(m, nn.Bilinear):
#             torch.nn.init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.fill_(0.0)
#
#     def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
#         c_x = torch.unsqueeze(c, 1)
#         c_x = c_x.expand_as(h_pl)
#
#         sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
#         sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)
#
#         if s_bias1 is not None:
#             sc_1 += s_bias1
#         if s_bias2 is not None:
#             sc_2 += s_bias2
#
#         logits = torch.cat((sc_1, sc_2), 1)
#
#         return logits


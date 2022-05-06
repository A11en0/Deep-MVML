# -*- coding: UTF-8 -*-
import torch
import torch.nn.functional as F


def _KL_loss(predictions, true_distributions):
    predictions = F.log_softmax(predictions, dim=1)
    KL = (true_distributions * predictions).sum()
    KL = -1.0 * KL / predictions.shape[0]
    return KL

def compute_loss(input_label, fe_out, fe_mu, fe_logvar, fx_out, fx_mu, fx_logvar, args):
    kl_loss = torch.mean(0.5*torch.sum((fx_logvar-fe_logvar)-1
                                       +torch.exp(fe_logvar-fx_logvar)
                                       +torch.square(fx_mu-fe_mu)/(torch.exp(fx_logvar)+1e-6), dim=1))
    return kl_loss

def calc_kl_loss(fx_mu, fx_logvar, fe_mu, fe_logvar, input_label):
    # GM-VAE
    # std = torch.exp(0.5 * fx_logvar)
    # eps = torch.randn_like(std)
    # fx_sample = fx_mu + eps * std
    # fx_var = torch.exp(fx_logvar)
    # fe_var = torch.exp(fe_logvar)
    # kl_loss = (log_normal(fx_sample, fx_mu, fx_var) - log_normal_mixture(fx_sample, fe_mu, fe_var, input_label)).mean()

    # VAE
    kl_loss = torch.mean(0.5 * torch.sum(
        (fx_logvar - fe_logvar) - 1 + torch.exp(fe_logvar - fx_logvar) + torch.square(fx_mu - fe_mu) / (
                    torch.exp(fx_logvar) + 1e-6), dim=1))
    return kl_loss

def log_normal_mixture(z, m, v, mask=None):
    # m = m.unsqueeze(0).expand(z.shape[0], -1, -1)
    # v = v.unsqueeze(0).expand(z.shape[0], -1, -1)
    # batch, mix, dim = m.size()
    # z = z.view(batch, 1, dim).expand(batch, mix, dim)

    z = z.unsqueeze(1).expand(-1, mask.shape[1], -1)
    m = m.unsqueeze(1).expand(-1, mask.shape[1], -1)
    v = v.unsqueeze(1).expand(-1, mask.shape[1], -1)

    indiv_log_prob = log_normal(z, m, v) + torch.ones_like(mask)*(-1e6)*(1.-mask)
    log_prob = log_mean_exp(indiv_log_prob, mask)
    return log_prob

def _log_normal_mixture(z, m, v, mask=None):
    m = m.unsqueeze(0).expand(z.shape[0], -1, -1)
    v = v.unsqueeze(0).expand(z.shape[0], -1, -1)
    batch, mix, dim = m.size()
    z = z.view(batch, 1, dim).expand(batch, mix, dim)
    indiv_log_prob = log_normal(z, m, v) + torch.ones_like(mask)*(-1e6)*(1.-mask)
    log_prob = log_mean_exp(indiv_log_prob, mask)
    return log_prob

def log_normal(x, m, v):
    log_prob = (-0.5 * (torch.log(v + 1e-6) + (x-m).pow(2) / v)).sum(-1)
    return log_prob

def log_mean_exp(x, mask):
    return log_sum_exp(x, mask) - torch.log(mask.sum(1) + 1e-6)

def log_sum_exp(x, mask):
    max_x = torch.max(x, 1)[0]
    new_x = x - max_x.unsqueeze(1).expand_as(x)
    return max_x + (new_x.exp().sum(1) + 1e-6).log()

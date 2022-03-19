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

# def _logit_ML_loss(view_predictions, true_labels):
#     view_predictions_sig = torch.sigmoid(view_predictions)
#     criterion = nn.BCELoss()
#     ML_loss = criterion(view_predictions_sig, true_labels)
#     return ML_loss

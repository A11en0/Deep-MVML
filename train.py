# -*- coding: UTF-8 -*-
import os
from functools import reduce

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.loss import Loss, calc_kl_loss, cal_kl_loss
from utils.ml_metrics import all_metrics


@ torch.no_grad()
def test(model, features, labels, device, model_state_path=None, is_eval=False):
    if model_state_path:
        model.load_state_dict(torch.load(model_state_path))

    metrics_results = None

    model.eval()

    # CUDA
    for i, _ in enumerate(features):
        features[i] = features[i].to(device)
    labels = labels.to(device)

    # prediction
    with torch.no_grad():
        feat_outs, feat_mus, feat_logvars, label_out, label_mu, label_logvar, hs = model(features, labels)
        # feat_outs, label_out, hs, zs = model(features, labels)
        outputs = reduce(lambda x, y: x + y, feat_outs) / len(feat_outs)

    outputs = outputs.cpu().numpy()
    preds = (outputs > 0.5).astype(int)

    # eval
    if is_eval:
        target = labels.int().cpu().numpy()
        metrics_results = all_metrics(outputs, preds, target)

    return metrics_results, preds

class Trainer(object):
    def __init__(self, model, args, device):
        self.model = model
        self.epochs = args.epochs
        self.show_epoch = args.show_epoch
        self.model_save_epoch = args.model_save_epoch
        self.model_save_dir = args.model_save_dir
        self.device = device
        self.args = args

        if args.opt == 'adam':
            self.opti = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd':
            self.opti = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.lr_s = torch.optim.lr_scheduler.StepLR(self.opti, step_size=20, gamma=0.9)

    def fit(self, train_loader, train_features, train_partial_labels, test_features, test_labels, class_num, fold):
        loss_list = []
        best_F1, best_epoch = 0.0, 0.0
        writer = SummaryWriter()
        mse = torch.nn.MSELoss()
        criterion = Loss(self.args.batch_size, class_num, self.args.temperature_f, self.args.temperature_l,
                         self.args, self.device).to(self.device)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        for epoch in range(self.epochs):
            self.model.train()
            for step, (inputs, labels, index) in enumerate(train_loader):

                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)

                labels = labels.to(self.device)

                feat_outs, feat_mus, feat_logvars, label_out, label_mu, label_logvar, hs \
                    = self.model(inputs, labels)

                # feat_outs, label_out, hs, zs = self.model(inputs, labels)
                # cls, feat_embs, hs, zs = self.model(inputs, labels)

                _cl_loss = []
                _cls_loss = []
                # _mse_loss = []

                _kl_loss = []

                # kl_loss
                for v in range(self.model.view):
                    feat_mu, feat_logvar = feat_mus[v], feat_logvars[v]
                    _kl_loss.append(cal_kl_loss(feat_mu, feat_logvar, label_mu, label_logvar,))
                kl_loss = sum(_kl_loss)

                # close loss
                # for v in range(self.model.view):
                #     _mse_loss.append(mse(feat_outs[v], label_out))
                # mse_loss = sum(_mse_loss)

                # contrastive loss
                for v in range(self.model.view):
                    for w in range(v + 1, self.model.view):
                        _cl_loss.append(criterion.info_nce_loss(hs[v], hs[w]))
                        # _cl_loss.append(criterion.info_nce_loss(zs[v], zs[w]))
                        # _cl_loss.append(criterion.info_nce_loss(xrs[v], xrs[w]))
                    # _cl_loss.append(mse(xs[v], xrs[v]))
                cl_loss = sum(_cl_loss)

                # classification loss
                for v in range(len(feat_outs)):
                    _cls_loss.append(F.binary_cross_entropy(feat_outs[v], labels))
                nll_loss_x = sum(_cls_loss)
                nll_loss_y = F.binary_cross_entropy(label_out, labels)
                ml_loss = 0.5*(nll_loss_x + nll_loss_y)

                # for v in range(len(cls)):
                #     _cls_loss.append(F.binary_cross_entropy(cls[v], labels))
                # ml_loss = sum(_cls_loss)

                loss = self.args.coef_cl * cl_loss + self.args.coef_ml * ml_loss + self.args.coef_kl*kl_loss
                # loss = self.args.coef_cl * cl_loss + self.args.coef_ml * ml_loss + mse_loss

                # reconstruction loss
                # for v in range(len(xrs)):
                #     loss_list.append(criterion(inputs[v], xrs[v]))
                # loss = sum(loss_list)

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}\t CL Loss: {self.args.coef_cl*cl_loss:.4f}' \
                            f'\tML Loss: {self.args.coef_ml*ml_loss:.4f}\t KL Loss: {self.args.coef_kl*kl_loss:.4f}'

                # show loss info
                if epoch % self.show_epoch == 0 and step == 0:
                    print(print_str)
                    epoch_loss = dict()
                    # writer.add_scalar("Loss/train", loss, epoch)  # log
                    # plotter.plot('loss', 'train', 'Class Loss', epoch, _ML_loss)
                    loss_list.append(epoch_loss)

                self.opti.zero_grad()
                loss.backward()
                self.opti.step()

                # evaluation
                if epoch % self.show_epoch == 0 and step == 0 and self.args.is_test_in_train:
                    metrics_results, _ = test(self.model, test_features, test_labels, self.device, is_eval=True)

                    # draw figure to find best epoch number
                    for i, key in enumerate(metrics_results):
                        print(f"{key}: {metrics_results[key]:.4f}", end='\t')
                        loss_list[epoch][key] = metrics_results[key]
                    print("\n")

                    if best_F1 < metrics_results['micro_f1']:
                        best_F1, best_epoch = metrics_results['micro_f1'], epoch

                # if (epoch + 1) % self.model_save_epoch == 0:
                #     torch.save(self.model.state_dict(),
                #             os.path.join(self.model_save_dir,
                #                          'fold' + str(fold)+'_' + 'epoch' + str(epoch + 1) + '.pth'))

        writer.flush()
        writer.close()
        print(f"best_F1: {best_F1}, epoch {best_epoch}")

        return loss_list


if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)
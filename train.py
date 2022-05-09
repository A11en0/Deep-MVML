# -*- coding: UTF-8 -*-
import os
from functools import reduce

import torch
import torch.optim as optim
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

from model import estimating_label_correlation_matrix, build_graph, label_propagation
from utils.loss import Loss
from utils.ml_metrics import all_metrics


@ torch.no_grad()
def test(model, features, labels, weight_var, device, model_state_path=None, is_eval=False, args=None):
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
        feat_outs, label_out, hs, zs = model(features, labels)

        # for v in range(len(feat_outs)):
        #     loss_temp = criterion(feat_outs[v], labels)
        #     loss += (weight_var[v] ** gamma) * loss_temp

        if args.using_weight:
            output_var = torch.stack(feat_outs)
            weight_var = weight_var.unsqueeze(1)
            output_weighted = weight_var * output_var
            outputs = torch.sum(output_weighted, 0)

        else:
            outputs = torch.stack(feat_outs).mean(dim=0)

        # weight_var = weight_var.unsqueeze(2)
        # test = weight_var * output_var
        # weight_var = weight_var.expand(weight_var.size(0), output_var.shape[1], output_var.shape[2])

        # weight_var = weight_var[:, :, 1]
        # weight_var = weight_var[:, 1]

        # outputs = reduce(lambda x, y: x + y, feat_outs) / len(feat_outs)

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
        elif args.opt == 'adagrad':
            self.opti = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'adadelta':
            self.opti = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'rmsprop':
            self.opti = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.lr_s = torch.optim.lr_scheduler.StepLR(self.opti, step_size=20, gamma=0.9)

    def fit(self, train_loader, train_features, train_partial_labels, test_features, test_labels, class_num, fold):
        loss_list = []
        best_F1, best_epoch = 0.0, 0.0
        ml_loss, cl_loss = 0.0, 0.0
        train_partial_labels_np, train_pred_np, train_lp_np, labels_lp = [], [], [], []
        Wn, L = [], []

        # writer = SummaryWriter()
        mse = torch.nn.MSELoss()
        criterion = Loss(self.args.batch_size, class_num, self.args.temperature_f, self.args.temperature_l,
                         self.args, self.device).to(self.device)

        weight_var = torch.ones(self.model.view, class_num) * (1 / self.model.view)
        weight_var = weight_var.to(self.device)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        if self.args.using_lp:
            train_partial_labels_np = train_partial_labels.numpy().copy()
            train_pred_np = train_partial_labels_np.copy()
            train_lp_np = train_partial_labels_np.copy()

            Wn = 0.0
            for id, view_feature in train_features.items():
                view_feature = view_feature.numpy()
                Wn_tmp = build_graph(view_feature, k=self.args.neighbors_num, args=self.args)
                Wn += Wn_tmp

            L = estimating_label_correlation_matrix(train_partial_labels_np)

        for epoch in range(self.epochs):
            self.model.train()

            if self.args.using_lp:
                maxiter = self.args.maxiter
                train_lp_np = label_propagation(self.args, Wn, L, train_pred_np, train_partial_labels_np,
                                              train_lp_np, mu=self.args.mu, alpha=self.args.alpha,
                                              zeta=self.args.zeta, maxiter=maxiter)

            for step, (inputs, labels, index) in enumerate(train_loader):

                if self.args.using_lp:
                    labels_lp = torch.from_numpy(train_lp_np[index]).float().to(self.device)

                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)
                labels = labels.to(self.device)

                feat_outs, label_out, hs, zs = self.model(inputs, labels)
                _cl_loss = []
                _cls_loss = []
                weight_up_list = []

                # contrastive loss
                # for v in range(self.model.view):
                #     for w in range(v + 1, self.model.view):
                #         _cl_loss.append(criterion.info_nce_loss(hs[v], hs[w]))
                # cl_loss = sum(_cl_loss)

                # classification loss
                if self.args.using_lp:
                    for v in range(len(feat_outs)):
                        # 采用多标签设置
                        _weight_var = weight_var[v] ** self.args.gamma
                        # _loss_tmp = F.binary_cross_entropy(feat_outs[v], labels, weight=_weight_var)
                        _loss_tmp = F.binary_cross_entropy(feat_outs[v], labels_lp, reduction='none')
                        _loss_tmp = _weight_var * _loss_tmp
                        _loss_tmp = _loss_tmp.mean(dim=0)
                        _cls_loss.append(_loss_tmp)
                        weight_up_temp = _loss_tmp ** (1 / (1 - self.args.gamma))
                        weight_up_list.append(weight_up_temp)

                    nll_loss_x = torch.stack(_cls_loss).mean(dim=1).sum()
                    nll_loss_y = F.binary_cross_entropy(label_out, labels_lp)
                    ml_loss = 0.5 * (nll_loss_x + nll_loss_y)
                    # ml_loss = F.binary_cross_entropy(outputs, labels_lp)
                    weight_up_var = torch.stack(weight_up_list).detach()
                    weight_down_var = weight_up_var.sum(dim=0)
                    weight_var = torch.div(weight_up_var, weight_down_var)

                elif self.args.using_weight:
                    for v in range(len(feat_outs)):
                        # 采用多标签设置
                        _weight_var = weight_var[v] ** self.args.gamma
                        # _loss_tmp = F.binary_cross_entropy(feat_outs[v], labels, weight=_weight_var)
                        _loss_tmp = F.binary_cross_entropy(feat_outs[v], labels, reduction='none')
                        _loss_tmp = _weight_var * _loss_tmp
                        _loss_tmp = _loss_tmp.mean(dim=0)
                        _cls_loss.append(_loss_tmp)
                        weight_up_temp = _loss_tmp ** (1 / (1 - self.args.gamma))
                        weight_up_list.append(weight_up_temp)

                    nll_loss_x = torch.stack(_cls_loss).mean(dim=1).sum()
                    nll_loss_y = F.binary_cross_entropy(label_out, labels)
                    ml_loss = 0.5 * (nll_loss_x + nll_loss_y)
                    # ml_loss = F.binary_cross_entropy(outputs, labels)

                    weight_up_var = torch.stack(weight_up_list).detach()
                    weight_down_var = weight_up_var.sum(dim=0)
                    weight_var = torch.div(weight_up_var, weight_down_var)
                else:
                    for v in range(len(feat_outs)):
                        _loss_tmp = F.binary_cross_entropy(feat_outs[v], labels)
                        _cls_loss.append(_loss_tmp)

                    nll_loss_x = torch.stack(_cls_loss).sum()
                    nll_loss_y = F.binary_cross_entropy(label_out, labels)
                    ml_loss = 0.5 * (nll_loss_x + nll_loss_y)

                # nll_loss_x = torch.stack(_cls_loss).mean(dim=1).sum()
                # nll_loss_y = F.binary_cross_entropy(label_out, labels)
                # ml_loss = 0.5*(nll_loss_x + nll_loss_y)

                # for v in range(len(cls)):
                #     _cls_loss.append(F.binary_cross_entropy(cls[v], labels))
                # cls_loss = sum(_cls_loss)

                # loss = self.args.coef_cl * cl_loss + self.args.coef_ml * ml_loss
                loss = self.args.coef_ml * ml_loss

                # reconstruction loss
                # for v in range(len(xrs)):
                #     loss_list.append(criterion(inputs[v], xrs[v]))
                # loss = sum(loss_list)

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}\t CL Loss: {self.args.coef_cl*cl_loss:.4f}' \
                            f'\tML Loss: {self.args.coef_ml*ml_loss:.4f}'

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

                # test
                if self.args.using_lp:
                    _, train_pred_np = test(self.model, train_features, train_partial_labels, weight_var, self.device, is_eval=False, args=self.args)

                # evaluation
                if epoch % self.show_epoch == 0 and step == 0 and self.args.is_test_in_train:
                    metrics_results, _ = test(self.model, test_features, test_labels, weight_var, self.device, is_eval=True, args=self.args)

                    # draw figure to find best epoch number
                    for i, key in enumerate(metrics_results):
                        print(f"{key}: {metrics_results[key]:.4f}", end='\t')
                        loss_list[epoch][key] = metrics_results[key]
                    print("\n")

                    if best_F1 < metrics_results['micro_f1']:
                        best_F1, best_epoch = metrics_results['micro_f1'], epoch

          print(f"best_F1: {best_F1}, epoch {best_epoch}")

        return loss_list, weight_var


if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)
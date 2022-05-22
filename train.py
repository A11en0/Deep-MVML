# -*- coding: UTF-8 -*-
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.utils.tensorboard import SummaryWriter

# from model import estimating_label_correlation_matrix, build_graph, label_propagation
from models.Disambiguation import estimating_label_correlation_matrix, build_graph, label_propagation
from utils.draw import plot_embedding
from utils.loss import Loss, cross_modal_contrastive_ctriterion
from utils.ml_metrics import all_metrics


@ torch.no_grad()
def test(model, features, labels, device, model_state_path=None, is_eval=False, args=None):
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
        outputs, _, _, _ = model(features, labels)
        outputs = outputs.sigmoid()

    outputs = outputs.cpu().numpy()
    preds = (outputs > 0.5).astype(int)

    # eval
    if is_eval:
        target = labels.int().cpu().numpy()
        metrics_results = all_metrics(outputs, preds, target)

    return metrics_results, preds


class Trainer(object):
    def __init__(self, model, writer, args, device):
        self.model = model
        self.epochs = args.epochs
        self.pretrain_epochs = args.pretrain_epochs
        self.show_epoch = args.show_epoch
        self.model_save_epoch = args.model_save_epoch
        self.model_save_dir = args.model_save_dir
        self.writer = writer
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

        # CL = Loss(self.args.batch_size, class_num, self.args.temperature_f, self.args.temperature_l,
        #                  self.args, self.device).to(self.device)
        # kl_div = nn.KLDivLoss(size_average=False)
        # kmeans = KMeans(n_clusters=class_num, n_init=20)

        # fine-tune
        for encoder in self.model.encoders.children():
            for param in encoder.parameters():
                param.requires_grad = False

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

                outputs, hs, xrs, zs = self.model(inputs, labels)

                # contrastive loss
                # cl_loss_list = []
                # for v in range(self.model.view):
                #     for w in range(v + 1, self.model.view):
                #         cl_loss_list.append(CL.info_nce_loss(hs[v], hs[w]))
                # cl_loss = sum(cl_loss_list) / len(cl_loss_list)

                # classification loss
                if self.args.using_lp:
                    ml_loss = F.binary_cross_entropy_with_logits(outputs, labels_lp)
                else:
                    ml_loss = F.binary_cross_entropy_with_logits(outputs, labels)

                loss = self.args.coef_ml * ml_loss

                # loss = ml_loss
                # cl_loss = self.args.coef_cl * cl_loss
                # loss = ml_loss + cl_loss
                # info_loss = self.args.coef_mi * info_loss

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}'
                    # f'\t ML Loss: {ml_loss:.4f}' \
                            # f'\t CL Loss: {cl_loss:.4f}' \

                # show loss info
                if epoch % self.show_epoch == 0 and step == 0:
                    print(print_str)
                    self.writer.add_scalar("Loss/Loss", loss.item(), global_step=epoch)
                    # self.writer.add_scalar("Loss/ML Loss", ml_loss.item(), global_step=epoch)
                    # self.writer.add_scalar("Loss/CL Loss", cl_loss.item(), global_step=epoch)
                    # self.writer.add_scalar("Loss/MI Loss", info_loss.item(), global_step=epoch)
                    # self.writer.add_embedding(mat=hs[1], metadata=labels, global_step=epoch)
                    # plot_embedding(hs[0], hs[1], labels)
                    epoch_loss = dict()
                    loss_list.append(epoch_loss)
                    # self.writer.add_embedding(mat=hs[0], metadata=labels, global_step=epoch)
                    # self.writer.add_embedding(mat=hs[1], metadata=labels, global_step=epoch)

                # if epoch % 100 == 0 and step == 0:
                #     plot_embedding(hs[0], hs[1], labels)
                    # self.writer.add_embedding(mat=hs[0], metadata=labels, global_step=epoch)
                    # self.writer.add_embedding(mat=hs[1], metadata=labels, global_step=epoch)

                self.opti.zero_grad()
                loss.backward()
                self.opti.step()

                # 每一个epoch，记录各层权重、梯度
                # for name, param in self.model.named_parameters():  # 返回网络的
                #     self.writer.add_histogram(name + '_grad', param.grad, epoch)
                #     self.writer.add_histogram(name + '_data', param, epoch)

                # test
                if self.args.using_lp:
                    _, train_pred_np = test(self.model, train_features, train_partial_labels, self.device, is_eval=False, args=self.args)

                # evaluation
                if epoch % self.show_epoch == 0 and step == 0 and self.args.is_test_in_train:
                    metrics_results, _ = test(self.model, test_features, test_labels, self.device, is_eval=True, args=self.args)

                    # draw figure to find best epoch number
                    for i, key in enumerate(metrics_results):
                        print(f"{key}: {metrics_results[key]:.4f}", end='\t')
                        self.writer.add_scalar(f"Metrics/{key}", metrics_results[key], global_step=epoch)
                        loss_list[epoch][key] = metrics_results[key]
                    print("\n")

                    if best_F1 < metrics_results['micro_f1']:
                        best_F1, best_epoch = metrics_results['micro_f1'], epoch

        print(f"best_F1: {best_F1}, epoch {best_epoch}")
        self.writer.close()
        return loss_list

    def pretrain(self, train_loader, class_num):

        criterion = Loss(self.args.batch_size, class_num, self.args.temperature_f, self.args.temperature_l,
                         self.args, self.device).to(self.device)

        for epoch in range(self.pretrain_epochs):
            self.model.train()
            tot_loss = 0.

            for step, (inputs, labels, index) in enumerate(train_loader):
                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)
                labels = labels.to(self.device)
                outputs, hs, xrs, zs = self.model(inputs, labels)

                loss_list = []
                for v in range(self.model.view):
                    for w in range(v + 1, self.model.view):
                        loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                        # loss_list.append(criterion.forward_label(qs[v], qs[w]))
                    # loss_list.append(mes(xs[v], xrs[v]))

                loss = sum(loss_list)
                self.opti.zero_grad()
                loss.backward()
                self.opti.step()
                tot_loss += loss.item()

                if epoch % 50 == 0 and step == 0:
                    self.writer.add_scalar("Loss/CL Loss", loss.item(), global_step=epoch)
                                    #     plot_embedding(hs[0], hs[1], labels)
                #     self.writer.add_embedding(mat=hs[0], metadata=labels, global_step=epoch)
                #     self.writer.add_embedding(mat=hs[1], metadata=labels, global_step=epoch)

            print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(train_loader)))
        self.writer.close()

if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)
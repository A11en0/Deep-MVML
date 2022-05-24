# -*- coding: UTF-8 -*-
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.Disambiguation import estimating_label_correlation_matrix, build_graph, label_propagation
from utils.draw import plot_embedding
from utils.loss import Loss, LinkPredictionLoss_cosine, cross_modal_contrastive_ctriterion
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
        outputs, _, _ = model(features, labels)
        outputs = outputs.sigmoid_()

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

    def fit(self, train_loader, train_features, train_labels, test_features, test_labels, class_num, fold):
        loss_list = []
        best_F1, best_epoch = 0.0, 0.0

        # Defining loss function
        criterion = nn.MultiLabelSoftMarginLoss()
        # criterion = F.binary_cross_entropy_with_logits()
        emb_criterion = LinkPredictionLoss_cosine()
        CL = Loss(self.args.batch_size, class_num, self.args.temperature_f, self.args.temperature_l,
                         self.args, self.device).to(self.device)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Generating the adjacency matrix for the label semantic encoding module
        self.model.label_adj.data = self.sym_conditional_prob(train_labels).to(self.device)

        # Adjacency matrix with self-loop
        self.adj = self.model.label_adj.data + torch.eye(self.model.label_adj.data.size(0),
                                                             dtype=self.model.label_adj.data.dtype,
                                                             device=self.model.label_adj.data.device)

        for epoch in range(self.epochs):
            self.model.train()

            for step, (inputs, labels, index) in enumerate(train_loader):

                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)
                labels = labels.to(self.device)
                output, label_emb, hs = self.model(inputs, labels)

                # classification loss
                ml_loss = criterion(output, labels)

                # label graph reconstruction loss
                rec_loss = emb_criterion(label_emb, self.adj)

                # contrastive loss
                cl_loss_list = []
                for v in range(self.model.view):
                    for w in range(v + 1, self.model.view):
                        cl_loss_list.append(CL.info_nce_loss(hs[v], hs[w]))
                cl_loss = sum(cl_loss_list) / len(cl_loss_list)

                loss = self.args.coef_ml * ml_loss + self.args.coef_rec * rec_loss + self.args.coef_cl*cl_loss

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}\t CL Loss: {cl_loss.item():.4f}\t Rec Loss: {rec_loss.item():.4f}'

                # show loss info
                if epoch % self.show_epoch == 0 and step == 0:
                    print(print_str)
                    self.writer.add_scalar("Loss/Loss", loss.item(), global_step=epoch)
                    self.writer.add_scalar("Loss/ML Loss", ml_loss.item(), global_step=epoch)
                    self.writer.add_scalar("Loss/CL Loss", cl_loss.item(), global_step=epoch)
                    self.writer.add_scalar("Loss/Rec Loss", rec_loss.item(), global_step=epoch)
                    epoch_loss = dict()
                    loss_list.append(epoch_loss)

                    # self.writer.add_scalar("Loss/MI Loss", info_loss.item(), global_step=epoch)
                    # self.writer.add_embedding(mat=hs[1], metadata=labels, global_step=epoch)

                if epoch % 50 == 0 and step == 0 and self.args.plot:
                    plot_embedding(hs[0], hs[1], labels)

                self.opti.zero_grad()
                loss.backward()
                self.opti.step()

                # 每一个epoch，记录各层权重、梯度
                # for name, param in self.model.named_parameters():  # 返回网络的
                #     self.writer.add_histogram(name + '_grad', param.grad, epoch)
                #     self.writer.add_histogram(name + '_data', param, epoch)

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

    def sym_conditional_prob(self, y):
        adj = torch.matmul(y.t(), y)
        y_sum = torch.sum(y.t(), dim=1, keepdim=True)
        y_sum[y_sum<1e-6] = 1e-6
        adj = adj / y_sum
        for i in range(adj.size(0)):
            adj[i, i] = 0.0
        adj = (adj + adj.t()) * 0.5
        return adj

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
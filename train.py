# -*- coding: UTF-8 -*-
import os
from functools import reduce

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.loss import Loss
from utils.ml_metrics import all_metrics


@ torch.no_grad()
def test(model, classifier, features, labels, weight_var, device, model_state_path=None, is_eval=False):
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
        features, _, _, _ = model(features, labels)
        features = torch.cat(features, dim=1)
        outputs = classifier(features)
        # for v in range(len(feat_outs)):
        #     loss_temp = criterion(feat_outs[v], labels)
        #     loss += (weight_var[v] ** gamma) * loss_temp

        # output_var = torch.stack(feat_outs).to(device)
        # weight_var = weight_var.unsqueeze(1)
        # weight_var = weight_var.unsqueeze(2)
        # weight_var = weight_var.expand(weight_var.size(0), output_var.shape[1], output_var.shape[2])
        # output_weighted = weight_var * output_var
        # outputs = torch.sum(output_weighted, 0)

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
    def __init__(self, model, classifier, args, device):
        self.model = model
        self.classifier = classifier
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

                #     adjust_learning_rate(epoch, args, optimizer)

                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)

                labels = labels.to(self.device)

                feat_embs, latent_embs, _, _ = self.model(inputs, labels)

                feats = torch.cat(feat_embs, dim=1)

                outputs = self.classifier(feats)

                loss = F.binary_cross_entropy(outputs, labels)

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}'

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
                    metrics_results, _ = test(self.model, self.classifier, test_features, test_labels, None, self.device, is_eval=True)

                    # draw figure to find best epoch number
                    for i, key in enumerate(metrics_results):
                        print(f"{key}: {metrics_results[key]:.4f}", end='\t')
                        loss_list[epoch][key] = metrics_results[key]
                    print("\n")

                    if best_F1 < metrics_results['micro_f1']:
                        best_F1, best_epoch = metrics_results['micro_f1'], epoch

            # save model
            # if epoch != 0 and epoch % self.model_save_epoch == 0:
            #     print('==> Saving...')
            #     state = {
            #         'opt': self.args,
            #         'model': self.model.state_dict(),
            #         'optimizer': self.opti.state_dict(),
            #         # 'contrast': contrast.state_dict(),
            #         'epoch': epoch,
            #     }
            #     save_file = os.path.join(self.args.model_save_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            #     torch.save(state, save_file)

                # if (epoch + 1) % self.model_save_epoch == 0:
                #     torch.save(self.model.state_dict(),
                #             os.path.join(self.model_save_dir,
                #                          'fold' + str(fold)+'_' + 'epoch' + str(epoch + 1) + '.pth'))

        writer.flush()
        writer.close()

        return loss_list

    def pretrain(self, train_loader, class_num):
        loss_list = []
        best_F1, best_epoch = 0.0, 0.0
        writer = SummaryWriter()
        criterion = Loss(self.args.batch_size, class_num, self.args.temperature_f, self.args.temperature_l,
                         self.args, self.device).to(self.device)

        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        for epoch in range(self.epochs):
            self.model.train()
            for step, (inputs, labels, index) in enumerate(train_loader):

                #     adjust_learning_rate(epoch, args, optimizer)

                for i, _ in enumerate(inputs):
                    inputs[i] = inputs[i].to(self.device)

                labels = labels.to(self.device)

                feat_embs, hs, zs, _ = self.model(inputs, labels)

                # contrastive loss
                _cl_loss = []
                for v in range(self.model.view):
                    for w in range(v + 1, self.model.view):
                        # _cl_loss.append(criterion.info_nce_loss(latent_embs[v], latent_embs[w]))
                        # _cl_loss.append(criterion.info_nce_loss(feat_embs[v], feat_embs[w]))
                        _cl_loss.append(criterion.info_nce_loss(hs[v], hs[w]))
                    # _cl_loss.append(mse(xs[v], xrs[v]))

                loss = sum(_cl_loss)

                print_str = f'Epoch: {epoch}\t Loss: {loss.item():.4f}'

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
                # if epoch % self.show_epoch == 0 and step == 0 and self.args.is_test_in_train and False:
                #     metrics_results, _ = test(self.model, test_features, test_labels, weight_var, self.device, is_eval=True)
                #
                #     # draw figure to find best epoch number
                #     for i, key in enumerate(metrics_results):
                #         print(f"{key}: {metrics_results[key]:.4f}", end='\t')
                #         loss_list[epoch][key] = metrics_results[key]
                #     print("\n")
                #
                #     if best_F1 < metrics_results['micro_f1']:
                #         best_F1, best_epoch = metrics_results['micro_f1'], epoch

            # save model
            if epoch != 0 and epoch % self.model_save_epoch == 0:
                print('==> Saving...')
                state = {
                    'opt': self.args,
                    'model': self.model.state_dict(),
                    'optimizer': self.opti.state_dict(),
                    # 'contrast': contrast.state_dict(),
                    'epoch': epoch,
                }

                save_file = os.path.join(self.args.model_save_dir, '{dataname}-ckpt_epoch_{epoch}.pth'.format(dataname=self.args.DATA_SET_NAME, epoch=epoch))
                torch.save(state, save_file)

                # if (epoch + 1) % self.model_save_epoch == 0:
                #     torch.save(self.model.state_dict(),
                #             os.path.join(self.model_save_dir,
                #                          'fold' + str(fold)+'_' + 'epoch' + str(epoch + 1) + '.pth'))

        writer.flush()
        writer.close()

        return loss_list


if __name__ == '__main__':
    f1 = torch.randn(1000, 100)
    f2 = torch.randn(1000, 100)
    train_features = {0: f1, 1: f2}
    train_labels = torch.randn(1000, 14)
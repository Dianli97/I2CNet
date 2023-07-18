# -*- coding: utf-8 -*-
"""
# @file name  : tools.py
# @author     : Peiji, Chen
# @date       : 2022/08/14
# @brief      :
"""
# -*- coding: utf-8 -*-
import numpy as np
import os
import torch
import random
import torch.nn as nn
from torch.nn import functional as F


def seed_everything(seed):
    """
    固定各类随机种子，方便消融实验
    """

    # 固定 scipy 的随机种子
    random.seed(seed)  # 固定 random 库的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 固定 python hash 的随机性（并不一定有效）
    np.random.seed(seed)  # 固定 numpy  的随机种子
    torch.manual_seed(seed)  # 固定 torch cpu 计算的随机种子
    torch.cuda.manual_seed(seed)  # 固定 cuda 计算的随机种子
    torch.backends.cudnn.deterministic = True  # 是否将卷积算子的计算实现固定。torch 的底层有不同的库来实现卷积算子
    torch.backends.cudnn.benchmark = True  # 是否开启自动优化，选择最快的卷积计算方法


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, factor=0.5, k=1):
        model.train()

        conf_mat = np.zeros((52, 52))
        correct_ = 0
        total_ = 0
        loss_sigma = []

        for idx, data in enumerate(data_loader):
            inputs, labels = data
            # print(inputs.shape)
            if inputs.ndim == 4:
                inputs = inputs.view(-1, 10, 20)
                labels = labels.view(-1, )

            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)

            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = torch.squeeze(inputs, dim=1)

            outputs, _, _ = model(inputs)
            # print(torch.softmax(outputs, dim=1)[0])

            optimizer.zero_grad()
            loss_1 = loss_f(outputs, labels)
            loss_2 = loss_f(outputs, labels)
            loss = factor * loss_1 + (1 - factor) * loss_2
            loss.backward()
            optimizer.step()

            # 计算Top 1准确率
            # _, predicted = torch.max(outputs.data, 1)

            # 计算Top K准确率
            _, predicted_topk = torch.sort(outputs.data, descending=True)
            predicted_topk = predicted_topk.cpu().numpy()
            predicted_topk = predicted_topk[:, :k]

            for j in range(len(labels)):
                true_label = labels[j].cpu().numpy()
                predicted_label = predicted_topk[j]
                for i in range(len(predicted_label)):
                    if true_label == predicted_label[i]:
                        correct_ += 1
                        break

            total_ += labels.shape[0]

            # for j in range(len(labels)):
            #     cate_i = labels[j].cpu().numpy()
            #     pre_i = predicted[j].cpu().numpy()
            #     conf_mat[cate_i, pre_i] += 1.

            # 统计loss值
            loss_sigma.append(loss.item())
            # acc_avg = conf_mat.trace() / conf_mat.sum()
            acc_avg = correct_ / total_

            # 每50个iteration 打印一次训练信息, loss为50个iteration的均值
            if idx % 50 == 50 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%}".format(
                    epoch_id + 1, max_epoch, idx + 1, len(data_loader), np.mean(loss_sigma), acc_avg
                ))

        return np.mean(loss_sigma), acc_avg

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()

        conf_mat = np.zeros((52, 52))
        correct_ = 0
        total_ = 0
        loss_sigma = []

        for idx, data in enumerate(data_loader):

            inputs, labels = data
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)

            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = torch.squeeze(inputs, dim=1)

            outputs, _, _ = model(inputs)
            loss = loss_f(outputs, labels)

            # # 统计预测值
            # _, predicted = torch.max(outputs.data, 1)
            #
            # for j in range(len(labels)):
            #     cate_i = labels[j].cpu().numpy()
            #     pre_i = predicted[j].cpu().numpy()
            #     conf_mat[cate_i, pre_i] += 1.

            # # 计算Top K准确率
            _, predicted_topk = torch.sort(outputs.data, descending=True)
            predicted_topk = predicted_topk.cpu().numpy()
            predicted_top3 = predicted_topk[:, :1]

            for j in range(len(labels)):
                true_label = labels[j].cpu().numpy()
                predicted_label = predicted_top3[j]
                for i in range(len(predicted_label)):
                    if true_label == predicted_label[i]:
                        correct_ += 1
                        break

            total_ += labels.shape[0]
            # 统计loss值t
            loss_sigma.append(loss.item())

        # acc_avg = conf_mat.trace() / conf_mat.sum()
        acc_avg = correct_ / total_

        return np.mean(loss_sigma), acc_avg

    @staticmethod
    def train_dla(data_loader, model1, model2, ce_loss, kl_loss1, kl_loss2, optimizer1, optimizer2, epoch_id, device,
                  max_epoch, writer, factor=0.5, prob=0.85):
        model1.train()
        model2.train()

        conf_mat = np.zeros((52, 52))
        loss_sigma = []
        pre_pen_res = []
        penultimate_res = []
        target = []
        total_ = 0
        correct_ = 0

        for idx, data in enumerate(data_loader):
            inputs, labels = data

            if inputs.ndim == 4:
                inputs = inputs.view(-1, 10, 20)
                labels = labels.view(-1, )

            # 准备数据
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)
            y_oneHot = ModelTrainer.onehot(labels.shape[0], labels, classes=52)
            inputs, labels, y_oneHot = inputs.to(device), labels.to(device), y_oneHot.to(device)

            # ----------------
            # train main model
            # ----------------

            outputs, extracted_features, _ = model1(inputs)
            adjusted_label = model2(extracted_features, y_oneHot)

            optimizer1.zero_grad()
            loss_1 = ce_loss(outputs, labels)
            loss_2 = kl_loss1(F.log_softmax(outputs, -1), adjusted_label)
            loss = (1 - factor) * loss_1 + factor * loss_2

            loss.backward(retain_graph=True)
            optimizer1.step()

            # ----------------
            # train auxiliary model2
            # ----------------
            adjusted_label1 = model2(extracted_features, y_oneHot)
            _, extracted_features, embedded_out = model1(inputs)

            optimizer2.zero_grad()
            y_adj_true = ModelTrainer.area(extracted_features, y_oneHot, max_prob=prob)
            loss_3 = kl_loss2(torch.log(adjusted_label1), y_adj_true)
            loss_3.backward(inputs=list(model2.parameters()))

            optimizer2.step()

            # 统计预测值
            _, predicted = torch.max(outputs.data, 1)

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # # 计算Top K准确率
            # _, predicted_topk = torch.sort(outputs.data, descending=True)
            # predicted_topk = predicted_topk.cpu().numpy()
            # predicted_top3 = predicted_topk[:, :3]
            #
            # for j in range(len(labels)):
            #     true_label = labels[j].cpu().numpy()
            #     predicted_label = predicted_top3[j]
            #     for i in range(len(predicted_label)):
            #         if true_label == predicted_label[i]:
            #             correct_ += 1
            #             break
            #
            # total_ += labels.shape[0]
            # 统计loss值
            loss_sigma.append(loss.item())
            acc_avg = conf_mat.trace() / conf_mat.sum()
            # acc_avg = correct_ / total_

            # 每50个iteration 打印一次训练信息, loss为50个iteration的均值
            if idx % 50 == 50 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc: {:.2%}".format(
                    epoch_id + 1, max_epoch, idx + 1, len(data_loader), np.mean(loss_sigma), acc_avg
                ))

            # pre_pen_res.append(extracted_features.cpu().detach().numpy())
            # penultimate_res.append(embedded_out.cpu().detach().numpy())
            # target.append(labels.cpu().detach().numpy())

        # pre_pen_res = np.concatenate([i for i in pre_pen_res[:-1]], axis=0)
        # penultimate_res = np.concatenate([i for i in penultimate_res[:-1]], axis=0)
        # target = np.concatenate([i for i in target[:-1]])

       # writer.add_scalar('loss/train', np.mean(loss_sigma), epoch_id)
       #  writer.add_scalar('accuracy/train', acc_avg, epoch_id)

        # if epoch_id == 200:
        #     writer.add_embedding(pre_pen_res[:, :], metadata=target[:], tag='extracted feature')
        #     writer.add_embedding(penultimate_res[:, :], metadata=target[:], tag='penultimate layer')

        return np.mean(loss_sigma), acc_avg

    @staticmethod
    def valid_dla(data_loader, model1, model2, loss1, loss2, device, writer, epoch_id, factor=0.5):
        model1.eval()
        model2.eval()

        conf_mat = np.zeros((52, 52))
        loss_sigma = []
        correct_ = 0
        total_ = 0

        for idx, data in enumerate(data_loader):

            inputs, labels = data

            # 准备数据
            inputs = inputs.to(torch.float32)
            labels = labels.type(torch.LongTensor)
            y_oneHot = ModelTrainer.onehot(labels.shape[0], labels, classes=52)
            inputs, labels, y_oneHot = inputs.to(device), labels.to(device), y_oneHot.to(device)

            outputs, extracted_features, _ = model1(inputs)
            adjusted_label = model2(extracted_features, y_oneHot)

            loss_1 = loss1(outputs, labels)
            loss_2 = loss2(F.log_softmax(outputs, -1), adjusted_label)
            loss = factor * loss_1 + (1 - factor) * loss_2

            # 统计预测值
            _, predicted = torch.max(outputs.data, 1)

            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            # 计算Top K准确率
            # _, predicted_topk = torch.sort(outputs.data, descending=True)
            # predicted_topk = predicted_topk.cpu().numpy()
            # predicted_top3 = predicted_topk[:, :3]
            #
            # for j in range(len(labels)):
            #     true_label = labels[j].cpu().numpy()
            #     predicted_label = predicted_top3[j]
            #     for i in range(len(predicted_label)):
            #         if true_label == predicted_label[i]:
            #             correct_ += 1
            #             break
            #
            # total_ += labels.shape[0]
            # 统计loss值
            loss_sigma.append(loss.item())
            # acc_avg = conf_mat.trace() / conf_mat.sum()
        # acc_avg = correct_ / total_

        acc_avg = conf_mat.trace() / conf_mat.sum()

        # writer.add_scalar('loss/test', np.mean(loss_sigma), epoch_id)
        # writer.add_scalar('accuracy/test', acc_avg, epoch_id)

        return np.mean(loss_sigma), acc_avg

    @staticmethod
    def area(X, y, max_prob=0.9):
        factor = max_prob / (1 - max_prob)
        zero = torch.zeros_like(y)
        one = torch.ones_like(y)
        y_inverse = torch.where(y == 0, one, zero)
        X = y_inverse * X
        X_max = torch.log(factor * torch.sum(torch.exp(X), dim=1)).unsqueeze(-1)
        out = torch.add(X_max * y, X)
        out = F.softmax(out, dim=1)
        return out

    @staticmethod
    def onehot(b, y, classes=52):
        y = y.unsqueeze(-1)
        y = torch.zeros(b, classes).scatter_(1, y, 1)
        return y


class LabelSmooth(nn.Module):

    def __init__(self, num_classes, alpha):
        super(LabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.softmax = torch.nn.Softmax(dim=1)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.alpha = alpha
        self.device = 'cuda'

    # Cross Entropy Loss
    # def forward(self, output, target):
    #     probs = self.softmax(output)
    #     label_one_hot = F.one_hot(target, self.num_classes).float().to(self.device)
    #     label_one_hot = label_one_hot * (1 - self.alpha) + self.alpha / float(self.num_classes)
    #     loss = torch.sum(-label_one_hot * torch.log(probs), dim=1).mean()
    #     return loss

    # KL Loss
    def forward(self, output, target):
        probs = self.softmax(output)
        label_one_hot = F.one_hot(target, self.num_classes).float().to(self.device)
        label_one_hot = label_one_hot * (1 - self.alpha) + self.alpha / float(self.num_classes)
        loss = self.kl_loss(torch.log(probs), label_one_hot)
        return loss

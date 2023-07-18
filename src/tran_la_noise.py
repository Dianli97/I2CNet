# -*- coding: utf-8 -*-
"""
# @file name  : train.py
# @author     : Peiji, Chen
# @date       : 2022/08/15
# @brief      :
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from models.new_model import *
from tools.tools import *
from utils.utils import NinaProDataset
from utils.augmentations import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    seed_everything(42)
    # 93.84%  93.90% (sigma=0.6)
    BATCH_SIZE = 64
    LR = 0.0015
    MAX_EPOCH = 1000

    milestones = [20, 40, 60, 80]

    torch.cuda.set_device(0)
    # ======================== step 1/5 Data ==============================

    train_transform = transforms.Compose([
        uLawNormalization(p=1, u=256),
        GaussianNoise(p=0.5, SNR=25),
        TimeWarping(p=0.5, sigma=0.6, wSize=20, channels=10, keep=False)
    ])

    valid_transform = transforms.Compose([
        uLawNormalization(p=1, u=256),
        GaussianNoise(p=0.5, SNR=25),
    ])

    # 构建Dataset实例
    train_data = NinaProDataset(DB_id='1', mode='train', transforms=train_transform, subjectID='S22')  # S1 S7 S10 S20 (89%) S21(91.06%) S22 (93.9%) S24 (88.8%)  S25 (89.9%) S26 (90.78%)
    test_data = NinaProDataset(DB_id='1', mode='test', transforms=valid_transform, subjectID='S22')

    # 构建DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=36)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=36)

    # ======================== step 2/5 Model ==============================
    emg_model = emgmodel18()
    emg_model.to(device)

    # ======================== step 3/5 Loss function ==============================
    main_criterion = LabelSmooth(num_classes=52, alpha=0.1)
    ce_loss = nn.CrossEntropyLoss()

    # ======================== step 4/5 Optimizers ==============================
    main_optimizer = optim.AdamW(emg_model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(main_optimizer, gamma=0.8, milestones=milestones)

    # ======================== step 5/5 Train ==============================
    loss_rec = {"trian": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(MAX_EPOCH):
        # train
        loss_train, acc_train = ModelTrainer.train(train_loader, emg_model, main_criterion, ce_loss, main_optimizer, epoch, device, MAX_EPOCH, factor=1.0)
        # loss_train, acc_train = ModelTrainer.train(train_loader, emg_model, dla_model, main_criterion,  main_optimizer, epoch, device, MAX_EPOCH)
        loss_val, acc_valid = ModelTrainer.valid(test_loader, emg_model, main_criterion, device)

        if acc_valid > best_acc:
            best_acc = acc_valid
            #torch.save(emg_model, 'trained/' + 'model_la_' + str(epoch) + '_' + str(round(best_acc, 4)) + '.pkl')

        print("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc: {:.2%} Train loss:{:.4f} Valid loss:{:.4f} Best Result: {:.2%}".format(
            epoch+1, MAX_EPOCH, acc_train, acc_valid, loss_train, loss_val, best_acc
        ))

        scheduler.step()



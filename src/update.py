#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import LambdaLR
import math
import time
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters

local_lr_adj = 0

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger, lrs, gtl):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.device = 'cuda:1' if args.gpu else 'cpu'
        # Default criterion set to CE loss function
        # self.criterion = nn.NLLLoss().to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.lrs = lrs

        self.gtl = gtl

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        '''validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)'''
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=5, shuffle=False)
        '''testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)'''
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=5, shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round, idx):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.args.ad_lr:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lrs[idx],
                                        momentum=0.5)
                lr_new = self.lrs[idx]

            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lrs[idx],
                                         weight_decay=0)
                lr_new = self.lrs[idx]
        else:
            if self.args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)

            elif self.args.optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)
        
        last_beita = 1
        last_gradient = self.gtl
        beita = 1
        lambdaDSA = lambda i: beita
        scheduler = LambdaLR(optimizer, lr_lambda = lambdaDSA)
        
        for iter in range(self.args.local_ep):
            batch_loss = []

            xxx = 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                last_para = parameters_to_vector(model.parameters())
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()

                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                # FedHyper optimization
                if self.args.function == "FedHyper-CL" or self.args.function == "FedHyper-FULL":
                    if xxx == 0:
                        xxx = 1
                    elif xxx == 1:
                        new_para = parameters_to_vector(model.parameters())
                        new_gradient = new_para - last_para
                        x = 0.1 * (new_gradient * self.gtl).sum().item() / (self.args.num_users * self.args.frac) # global model update
                        y = (new_gradient * last_gradient).sum().item()

                        gamma = 1
                        gamma += x + y
                        gamma = min(1.1, gamma)

                        beita = gamma * last_beita

                        #print('multi:', gamma)
                        if beita > 10:
                            beita =  10
                        elif beita < 0.1:
                            beita = 0.1
                        last_beita = beita

                        scheduler.step()  # Schedule Local lr
                        last_gradient = new_gradient  # Update Model Gradient

                '''if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | Local Epoch : {} | [{}/{} ({:.0f}%)] \tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))'''

                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())

        epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), lr_new

    def inference(self, model):
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda:1' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

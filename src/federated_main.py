#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os
import sys
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from resnet import resnet18
from utils import get_dataset, average_weights, exp_details
import math
from torch.nn.utils import parameters_to_vector, vector_to_parameters

Global_alpha = []
if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu_id:
        torch.cuda.set_device(args.gpu_id)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
            help_model1 = CNNFashion_Mnist(args=args).to(device)
            help_model2 = CNNFashion_Mnist(args=args).to(device)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)
            help_model1 = CNNCifar(args=args).to(device)
            help_model2 = CNNCifar(args=args).to(device)
    
    elif args.model == 'resnet':
        global_model = resnet18(10)
        help_model1 = resnet18(10).to(device)
        help_model2 = resnet18(10).to(device)

    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()

    # copy weight
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 1
    train_acc_record = []
    test_acc_record = []
    train_loss_record = []
    val_loss_pre, counter = 0, 0

    local_lr = []
    for i in range(0, args.num_users):
        local_lr.append(args.lr)
    
    origin_vector = parameters_to_vector(global_model.parameters())
    gradient_vector = origin_vector - origin_vector

    d = parameters_to_vector(global_model.parameters()).numel()
    delta = torch.zeros(d).to(device)
    grad_mom = torch.zeros(d).to(device)


    agg = 0 #Initialize the Global Learning Rate

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        local_lr = []

        origin_list = []
        for value in global_model.parameters():
            origin_list.append(value.detach().cpu())

        origin_vector = parameters_to_vector(global_model.parameters())

        for i in range(0, args.num_users):
            local_lr.append(args.lr)
   
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        old_weights = copy.deepcopy(global_weights)

        for idx in idxs_users:
            #print('Client ID:', idx)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, lrs = local_lr, gtl=gradient_vector)
            w, loss, lr_new = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, idx=idx)

            local_lr[idx] = lr_new  #Keep the New Local Learning Rate

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
        
        #print("New Local Learning Rates: ", local_lr)

        # Update Temp Global Model
        global_weights_help = average_weights(local_weights)
        global_model.load_state_dict(global_weights_help)
        
        new_vector = parameters_to_vector(global_model.parameters()) 
        gradient_vector = new_vector - origin_vector # Compute the Global Model Update

        #FedHyper: G, SL and GM(Recommended)
        if args.function == 'FedHyper-G':
            if epoch == 0:
                vector_to_parameters(parameters_to_vector(global_model.parameters()), global_model.parameters())
                Global_alpha.append(1)

            if epoch > 0:
                agg += (last_vector * gradient_vector).sum().item() # Compute Hypergradient
                #agg += 1/(1 + math.exp(-0.1 * (last_vector * gradient_vector).sum().item())) # Compute Normalized Hypergradient (Optional)
                if agg < -0.8:    # Clip
                    agg = -0.8
                elif agg >4:
                    agg = 4
                Global_alpha.append(1 + agg)
                vector_to_parameters(agg * gradient_vector + parameters_to_vector(global_model.parameters()), global_model.parameters())

            last_vector = gradient_vector # Update Model Update
        
        if args.function == 'FedHyper-GM' or args.function == "FedHyper-FULL":
            if epoch == 0:
                vector_to_parameters(parameters_to_vector(global_model.parameters()), global_model.parameters())
                Global_alpha.append(1)

                last_vector = gradient_vector

            if epoch > 0:
                agg += (last_vector * gradient_vector).sum().item()
                #agg += 1/(1 + math.exp(-0.1 * (last_vector * gradient_vector).sum().item()))
                if agg < -0.2:
                    agg = -0.2
                elif agg > 4:
                    agg = 4
                Global_alpha.append(1 + agg)

                vector_to_parameters(agg * gradient_vector + 0.9 * agg * last_vector + parameters_to_vector(global_model.parameters()), global_model.parameters())
                last_vector = gradient_vector + 0.9 * last_vector # With Momentumn

        if args.function == 'FedHyper-SL':
            if epoch == 0:
                vector_to_parameters(parameters_to_vector(global_model.parameters()), global_model.parameters())

            if epoch > 0:
                agg += (last_vector * gradient_vector).sum().item()
                #agg += 1/(1 + math.exp(-0.1 * (last_vector * gradient_vector).sum().item()))
                
                for i in range(0, args.num_users):
                    local_lr[i] = min(0.1, max(0.001, args.lr * (agg + 1)))   # Update Local lr by Global Hypergradient

            last_vector = gradient_vector

        #FedAdagrad
        if args.function == 'FedAdagrad':
            old_vector = gradient_vector
            gradient_vector = 0.1*gradient_vector 
            delta += gradient_vector ** 2
            new_gradient_vector = gradient_vector / (torch.sqrt(delta + 0.0316))
            vector_to_parameters(new_gradient_vector - old_vector + parameters_to_vector(global_model.parameters()), global_model.parameters())

        #Decay
        if args.function == 'Decay-G':
            if epoch == 0:
                agg = 1
                vector_to_parameters(parameters_to_vector(global_model.parameters()), global_model.parameters())

            if epoch > 0:
                agg *= 0.995
                vector_to_parameters(agg * gradient_vector - gradient_vector + parameters_to_vector(global_model.parameters()), global_model.parameters())

            last_vector = gradient_vector

        if args.function == 'Decay-SL':
            if epoch == 0:
                agg = 1
                vector_to_parameters(parameters_to_vector(global_model.parameters()), global_model.parameters())

            if epoch > 0:
                agg *= 0.995
                for i in range(0, args.num_users):
                    local_lr[i] = args.lr * agg

            last_vector = gradient_vector

        #FedExp
        if args.function == 'FedExp':
            local_sum = 0
            help_model2.load_state_dict(old_weights)
            for local_p in local_weights:
                help_model1.load_state_dict(local_p)
                local_sum += ((parameters_to_vector(help_model1.parameters()) - parameters_to_vector(help_model2.parameters())) * (parameters_to_vector(help_model1.parameters()) - parameters_to_vector(help_model2.parameters()))).sum().item()

            global_sum = ((parameters_to_vector(global_model.parameters()) - parameters_to_vector(help_model2.parameters())) * (parameters_to_vector(global_model.parameters()) - parameters_to_vector(help_model2.parameters()))).sum().item()
                
            agg = max(1, local_sum/(global_sum * 10 * 2))
            Global_alpha.append(agg)

            vector_to_parameters(agg * gradient_vector - gradient_vector + parameters_to_vector(global_model.parameters()), global_model.parameters())

        #FedAdam
        if args.function == 'FedAdam+FedHyper':
            old_vector = gradient_vector
            if epoch > 0:
                agg += (last_vector * gradient_vector).sum().item()
                if agg < -0.2:
                    agg = -0.2
                elif agg > 4:
                    agg = 4
            gradient_vector = 0.1 * (1 + agg) * gradient_vector 
            gradient_vector = 0.1 * gradient_vector + 0.9 * grad_mom
            grad_mom = gradient_vector

            delta = 0.01*gradient_vector**2 + 0.99*delta

            gradient_vector_normalized = gradient_vector/(0.1)
            delta_normalized = delta/(0.01)

            new_gradient_vector = (gradient_vector_normalized/torch.sqrt(delta_normalized + 0.0316))
            vector_to_parameters(new_gradient_vector - old_vector + parameters_to_vector(global_model.parameters()), global_model.parameters())

            last_vector = old_vector

        if args.function == 'FedAdam':
            old_vector = gradient_vector
            gradient_vector = 0.1*gradient_vector 
            gradient_vector = 0.1*gradient_vector + 0.9*grad_mom
            grad_mom = gradient_vector

            delta = 0.01*gradient_vector**2 + 0.99*delta

            gradient_vector_normalized = gradient_vector/(0.1)
            delta_normalized = delta/(0.01)

            new_gradient_vector = (gradient_vector_normalized/torch.sqrt(delta_normalized + 0.0316))
            vector_to_parameters(new_gradient_vector - old_vector + parameters_to_vector(global_model.parameters()), global_model.parameters())
        
        # compute loss
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, lrs = local_lr, gtl=gradient_vector)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            train_loss_record.append(np.mean(np.array(train_loss)))
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            train_acc_record.append(100*train_accuracy[-1])

            # Test inference after completion of training
            test_acc, test_loss = test_inference(args, global_model, test_dataset)

            print(f' \n Results after this global rounds of training:')
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))
            test_acc_record.append(100*test_acc)

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    print('Training Process:', train_loss_record, train_acc_record)
    print('Test Process:', test_acc_record)
    print('Global Learning Rates:', Global_alpha)

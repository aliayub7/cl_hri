"""
Created on Fri Jun 24 2022

@author: Ali Ayub
"""

import os
import torch
from torch.autograd import Variable
import time
#from tqdm import tqdm
import numpy as np
import random


def train(classify_net,dataloaders_train_classification,optimizer,loss_classify,lambda_based=None,seed=7):
    if lambda_based==True:
        my_lambda = m_lambda
    else:
        my_lambda = 1
    classify_net.train()

    total_loss = []
    for images, labels in dataloaders_train_classification:
        images = Variable(images)
        labels = Variable(labels)
        images = images.cuda()
        optimizer.zero_grad()
        #outputs,_,_ = classify_net(images)

        outputs = classify_net(images)

        images = None
        labels = labels.cuda()
        loss = my_lambda * loss_classify(outputs, labels)
        loss.backward()
        total_loss.append(loss.item())
        optimizer.step()
    return np.average(total_loss)

def eval_training(classify_net,dataloaders_test_classification,loss_classify,seed=7):
    classify_net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    for images, labels in dataloaders_test_classification:
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        #outputs,_,_ = classify_net(images)
        outputs = classify_net(images)
        images = None
        loss = loss_classify(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(dataloaders_test_classification.dataset),
        correct.float() / len(dataloaders_test_classification.dataset)
    ))
    return correct.float() / len(dataloaders_test_classification.dataset)

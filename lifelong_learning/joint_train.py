"""
Created on Thu Jun 23 2022

@author: Ali Ayub
"""

import numpy as np
import time
import os
from PIL import Image
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
from torch.autograd import Variable

from training_functions import train,eval_training
from get_transformed_data import getTransformedData


class JT:
    def __init__(self,prev_classes,path):
        self.path = path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prev_classes = prev_classes

        # hyperparameters
        self.weight_decay = 5e-4
        self.lr = 0.01
        self.max_epochs = 10
        self.batch_size = 32
        self.criterion = nn.CrossEntropyLoss()

        self.model = models.resnet18(pretrained=True)
        self.load_model()
        self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr,
                            momentum=0.9)

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        resolution = 256
        crop = 224

        # define transforms
        self.transforms_classification_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resolution),
            transforms.RandomCrop(crop, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean,imagenet_std)
        ])

        self.transforms_classification_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resolution),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean,imagenet_std)
        ])

    def load_model(self):
        if self.prev_classes!=0:
            prev_exp = len([name for name in os.listdir(self.path) if
                        os.path.isfile(os.path.join(self.path, name))])        # find the length of stuff already in there
            prev_exp = prev_exp/2
            prev_exp = int(prev_exp)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,self.prev_classes)
            self.model.load_state_dict(torch.load(self.path+str(prev_exp-1)+'_model'))
        self.model = self.model.to(self.device)

    def train_model(self,x_train,y_train):
        """
        x_train: training images
        y_train: labels for training images
        """
        total_classes = len(np.unique(y_train)) #+ self.prev_classes

        train_dataset_classification = getTransformedData(x_train,y_train,
                            transform=self.transforms_classification_train)
        dataloaders_train_classification = torch.utils.data.DataLoader(train_dataset_classification,
                            batch_size = self.batch_size,shuffle=True, num_workers = 4)

        test_dataset_classification = getTransformedData(x_train,y_train,
                                transform=self.transforms_classification_test)
        dataloaders_test_classification = torch.utils.data.DataLoader(test_dataset_classification,
                            batch_size=self.batch_size,shuffle=False, num_workers = 4)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,total_classes)
        self.model = self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9)

        for epoch in range(0, self.max_epochs):
            classification_loss = train(self.model,dataloaders_train_classification,
                            self.optimizer,self.criterion)
            print ('Epoch: {}, Loss: {}'.format(epoch,classification_loss))
            eval_training(self.model,dataloaders_test_classification,self.criterion)


        prev_exp = len([name for name in os.listdir(self.path) if
                    os.path.isfile(os.path.join(self.path, name))])        # find the length of stuff already in there
        prev_exp = int(prev_exp/2)
        torch.save(self.model.state_dict(), self.path+str(prev_exp)+'_model')
        np.savetxt(self.path+str(prev_exp)+'_classes.csv',np.array([total_classes]),delimiter=',')

    def get_predicted_labels(self,x_test):
        """
        x_test: test images
        """
        a = []
        for i in range(0,len(x_test)):
            a.append(self.transforms_classification_test(x_test[i]))
        if len(a)>=1:
            x_test = torch.stack(a).to(self.device)

            self.model.eval()
            outputs = self.model(x_test)
            _, labels = outputs.max(1)
            labels = labels.detach().to('cpu')
            labels = labels.numpy()
        else:
            labels = []
            labels = np.array(labels)
        return labels

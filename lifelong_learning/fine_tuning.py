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


class FT:
    def __init__(self,prev_classes,path):
        self.path = path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.prev_classes = prev_classes

        # hyperparameters
        #self.total_classes = total_classes
        self.weight_decay = 5e-4
        self.lr = 0.01
        self.max_epochs = 10
        self.batch_size = 32
        self.criterion = nn.CrossEntropyLoss()

        self.model = models.resnet18(pretrained=True)
        self.load_model()
        #num_ftrs = self.model.fc.in_features
        #self.model.fc = nn.Linear(num_ftrs,self.total_classes)
        #self.model = self.model.to(device)
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
        """
        prev_exp = len([name for name in os.listdir(self.path) if
                    os.path.isfile(os.path.join(self.path, name))])        # find the length of stuff already in there
        if prev_exp != 0: # previously saved model found
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs,prev_classes)
            self.model.load_state_dict(torch.load(self.path+str(prev_exp-1)))

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,self.total_classes)
        self.model = self.model.to(device)
        #self.optimizer = optim.SGD(self.model.parameters(),lr=self.lr,
        #                    momentum=0.9)
        """

    def train_model(self,x_train,y_train):
        """
        x_train: training images
        y_train: labels for training images
        """
        total_classes = len(np.unique(y_train)) + self.prev_classes

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

        """
        test_dataset_classification = getTransformedData(complete_x_test,complete_y_test,
                                transform=self.transforms_classification_test)
        dataloaders_test_classification = torch.utils.data.DataLoader(test_dataset_classification,
                            batch_size=1,shuffle=True, num_workers = 4)
        """























if __name__ == '__main__':
    #set_start_method('spawn',force=True)
    path_to_train = '/home/ali/860Evo/ILSVRC2012_Train'
    path_to_test = '/home/ali/860Evo/ILSVRC2012_Test'

    features_name = 'single_noStlye_65000_64'
    save_data = True

    # class info
    total_classes = 10
    full_classes = 1000
    limiter = 50
    dataset_name = 'imagenet'

    # hyperparameters
    weight_decay = 5e-4
    classify_lr = 0.1
    reconstruction_lr = 0.001
    reconstruction_epochs = 100
    classification_epochs = 70
    batch_size = 128

    # for centroids
    distance_threshold = 1000
    get_covariances = True
    diag_covariances = True
    clustering_type = 'k_means'
    centroids_limit = 50000
    centroid_finder = getCentroids(None,None,total_classes,seed=seed,get_covariances=get_covariances,diag_covariances=diag_covariances,centroids_limit=centroids_limit)

    # autoencoders_set
    net = auto_shallow(total_classes,seed=seed)
    net = net.cuda()
    optimizer_rec = optim.Adam(net.parameters(), lr=reconstruction_lr, weight_decay=weight_decay)
    train_scheduler_rec = optim.lr_scheduler.MultiStepLR(optimizer_rec, milestones=[50], gamma=0.1) #learning rate decay

    #classify_net
    classify_net = resnet18(total_classes)

    # loss functions and optimizers
    #loss_classify = nn.CrossEntropyLoss()
    loss_classify = LSR()
    loss_rec = nn.MSELoss()

    # get incremental data
    incremental_data_creator = getIncrementalData(path_to_train,path_to_test,full_classes=full_classes,seed=seed)
    incremental_data_creator.incremental_data(total_classes=total_classes,limiter=limiter)

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    resolution = 100

    # define transforms
    transforms_classification_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resolution),
        transforms.RandomCrop(resolution, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean,imagenet_std)
    ])
    transforms_classification_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean,imagenet_std)
    ])
    transforms_reconstruction = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resolution),
    transforms.CenterCrop(resolution),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean,imagenet_std)
    ])

    ##### INCREMENTAL PHASE #####
    complete_x_train = []
    complete_y_train = []
    complete_x_test = []
    complete_y_test = []
    complete_centroids = []
    complete_covariances = []
    complete_centroids_num = []
    Accus = []
    full_classes = limiter
    for increment in range(0,int(full_classes/total_classes)):
        print ('This is increment number: ',increment)
        train_images_increment,train_labels_increment,test_images_increment,test_labels_increment = incremental_data_creator.incremental_data_per_increment(increment)
        if increment==0:
            previous_images = deepcopy(train_images_increment)
            previous_labels = deepcopy(train_labels_increment)
        else:
            """ Regeneration of pseudo_images """
            """
            # generate pseudo_samples from centroids and covariances
            pack = []
            for i in range(0,len(complete_centroids_num)):
                pack.append([complete_centroids[i],complete_covariances[i],complete_centroids_num[i],i,diag_covariances,total_classes,increment,seed])

            previous_samples,previous_labels = get_pseudoSamples(pack)
            """
            previous_images = []
            previous_labels = []
            for i in range(0,len(complete_centroids)):
                temp = complete_centroids[i]
                previous_labels.extend([i for x in range(0,len(complete_centroids[i]))])
                temp = np.array(temp)
                temp = torch.from_numpy(temp)
                temp = temp.float()
                temp_images = get_pseudoimages(net,temp,class_number=i,seed=seed)
                temp_images = list(temp_images)
                previous_images.extend(temp_images)
            """
            previous_samples = []
            previous_labels = []
            for i in range(0,len(complete_centroids)):
                previous_samples.extend(complete_centroids[i])
                previous_labels.extend([i for x in range(0,len(complete_centroids[i]))])

            previous_samples = np.array(previous_samples)
            previous_samples = torch.from_numpy(previous_samples)
            previous_samples = previous_samples.float()
            previous_images = get_pseudoimages(net,previous_samples)
            previous_images = list(previous_images)
            """
            print ('actual previous images',np.array(previous_images).shape)
            print ('previous labels',np.array(previous_labels).shape)
            # append new images
            previous_images.extend(train_images_increment)
            previous_labels.extend(train_labels_increment)
        print ('train images',np.array(previous_images).shape)
        print ('train labels',np.array(previous_labels).shape)

        complete_x_test.extend(test_images_increment)
        complete_y_test.extend(test_labels_increment)
        #x_train,x_test,y_train,y_test = train_test_split(previous_images,previous_labels,test_size=0.2,stratify=previous_labels)
        x_train = previous_images
        y_train = previous_labels
        x_test = complete_x_test
        y_test = complete_y_test
        # classifier training
        train_dataset_classification = getTransformedData(x_train,y_train,transform=transforms_classification_train,seed=seed)
        val_dataset_classification = getTransformedData(x_test,y_test,transform=transforms_classification_test,seed=seed)
        test_dataset_classification = getTransformedData(complete_x_test,complete_y_test,transform=transforms_classification_test,seed=seed)

        dataloaders_train_classification = torch.utils.data.DataLoader(train_dataset_classification,batch_size = batch_size,
        shuffle=True, num_workers = 4)
        dataloaders_test_classification = torch.utils.data.DataLoader(test_dataset_classification,batch_size = batch_size,
        shuffle=True, num_workers = 4)
        dataloaders_val_classification = torch.utils.data.DataLoader(val_dataset_classification,batch_size = batch_size,
        shuffle=True, num_workers = 4)

        classify_net.fc = nn.Linear(512,total_classes+(total_classes*increment))
        optimizer = optim.SGD(classify_net.parameters(),lr=classify_lr,weight_decay=weight_decay,momentum=0.9)
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
        classify_net = classify_net.cuda()
        if increment>0:
            classification_epochs = 40
            train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[37], gamma=0.1) #learning rate decay
        if increment <0:
            classify_net.load_state_dict(torch.load("./checkpoint/"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name+str(resolution)))
            epoch_acc = eval_training(classify_net,dataloaders_test_classification,loss_classify,seed=seed)
            print ('test_acc',epoch_acc)
        else:
            since=time.time()
            best_acc = 0.0
            for epoch in range(0, classification_epochs):
                train_scheduler.step(epoch)
                classification_loss = train(classify_net,dataloaders_train_classification,optimizer,loss_classify,lambda_based=None,seed=seed)
                print ('epoch:', epoch, '  classification loss:', classification_loss, '  learning rate:', optimizer.param_groups[0]['lr'])
                #epoch_acc = eval_training(classify_net,dataloaders_test_classification,loss_classify,seed=seed)
                epoch_acc = eval_training(classify_net,dataloaders_val_classification,loss_classify,seed=seed)
                if epoch_acc>=best_acc:
                    best_acc = epoch_acc
                    best_model_wts = deepcopy(classify_net.state_dict())
                print (' ')
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print ('best_acc',best_acc)
            classify_net.load_state_dict(best_model_wts)
            epoch_acc = eval_training(classify_net,dataloaders_test_classification,loss_classify,seed=seed)
            print ('test_acc',epoch_acc)
            Accus.append(epoch_acc.cpu().numpy().tolist())
            torch.save(best_model_wts, "./checkpoint/"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name+str(resolution))

        ### Train the autoencoder ###
        train_dataset_reconstruction = getTransformedData(train_images_increment,train_labels_increment,
        transform=transforms_reconstruction,seed=seed)
        test_dataset_reconstruction = getTransformedData(test_images_increment,test_labels_increment,transform=transforms_reconstruction,seed=seed)

        dataloaders_train_reconstruction = torch.utils.data.DataLoader(train_dataset_reconstruction,batch_size = batch_size,
        shuffle=True, num_workers = 4)
        dataloaders_test_reconstruction = torch.utils.data.DataLoader(test_dataset_reconstruction,batch_size = batch_size,
        shuffle=True, num_workers = 4)
        for_embeddings_dataloader = torch.utils.data.DataLoader(train_dataset_reconstruction,batch_size = batch_size,
        shuffle=False, num_workers = 4)

        if increment ==-1:
            net.load_state_dict(torch.load("./checkpoint/autoencoder_"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name))
        elif increment == -1:
            net.load_state_dict(torch.load("./checkpoint/autoencoder_"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name))
        else:

            since=time.time()
            best_loss = 100.0
            for epoch in range(1, reconstruction_epochs):
                train_scheduler_rec.step(epoch)
                #reconstruction_loss = train_reconstruction(net,dataloaders_train_reconstruction,optimizer_rec,loss_rec,lambda_based=True,classify_net=classify_net)
                reconstruction_loss = train_reconstruction(net,dataloaders_train_reconstruction,optimizer_rec,loss_rec,seed=seed,epoch=epoch)
                print ('epoch:', epoch, ' reconstruction loss:', reconstruction_loss)
                test_loss = eval_reconstruction(net,dataloaders_test_reconstruction,loss_rec,seed=seed)
                if test_loss<=best_loss:
                    best_loss = test_loss
                    best_model_wts = deepcopy(net.state_dict())
            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print (' ')
            net.load_state_dict(best_model_wts)
            torch.save(best_model_wts, "./checkpoint/autoencoder_"+str(total_classes+(increment*total_classes))+"classes_"+dataset_name+str(resolution))

        # get embeddings from the trained autoencoder
        embeddings = get_embeddings(net,for_embeddings_dataloader,total_classes,seed=seed,increment=increment)
        print ('embeddings',np.array(embeddings).shape)
        """
        # initialize the centroid finder and find centroids
        if increment == 1:
            clustering_type = 'Agglomerative_variant'
            distance_threshold = 0
        else:
            clustering_type = 'k_means'
            distance_threshold = int(centroids_limit/((increment*total_classes)+total_classes))
            #clustering_type = 'Agglomerative_variant'
            #distance_threshold = 0
            print ('Centroids per class',distance_threshold)
        centroid_finder.initialize(None,None,total_classes+(increment*total_classes),increment=0,d_base=distance_threshold,get_covariances=get_covariances,
        diag_covariances=diag_covariances,seed=seed,current_centroids=[],complete_covariances=[],complete_centroids_num=[],clustering_type=clustering_type,
        centroids_limit=centroids_limit)


        centroid_finder.without_validation(embeddings)
        complete_centroids = centroid_finder.complete_centroids
        complete_covariances = centroid_finder.complete_covariances
        complete_centroids_num = centroid_finder.complete_centroids_num
        """

        complete_centroids.extend(embeddings)
        print ('complete centroids',np.array(complete_centroids).shape)

    print (Accus)
    experimental_data = dict()
    experimental_data['seed'] = seed
    experimental_data['acc'] = Accus
    if save_data == True:
        with open('data.json','r') as f:
            data=json.load(f)
        if features_name not in data:
            data[features_name] = dict()
        data[features_name][str(len(data[features_name])+1)] = experimental_data
        with open('data.json', 'w') as fp:
            json.dump(data, fp, indent=4, sort_keys=True)

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 2022

@author: Ali Ayub
"""
# Needed for float division in Python 2.7, not needed in Python 3
from __future__ import division

import numpy as np
from copy import deepcopy
import math

import random
import os
from centroid_finder import CentroidFinder
from fine_tuning import FT
from joint_train import JT
from Functions import load_object_centroids, process_images, save_centroid_data
from Functions import feature_extraction_contexts

class CLModel:
    def __init__(self,model_name='CBCL',path=None):
        """
        model_name: CL model to be used, e.g. CBCL, FT, iCaRL
        path: path to store the trained model
        """
        self.path = path
        self.model_name = model_name
        self.model = None
        if model_name == 'CBCL':
            self.distance_threshold = 16.5
            #self.model = CentroidFinder(distance_threshold=distance_threshold)
        elif model_name == 'FT':
            pass
        elif model_name == 'iCaRL':
            pass
        self.load_model()

    def load_model(self):
        if self.model_name == 'CBCL':
            centroids,centroids_num,total_centroids = load_object_centroids(self.path)
            if len(centroids)!=0:
                self.model = CentroidFinder(dist_thresh=self.distance_threshold,previous_centroids = centroids,
                                    previous_centroids_num=centroids_num,total_centroids=total_centroids,
                                    python3=True)
            else:
                self.model = CentroidFinder(dist_thresh=self.distance_threshold,python3=True)

        elif self.model_name == 'FT':
            prev_exp = len([name for name in os.listdir(self.path) if
                        os.path.isfile(os.path.join(self.path, name))])        # find the length of stuff already in there
            if prev_exp == 0:
                prev_classes = 0
            else:
                prev_exp = prev_exp/2
                prev_exp = int(prev_exp) - 1
                prev_classes = np.genfromtxt(self.path+str(prev_exp)+'_classes.csv',delimiter=',')
                prev_classes = int(prev_classes)

            self.model = FT(prev_classes,self.path)

        elif self.model_name == 'iCaRL':
            prev_exp = len([name for name in os.listdir(self.path) if
                        os.path.isfile(os.path.join(self.path, name))])        # find the length of stuff already in there
            if prev_exp == 0:
                prev_classes = 0
            else:
                prev_exp = prev_exp/2
                prev_exp = int(prev_exp) - 1
                prev_classes = np.genfromtxt(self.path+str(prev_exp)+'_classes.csv',delimiter=',')
                prev_classes = int(prev_classes)

            self.model = JT(prev_classes,self.path)

    def train_model(self,path_to_images,dictionary,session_id=None):

        if self.model_name == 'CBCL':
            if session_id is not None:
                path_to_images = path_to_images+str(session_id)+'/'
            full_features,full_labels,_ = process_images(path_to_images,dictionary)  # extract CNN features for objects
            if len(full_labels)!=0:
                self.model.update_centroids(full_features,full_labels)  # learn new centroids for CBCL
                save_centroid_data(self.path,self.model.centroids,self.model.centroids_num)   # store the learned centroid data
        elif self.model_name == 'FT':
            if session_id is not None:
                path_to_images = path_to_images+str(session_id)+'/'
            _,full_labels,full_images = process_images(path_to_images,dictionary)   # extract cropped images for objects
            if len(full_labels)!=0:
                self.model.train_model(full_images,full_labels)
        elif self.model_name == 'iCaRL':
            if session_id is not None:
                full_images = []
                full_labels = []
                for i in range(0,session_id+1):
                    temp_path = path_to_images+str(i)+'/'
                    _,temp_labels,temp_images = process_images(temp_path,dictionary)   # extract cropped images for objects
                    full_images.extend(temp_images)
                    full_labels.extend(temp_labels)
                full_images = np.array(full_images)
                full_labels = np.array(full_labels)
            else:
                _,full_labels,full_images = process_images(path_to_images,dictionary)   # extract cropped images for objects
            if len(full_labels)!=0:
                self.model.train_model(full_images,full_labels)

    def predict_labels(self,cur_img_dir):
        """
        cur_img_dir: location where the current image is stored
        """
        if self.model_name == 'CBCL':
            features,_ = feature_extraction_contexts(cur_img_dir)
            features = np.array(features)
            if len(features) == 512:
                features = features.reshape(1,512)
            else:
                features = features.reshape(-1,512)
            labels = self.model.get_predicted_labels(np.array(features))
            labels = np.array(labels)
        elif self.model_name == 'FT':
            _,images = feature_extraction_contexts(cur_img_dir)
            #if len(images)>1 and len(images)<100:
            #    images = np.array(images,dtype=object)
            #else:
            #images = np.array(images)
            labels = self.model.get_predicted_labels(images)
        elif self.model_name == 'iCaRL':
            _,images = feature_extraction_contexts(cur_img_dir)
            #if len(images)>1 and len(images)<100:
            #    images = np.array(images,dtype=object)
            #else:
            #    images = np.array(images)
            #images = np.array(images)
            labels = self.model.get_predicted_labels(images)
        return labels

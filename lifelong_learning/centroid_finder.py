# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 2021

@author: Ali Ayub
"""
# Needed for float division in Python 2.7, not needed in Python 3
from __future__ import division

import numpy as np
from copy import deepcopy
import math
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from Functions import get_centroids
from sklearn.model_selection import KFold
import random
# THE FOLLOWING IS DEFINITELY NEEDED WHEN WORKING WITH PYTORCH
import os
os.environ["OMP_NUM_THREADS"] = "1"

class CentroidFinder:
    def __init__(self,previous_centroids = [], previous_centroids_num = [], total_centroids = 0, dist_thresh = 16.5,
                    distance_metric='euclidean',fading_coefficient=0.0,centroids_time=[],
                    weight_saturation=1000.0,locations=None,python3=False, context_object_locations = False,
                    previous_centroids_locations = []):
        self.centroids = previous_centroids
        self.centroids_num = previous_centroids_num
        if context_object_locations:
            self.centroids_locations = previous_centroids_locations
        self.context_object_locations = context_object_locations
        self.centroids_time = centroids_time
        self.dist_thresh = dist_thresh
        self.total_classes = len(self.centroids)
        self.total_centroids = total_centroids
        self.distance_metric = distance_metric
        self.fading_coefficient = fading_coefficient
        self.weight_saturation = weight_saturation
        self.locations = locations
        self.python3 = python3

    def update_centroids(self, x_train, y_train,train_samples=None, x_train_locations=None):
        if train_samples is None:
            train_samples = np.ones(len(y_train))
        else:
            train_samples = np.array(train_samples)

        new_unique_classes = np.unique(y_train)
        #new_unique_classes = new_unique_classes.astype(int)
        more_classes = int(np.max(new_unique_classes) - self.total_classes) + 1

        self.centroids = list(self.centroids)
        self.centroids_num = list(self.centroids_num)
        if self.context_object_locations:
            self.centroids_locations = list(self.centroids_locations)
        if self.python3:
            self.centroids.extend([None for x in range(more_classes)])
            self.centroids_num.extend([None for x in range(more_classes)])
            if self.context_object_locations:
                self.centroids_locations.extend([None for x in range(more_classes)])
        else:
            self.centroids.extend([None for x in xrange(more_classes)])               # xrange is better with Python2.7
            self.centroids_num.extend([None for x in xrange(more_classes)])
            if self.context_object_locations:
                self.centroids_locations.extend([None for x in range(more_classes)])

        # this addition of extra None is needed fo context related centroids. Probably a better way to fix this, but for now.
        self.centroids.extend([None])
        self.centroids_num.extend([None])
        self.centroids = np.array(self.centroids)
        self.centroids_num = np.array(self.centroids_num)
        if self.context_object_locations:
            self.centroids_locations.extend([None])
            self.centroids_locations = np.array(self.centroids_locations)
        # Extra None is now removed and the centroids are of type object.
        self.centroids = np.delete(self.centroids,[len(self.centroids)-1])
        self.centroids_num = np.delete(self.centroids_num,[len(self.centroids_num)-1])
        if self.context_object_locations:
            self.centroids_locations = np.delete(self.centroids_locations,[len(self.centroids_locations)-1])

        centroids_to_be_updated = self.centroids[new_unique_classes]
        centroids_to_be_updated_num = self.centroids_num[new_unique_classes]
        if self.context_object_locations:
            centroids_to_be_updated_locations = self.centroids_locations[new_unique_classes]
        #sorted_y_train = np.argsort(y_train)

        if self.python3:
            train_pack = [None for x in range(len(new_unique_classes))]
        else:
            train_pack = [None for x in xrange(len(new_unique_classes))]
        train_pack = np.array(train_pack)
        for i in range(0,len(new_unique_classes)):
            indices = np.where(y_train == new_unique_classes[i])
            train_data = x_train[indices]
            train_data_samples = train_samples[indices]
            if x_train_locations is not None:
                train_data_locations = x_train_locations[indices]
                train_pack[i] = [train_data,centroids_to_be_updated[i], centroids_to_be_updated_num[i],
                self.dist_thresh,train_data_samples,self.context_object_locations,centroids_to_be_updated_locations[i],
                train_data_locations]
            else:
                train_data_locations = None
                train_pack[i] = [train_data,centroids_to_be_updated[i], centroids_to_be_updated_num[i],
                self.dist_thresh,train_data_samples,self.context_object_locations,[],
                train_data_locations]

        my_pool = Pool(len(new_unique_classes))
        return_pack = my_pool.map(get_centroids,train_pack)
        my_pool.close()

        for i in range(0,len(new_unique_classes)):
            self.centroids[new_unique_classes[i]] = return_pack[i][0]
            self.centroids_num[new_unique_classes[i]] = np.clip(return_pack[i][1],1.0,self.weight_saturation)
            if self.context_object_locations:
                self.centroids_locations[new_unique_classes[i]] = return_pack[i][2]
            #self.total_centroids += len(return_pack[i][0])
        self.centroids = np.array(self.centroids)
        self.centroids_num = np.array(self.centroids_num)
        if self.context_object_locations:
            self.centroids_locations = np.array(self.centroids_locations)
        self.total_classes = len(self.centroids)
        self.total_centroids = 0
        if self.python3:
            for i in range(0,len(self.centroids)):
                if self.centroids[i] is not None:
                    self.total_centroids += len(self.centroids[i])
        else:
            for i in xrange(0,len(self.centroids)):
                if self.centroids[i] is not None:
                    self.total_centroids += len(self.centroids[i])
        """
        for i in range(0,len(self.centroids_num)):
            temp = np.array(self.centroids_num[i])
            temp = temp.reshape(-1,)
            self.total_centroids += np.sum(temp)
        """
    def find_distance(self,x,centroids,single_feature=False,centroids_num_dist=None):
        sizes = list(x.shape)
        sizes.insert(1, 1)
        var = x.reshape(*sizes)
        dist = centroids - var   # distance between centroids and x

        if single_feature:
            if self.python3:
                for i in range(0,len(dist)):
                    dist[i] = dist[i]*x[i]
            else:
                for i in xrange(0,len(dist)):
                    dist[i] = dist[i]*x[i]
        raw_dist = np.linalg.norm(dist,axis=2)   # L1 distance
        if centroids_num_dist is not None:
            raw_dist = raw_dist*centroids_num_dist
        #raw_dist = raw_dist.float()
        return raw_dist

    def get_centroids_labels(self):
        # Could be a better way to convert centroids into the following format.
        # But for now it is just a For loop.
        centroids_dist = []
        centroids_labels = []
        centroids_num_dist = []
        if self.context_object_locations:
            centroids_locations_dist = []
        for i in range(0,len(self.centroids)):
            if self.centroids[i] is not None:
                centroids_dist.extend(self.centroids[i])
                centroids_labels.extend([i for z in range(len(self.centroids[i]))])
                centroids_num_dist.extend(self.centroids_num[i])
                if self.context_object_locations:
                    centroids_locations_dist.extend(self.centroids_locations[i])
        centroids_labels = np.array(centroids_labels)
        if self.context_object_locations:
            return centroids_dist,centroids_labels,centroids_num_dist,centroids_locations_dist
        else:
            return centroids_dist,centroids_labels,centroids_num_dist

    def get_predicted_labels(self,x_test,single_feature=False,raw_dist_return=False,
                            distance_weighting=False):
        """
        x_test [float vector]: test feature vector
        single_feature (bool): if a single feature should be used for prediction. For object finding in a context
        raw_dist_return (bool): if minimum distance from the closest centroids should be returned
        distance_weighting (bool): should cluster weights be used when predicting based on the disatnce
        """
        if self.context_object_locations:
            centroids_dist,centroids_labels,centroids_num_dist,centroids_locations = self.get_centroids_labels()   # get all centroids in list with labels in a separate list
        else:
            centroids_dist,centroids_labels,centroids_num_dist = self.get_centroids_labels()   # get all centroids in list with labels in a separate list
        if distance_weighting==False:
            centroids_num_dist = None
        raw_dist = self.find_distance(x_test,centroids_dist,single_feature,centroids_num_dist=centroids_num_dist)     # distance of each x_test from each centroid
        sorted_indices = np.argmin(raw_dist,axis=1)         # min distance from each centroid
        labels = centroids_labels[sorted_indices]           # predicted label is label of the closest centroid
        if self.context_object_locations:
            centroids_locations = np.array(centroids_locations)
            centroids_dist = np.array(centroids_dist)
            closest_centroid_location = centroids_locations[list(sorted_indices),:]
            raw_dist = raw_dist.reshape(-1,)
            raw_dist = raw_dist[sorted_indices]
        #print (np.min(raw_dist,axis=1))
        if self.context_object_locations:
            if raw_dist_return:
                return labels,closest_centroid_location,raw_dist
            else:
                return labels,closest_centroid_location
        else:
            if raw_dist_return:
                return labels,raw_dist
            else:
                return labels

    def get_accuracy(self,x_test,y_test,single_feature=False):
        labels = self.get_predicted_labels(x_test,single_feature)
        # for mean class accuracy
        if self.python3:
            accu = [0 for x in range(len(self.centroids))]
        else:
            accu = [0 for x in xrange(len(self.centroids))]
        for i in range(0,len(self.centroids)):
            indices = np.where(y_test==i)[0]
            # no test images for the class so it should not affect overall accuracy
            if len(indices) == 0:
                accu[i] = 1.0
            else:
                corrects = np.sum(labels[indices]==y_test[indices])
                accu[i] = corrects/len(y_test[indices])

        # for full accuracy
        #corrects = np.sum(labels==y_test)
        #return corrects/len(y_test)
        return accu

    def get_raw_difference (self,x_test):
        centroids_dist,centroids_labels = self.get_centroids_labels()   # get all centroids in list with labels in a separate list
        # first find distance of each x from each centroid
        sizes = list(x_test.shape)
        sizes.insert(1, 1)
        var = x_test.reshape(*sizes)
        dist = var - centroids_dist   # distance between centroids and x
        # clip distances to zero. If a value is above 0 i.e. positive, that item was not missing
        dist = dist.clip(max=0)

        # Sum the distances of each x from all the contexts. To get stuff missing from each x compared to all the contexts
        sum_per_x = np.zeros((len(dist),len(centroids_dist[0])))
        for i in range(0,len(dist)):
            sum_per_x[i] = np.sum(dist[i],axis=0)
        sum_per_x[sum_per_x>=-0.25] = 0.0       # thresholding: if the missing probability is so low, then we do not care to buy it.

        # as all non-missing items are zero. Simple multiplication of all values will give truly missing object from all the contexts.
        missing_objects = np.ones((len(centroids_dist[0]),))
        if self.python3:
            for i in range(0,len(sum_per_x)):
                missing_objects = missing_objects*sum_per_x[i]
        else:
            for i in xrange(0,len(sum_per_x)):
                missing_objects = missing_objects*sum_per_x[i]

        return missing_objects

    def memory_fading(self):
        """
        PPE MODEL:
        T(n) = sum(w(i)*t(i))
        w(i) = t(i)^(-x)/sum(t(j)^-x)
        x implies weight for events. Higher x means larger weights for recent events
        t(i) is the time since the ith event occured
        Finaly centroid_num = centroid_num^c * T^-d
        d is the decay coefficient
        c is the learning rate
        """
        pass

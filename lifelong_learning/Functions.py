"""
Created on Wed Oct 13 2021

@author: Ali Ayub
"""
# Needed for float division in Python 2.7, not needed in Python 3
from __future__ import division

import numpy as np
import json
import random
from sklearn.cluster import KMeans
from scipy.spatial import distance
from copy import deepcopy
import math
from multiprocessing import Pool
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import fclusterdata
from scipy.cluster.hierarchy import fcluster, ward, average, weighted, complete, single
from scipy.spatial.distance import pdist
import time
import pickle

import cv2
from PIL import Image
import os

from my_nms import non_max_suppression_fast

distance_metric = 'euclidean'
def get_centroids(train_pack):
    # unpack x_train
    x_train = train_pack[0]
    centroids = train_pack[1]
    total_num = train_pack[2]
    distance_threshold = train_pack[3]
    train_data_samples = train_pack[4]
    context_object_locations = train_pack[5]
    centroids_locations = train_pack[6]
    x_train_locations = train_pack[7]

    clustering_type = 'Agglomerative_variant'
    get_covariances = False

    if clustering_type == 'Agglomerative_variant':
        # for each training sample do the same stuff...
        if len(x_train)>0:
            starting_index = 0
            #if len(centroids) == 0:
            if centroids is None or total_num[0] == 0:
                centroids = [x_train[0]]
                #total_num = [1.0]
                total_num = [train_data_samples[0]]
                if context_object_locations and x_train_locations is not None:
                    centroids_locations = [x_train_locations[0]]
                starting_index = 1
            """
            centroids = [[0 for x in range(len(x_train[0]))]]
            for_cov = [[]]

            # initalize centroids
            centroids[0] = x_train[0]
            for_cov[0].append(x_train[0])
            total_num = [1]
            """
            centroids = list(centroids)
            total_num = list(total_num)
            centroids_locations = list(centroids_locations)
            for i in range(starting_index,len(x_train)):
                distances=[]
                indices = []
                temp_x_train = x_train[i].reshape(1,-1)
                distances = find_distance(temp_x_train,centroids)
                distances = distances.reshape(-1,)
                distances[distances>=distance_threshold] = 10000
                if min(distances)>=10000:
                    centroids.append(x_train[i])
                    #total_num.append(1)
                    total_num.append(train_data_samples[i])
                    if context_object_locations:
                        centroids_locations.append(x_train_locations[i])
                else:
                    min_d = np.argmin(distances)
                    #centroids[min_d] = np.add(np.multiply(total_num[min_d],centroids[min_d]),x_train[i])
                    centroids[min_d] = np.add(np.multiply(total_num[min_d],centroids[min_d]),
                                        np.multiply(x_train[i],train_data_samples[i]))
                    if context_object_locations:
                        centroids_locations[min_d] = np.add(np.multiply(total_num[min_d],centroids_locations[min_d]),
                                            np.multiply(x_train_locations[i],train_data_samples[i]))
                    #total_num[min_d]+=1
                    total_num[min_d] = total_num[min_d] + train_data_samples[i]
                    centroids[min_d] = np.divide(centroids[min_d],(total_num[min_d]))
                    if context_object_locations:
                        centroids_locations[min_d] = np.divide(centroids_locations[min_d],(total_num[min_d]))
                """
                for j in range(0,len(centroids)):
                    d = find_distance(x_train[i],centroids[j])
                    if d<distance_threshold:
                        distances.append(d)
                        indices.append(j)

                if len(distances)==0:
                    centroids.append(x_train[i])
                    total_num.append(1)
                    #for_cov.append([])
                    #for_cov[len(for_cov)-1].append(list(x_train[i]))
                else:
                    min_d = np.argmin(distances)
                    centroids[indices[min_d]] = np.add(np.multiply(total_num[indices[min_d]],centroids[indices[min_d]]),x_train[i])
                    total_num[indices[min_d]]+=1
                    centroids[indices[min_d]] = np.divide(centroids[indices[min_d]],(total_num[indices[min_d]]))
                    #for_cov[indices[min_d]].append(list(x_train[i]))
                """
            """
            # calculate covariances
            if get_covariances==True:
                covariances = deepcopy(for_cov)
                for j in range(0,len(for_cov)):
                    if total_num[j]>1:
                        if diag_covariances != True:
                            covariances[j] = np.cov(np.array(for_cov[j]).T)
                        else:
                            temp = np.cov(np.array(for_cov[j]).T)
                            covariances[j] = temp.diagonal()
                    else:
                        covariances[j] = None
                #or j in range(0,len(total_num)):
                #    centroids[j]=np.divide(centroids[j],total_num[j])
            """
        else:
            centroids = []

    elif clustering_type == 'k_means':
        kmeans = KMeans(n_clusters=distance_threshold, random_state = 0).fit(x_train)
        centroids = kmeans.cluster_centers_
    elif clustering_type == 'NCM':
        centroids = [[0 for x in range(len(x_train[0]))]]
        centroids[0] = np.average(x_train,0)

    if get_covariances == True:
        if context_object_locations:
            return [centroids,covariances,total_num,centroids_locations]
        else:
            return [centroids,covariances,total_num]
    else:
        if context_object_locations:
            return [centroids,total_num,centroids_locations]
        else:
            return [centroids,total_num]

def find_distance(x,centroids,centroids_num=None,distance_metric='euclidean'):
    sizes = list(x.shape)
    sizes.insert(1, 1)
    var = x.reshape(*sizes)
    dist = centroids - var   # distance between centroids and x
    raw_dist = np.linalg.norm(dist,axis=2)   # L1 distance
    #raw_dist = raw_dist.float()
    return raw_dist

def load_object_centroids(path):
    """
    path: directory path where the object centroids are stored for CBCL
    """
    prev_exp = len([name for name in os.listdir(path) if
                os.path.isfile(os.path.join(path, name))])        # find the length of stuff already in there
    prev_exp = prev_exp/2
    prev_exp = int(prev_exp)
    if prev_exp>0:
        prev_exp-=1
    centroid_file = path+str(prev_exp)+'_centroids.p'
    centroid_num_file = path+str(prev_exp)+'_centroids_num.p'

    if os.path.exists(centroid_file) and os.path.exists(centroid_num_file):
        f = open(centroid_file,'rb')
        centroids = pickle.load(f)
        f.close()

        f = open(centroid_num_file,'rb')
        centroids_num = pickle.load(f)
        f.close()

        total_centroids = 0
        for k in range(0,len(centroids)):
            if centroids[k] is not None:
                total_centroids += len(centroids[k])
    else:
        centroids = []
        centroids_num = []
        total_centroids = 0
    return centroids, centroids_num, total_centroids

# save centroid_finder centroids in directory
def save_centroid_data(path,centroids,centroids_num):
    """
    path: directory to store centroid data
    centrroids: 2D array containing all the centroids
    centroids_num: 2D array containing the weight for all the centroids
    """
    prev_exp = len([name for name in os.listdir(path) if
                os.path.isfile(os.path.join(path, name))])        # find the length of stuff already in there

    prev_exp = prev_exp/2
    prev_exp = int(prev_exp)
    f = open(path+str(prev_exp)+'_centroids.p','wb')
    pickle.dump(centroids,f)
    f.close()

    f = open(path+str(prev_exp)+'_centroids_num.p','wb')
    pickle.dump(centroids_num,f)
    f.close()

def load_contexts(path_to_contexts,path_to_context_labels,ordered_contexts):
    """
    path_to_contexts: os path to all the contexts learned
    path_to_contexts: os path to the contexts represented in terms of object labels, present in the contexts.
    """
    context_object_labels = []
    context_object_features = []
    context_labels = []
    for i in xrange(0,len(ordered_contexts)):
        path = path_to_context_labels +'/' + ordered_contexts[i]+'_object_labels.csv'
        labels = np.genfromtxt(path,delimiter=',')
        labels = labels.astype(int)
        context_object_labels.append(labels)
        path = path_to_contexts + '/' + ordered_contexts[i] + '_.csv'
        features = np.genfromtxt(path,delimiter=',')
        context_object_features.append(features)
        if 'office' in ordered_contexts[i]:
            context_labels.append(0)
        elif 'kitchen' in ordered_contexts[i]:
            context_labels.append(1)
    return context_object_features,context_labels,context_object_labels

def for_memory_consolidation(centroids,centroids_num):
    centroids_dist = []
    centroids_labels = []
    centroids_num_dist = []
    for i in range(0,len(self.centroids)):
        centroids_dist.extend(self.centroids[i])
        centroids_labels.extend([i for z in range(len(self.centroids[i]))])
        centroids_num_dist.extend(self.centroids_num[i])
    centroids_labels = np.array(centroids_labels)
    centroids_dist = np.array(centroids_dist)
    centroids_num_dist = np.array(centroids_num_dist)
    return centroids_labels,centroids_dist,centroids_num_dist

def feature_extraction_contexts(path_to_CUBS):
    from img_to_vec import Img2Vec
    #path_to_CUBS = '/home/mobilerobot/ali/cognitive_architecture_fetch/my_tasks'

    img2vec = Img2Vec(cuda=False)
    total_features = []
    total_labels = []
    total_images = []
    #import types
    iter = 0

    folder = path_to_CUBS
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder,file)
            #print (file)
            base_file_name = file[:len(file) - 18]
            with open(path) as f:
                full_boxes = f.readlines()
            cropped_images = []
            array_cropped_images = []
            if full_boxes:
                file = base_file_name + "rgb.png"
                path = os.path.join(folder,file)
                #img = Image.open(path)
                img = cv2.imread(path)
                #cv2.imshow('a', img)
                #cv2.waitKey(0)

                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

                # convert bounding_boxes into integer values
                int_boxes = []
                for i in range(0,len(full_boxes)):
                    boxes = full_boxes[i].split(',')
                    inc_percent = 0.05
                    # following is for yolo bbox format
                    bounding_boxes = [int(float(float(boxes[1])-(inc_percent*float(boxes[1])))),
                    int(float(boxes[2])-(inc_percent*float(boxes[2]))),int(float(boxes[3])+(inc_percent*float(boxes[3]))),
                    int(float(boxes[4])+(inc_percent*float(boxes[4])))]

                    bounding_boxes[0] = max(0,bounding_boxes[0])
                    bounding_boxes[1] = max(0,bounding_boxes[1])
                    bounding_boxes[2] = min(img.shape[1],bounding_boxes[2])
                    bounding_boxes[3] = min(img.shape[0],bounding_boxes[3])
                    int_boxes.append(bounding_boxes)
                int_boxes = np.array(int_boxes)
                full_boxes,_ = non_max_suppression_fast(int_boxes,int_boxes,0.5)    #filter overlapping bounding_boxes
                full_boxes = full_boxes.astype(int)
                np.savetxt(folder+'suppressed_boxes.csv',full_boxes,delimiter=',')
                for i in range(0,len(full_boxes)):
                    bounding_boxes = [full_boxes[i][1],full_boxes[i][3],full_boxes[i][0],full_boxes[i][2]]#format for cv2 crop
                    cropped_image = img[bounding_boxes[0]:bounding_boxes[1],
                    bounding_boxes[2]:bounding_boxes[3]]
                    #plt.imshow(cropped_image, cmap='gray')
                    #plt.show()
                    #cv2.imshow('img',cropped_image)
                    #cv2.waitKey(0)
                    array_cropped_images.append(cropped_image)
                    cropped_image = Image.fromarray(cropped_image)
                    cropped_images.append(cropped_image)

                total_images.extend(array_cropped_images)

                vec = img2vec.get_vec(cropped_images)
                if vec is not None:
                    features_np=np.array(vec)
                    if len(features_np)!=512:
                        for k in range(0,len(features_np)):
                            total_features.append(features_np[k])
                    iter+=1

    return total_features,total_images

# get the CNN feature vectors for all the objects in the given folder
def process_images(path,dictionary):

    full_features = []
    full_images = []
    full_labels = []

    if path is not None:
        for folder in os.listdir(path):
            new_path = path+folder+'/'
            print ('new_path', new_path)
            total_features,total_images = feature_extraction_contexts(new_path)
            ### This does not work if images in total_images only differ in a single
            ### dimension. Broadcasting error occurs.
            #total_images = np.array(total_images)
            total_features = np.array(total_features)
            total_features = total_features.reshape(-1,512)
            #print ('this is object features',total_features.shape)
            #np.savetxt(new_path+'/'+'all_features.csv',total_features,delimiter= ',')
            full_features.extend(total_features)
            full_labels.extend([dictionary[folder] for _ in range(len(total_features))])
            full_images.extend(total_images)
        full_features = np.array(full_features)
        full_labels = np.array(full_labels)
        full_images = np.array(full_images)

    return full_features,full_labels,full_images

# get paths to store data
def get_paths():
    # paths to store data
    base_path = '/home/fetch_user2/zach/projects/fetchGUI_learning_testing/'
    cur_img_dir = base_path + 'cur_img_dir/'
    path_to_object_data = base_path + 'learned_objects/'
    path_to_models = base_path + 'learned_models/'
    path_to_test_data = base_path + 'test_objects/'
    path_dict = dict()
    path_dict['base_path'] = base_path
    path_dict['cur_img_dir'] = cur_img_dir
    path_dict['path_to_object_data'] = path_to_object_data
    path_dict['path_to_models'] = path_to_models
    path_dict['path_to_test_data'] = path_to_test_data

    return path_dict

def setSessionInfo(path_dict,total_models=3,participant_limit=10):
    # participant and session number
    participant_id = input('Participant ID?')
    session_id = input('Session Number?')
    participant_id = int(participant_id)-1
    session_id = int(session_id)-1
    #participant_id = 0
    #session_id = 0

    with open(path_dict['base_path']+'participant_models.json', 'r') as f:
        model_dict = json.load(f)

    if str(participant_id) not in model_dict:
        model_counter = np.zeros(total_models)  # number of times each model has been assigned
        for key in model_dict:
            model_counter[model_dict[key]]+=1

        model_number = random.randrange(total_models)
        while model_counter[model_number]>=participant_limit:   # avoid models that have already reached the participant limit
            model_number = random.randrange(total_models)

        #########CHANGE THIS ######
        #model_number = 0

        model_dict[str(participant_id)] = model_number
        print ('after',model_dict)
        with open(path_dict['base_path']+'participant_models.json','w') as fp:
            json.dump(model_dict,fp,indent=4,sort_keys=True)

    path_to_object_data = path_dict['path_to_object_data']

    # create folders if not already in memory
    participant_folder = path_to_object_data + str(participant_id)+'/'
    if os.path.isdir(participant_folder)==False:
        os.mkdir(participant_folder)
    session_folder = participant_folder + str(session_id) + '/'
    if os.path.isdir(session_folder)==False:
        os.mkdir(session_folder)
    participant_test_folder = path_dict['path_to_test_data'] + str(participant_id)+'/'       # participant folder for test data
    if os.path.isdir(participant_test_folder)==False:
        os.mkdir(participant_test_folder)
    session_test_folder = participant_test_folder + str(session_id) + '/'       # session test folder for participant
    if os.path.isdir(session_test_folder)==False:
        os.mkdir(session_test_folder)
    path_to_participant_model = path_dict['path_to_models'] + str(participant_id)+'/'
    if os.path.isdir(path_to_participant_model)==False:
        os.mkdir(path_to_participant_model)

    # create dictionary files for the participant
    if os.path.exists(participant_folder+'object_label_dictionary.json')==False:
        with open(participant_folder+'object_label_dictionary.json','w') as f:
            json.dump(dict(),f,indent=4,sort_keys=True)
        with open(participant_folder+'object_label_back_dictionary.json','w') as f:
            json.dump(dict(),f,indent=4,sort_keys=True)

    return participant_id,session_id,model_dict[str(participant_id)]


if __name__=='__main__':
    model_dict = {0:'CBCL',1:'FT',2:'iCaRL'}
    dictionary = {'book':0, 'pen':1, 'mouse':2, 'marker':3,
    'stapler':4, 'glue':5, 'milk':6, 'apple':7, 'banana':8, 'cereal':9, 'bowl':10,
    'cup':11, 'plate':12, 'fork':13, 'spoon':14, 'mug':15, 'scotchtape':16,
    'orange':17, 'honey':18}

    path_dict = get_paths()     # directories to store data
    participant_folder, session_folder, model_number = setSessionInfo(path_dict)   # folders for storing images in current session
    model_name = model_dict[model_number]
    full_features,full_labels,_ = process_images(session_folder,dictionary)
    print (np.array(full_features).shape)

import sys
from math import pi

import json
import os
import time
import cv2
from img_to_vec import Img2Vec
from PIL import Image

import pandas as pd
import time
import zmq

import numpy as np
from Functions import get_paths, setSessionInfo
from cl_model import CLModel

############ initialize session parameters #############
model_dict = {0:'CBCL',1:'FT',2:'iCaRL'}

path_dict = get_paths()     # directories to store data
total_models = 3
participant_limit = 21
participant_id, session_id, model_number = setSessionInfo(path_dict,total_models,
                                            participant_limit)   # folders for storing images in current session

print ('participant_id',participant_id)
print ('session_id',session_id)

model_name = model_dict[model_number]
#model_name = 'FT'

participant_folder = path_dict['path_to_object_data'] + str(participant_id)+'/'
session_folder = participant_folder+str(session_id)+'/'
path_to_participant_model = path_dict['path_to_models']+str(participant_id)+'/'

####### CL Model initialization ##########
model = CLModel(model_name=model_name,path=path_to_participant_model)   # CLModel class to handle training, testing
model.load_model()  # load the CL model for the current participant

with open(participant_folder+'object_label_dictionary.json','r') as f:
    object_label_dictionary = json.load(f)   # integer id to label of the object class
with open(participant_folder+'object_label_back_dictionary.json','r') as f:
    object_label_back_dictionary = json.load(f)  # label to integer id of the object class

#model.train_model(session_folder,object_label_back_dictionary)
#model.train_model(participant_folder,object_label_back_dictionary,session_id=session_id)

if __name__ == '__main__':
    # learning request
    #model.train_model(session_folder,dictionary)
    #print (model.predict_labels(path_dict['cur_img_dir']))

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    while True:
        message = socket.recv()
        print ('message: ',message)
        message = message.decode()

        if message == 'learn':
            with open(participant_folder+'object_label_dictionary.json','r') as f:
                object_label_dictionary = json.load(f)   # integer id to label of the object class
            with open(participant_folder+'object_label_back_dictionary.json','r') as f:
                object_label_back_dictionary = json.load(f)       # label to integer id of the object class
            #model.train_model(session_folder,object_label_back_dictionary)
            model.train_model(participant_folder,object_label_back_dictionary,session_id=session_id)
        elif message == 'test':
            labels = model.predict_labels(path_dict['cur_img_dir'])
            labels = np.array(labels)
            np.savetxt(path_dict['cur_img_dir']+'labels.csv',labels,delimiter=',')
        else:
            pass
        socket.send(b"Done")
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")
    while True:
        #  Wait for next request from client

        message = socket.recv()
        print("Received request: %s" % message)

        message = message.decode()

        #################TODO###################
        # is it a learning or a prediction request
        ########################################

        #  Do some 'work'
        process_images(path=message)
        #  Send reply back to client
        socket.send(b"Images are processed")
    """

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
from Functions import feature_extraction_contexts

def process_images():
    #my_creative_breakfast = get_creative_breakfast()
    path_to_CUBS = '/home/fetch_user2/lifelong_context_learning_fetch/grocery_reminder/cur_img_dir'#'/home/fetch/lifelong_context_learning_fetch/grocery_reminder/cur_img_dir'
    total_features,categories,categories_back = feature_extraction_contexts(path_to_CUBS)
    #print ('this is features',np.array(total_features,dtype='object').shape)
    #print ('total_features',total_features)
    for i in range(0,len(total_features)):
        np.savetxt(path_to_CUBS+'/cur_img_features.csv',total_features[i],delimiter=',')

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5551")
    while True:
        #  Wait for next request from client
        message = socket.recv()
        print("Received request: %s" % message)

        #  Do some 'work'
        process_images()
        #  Send reply back to client
        socket.send(b"Images are processed")

import socket
import os
import time
from enum import Enum
from PIL import Image
import threading
import struct
import cv2
from math import pi
from fetchAPI import YoloDetector, RGBCamera, HeadControl, ArmControl
import numpy as np

# get paths to store data
def get_paths():
    # paths to store data
    base_path = '/home/fetch_user2/zach/projects/fetchGUI_learning_testing/'
    cur_img_dir = base_path + 'cur_img_dir/'
    path_to_object_data = base_path + 'learned_objects/'
    path_to_test_data = base_path + 'test_objects/'
    path_to_models = base_path + 'learned_models/'
    path_dict = dict()
    path_dict['base_path'] = base_path
    path_dict['cur_img_dir'] = cur_img_dir
    path_dict['path_to_object_data'] = path_to_object_data
    path_dict['path_to_test_data'] = path_to_test_data
    path_dict['path_to_models'] = path_to_models
    return path_dict

def setSessionInfo(path_dict):
    # participant and session number
    participant_id = raw_input('Participant ID?')
    session_id = raw_input('Session Number?')
    participant_id = int(participant_id)-1
    session_id = int(session_id)-1
    #participant_id = 0
    #session_number = 0

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

    return participant_folder,session_folder, participant_test_folder,session_test_folder

def save_object(current_image_rgb,yd,object_folder):
    """
    current_image_rgb: RGB image
    yd: YoloDetector class instance
    object_folder: path to store the object image data (with object id name added)
    """
    table_offset = 0.8#0.75 #1.02
    current_image_yolo = yd.getDetectionImage()

    cv2.imwrite(object_folder + "_rgb.png", current_image_rgb) #robot.getRGBImage()
    cv2.imwrite(object_folder + "_yolo.png", current_image_yolo)
    f = open(object_folder + "_bounding_boxes.txt", "w")

    all_bboxes = []
    bboxes_len = []
    all_class_names = []
    bbox_range = 20
    bbox_delay = 0.01
    for i in range(bbox_range):
        all_bboxes.append(yd.boxes)
        bboxes_len.append(len(yd.boxes))
        all_class_names.append(yd.class_names)
        time.sleep(bbox_delay)
    temp_ind = np.argmax(bboxes_len)
    all_bboxes = np.array(all_bboxes)
    cur_bbox = all_bboxes[temp_ind]
    cur_class_names = all_class_names[temp_ind]
    #print ('all_class_names',all_class_names)
    #print ('cur_class_names',cur_class_names)
    """
    cur_bbox = yd.boxes
    cur_class_names = yd.class_names
    """
    is_bbox = False
    # filter objects that might be clutter
    for k, box in enumerate(cur_bbox):
        coord_3d, up_3d = yd.get_item_3d_coordinates(None, None,obj_box=box)
        if coord_3d is not None and coord_3d[0]<table_offset and cur_class_names[k]!='person' and int(box[2]-box[0])<(current_image_yolo.shape[1]/2):#1.02:
            f.write(cur_class_names[k] + "," + ",".join([str(b) for b in box]) + "\n")
            is_bbox = True
        elif coord_3d is None:
            if up_3d is not None and up_3d[0]<table_offset:
                f.write(cur_class_names[k] + "," + ",".join([str(b) for b in box]) + "\n")
                is_bbox = True
    f.close()
    return is_bbox

def pointToObject(coord_3d,torso,arm,gripper):
    gripper.close()
    #pick_now = raw_input('should I pick?')
    pick_now = 'y'
    if pick_now == 'y':
        torso.move_to(.4)
        depth_offset = 0.027    # works nicely for straight objects
        up_offset = 0.3         # when using the mid point along height direction

        arm.move_cartesian_position([coord_3d[0]+depth_offset, coord_3d[1], coord_3d[2] + up_offset], [0, pi/2, pi/2])
        time.sleep(1.0)
        arm.stow_planning()
        torso.move_to(0.0)
    else:
        return False
    return True

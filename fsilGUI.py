import socket
import os
import time
from enum import Enum
from PIL import Image
import threading
import struct
import cv2
from fetchAPI import YoloDetector, RGBCamera, HeadControl, ArmControl, SpeechControl
from fetchAPI import TorsoControl, GripperControl
from sound_play.libsoundplay import SoundClient
from utils import get_paths, setSessionInfo, save_object, pointToObject
import zmq
import json
import numpy as np

context = zmq.Context()
socket_cl = context.socket(zmq.REQ)
socket_cl.connect("tcp://localhost:5555")

global ss
global cs
global cs_display

global script_run
global mode
taught_objects=0
global prev_match_value

path_dict = get_paths()     # directories to store data
participant_folder, session_folder, participant_test_folder, session_test_folder = setSessionInfo(path_dict)   # folders for storing images in current session

with open(participant_folder+'object_label_dictionary.json','r') as f:
    object_label_dictionary = json.load(f)
with open(participant_folder+'object_label_back_dictionary.json', 'r') as f:
    object_label_back_dictionary = json.load(f)

import rospy
rospy.init_node('a')

yd = YoloDetector()
rgb = RGBCamera()
speech_module = SpeechControl()
head = HeadControl()
arm = ArmControl()
torso = TorsoControl()
gripper = GripperControl()

head.move_head(-0.0006921291351318359, 0.37409258815460206)

# Enum for each available request by the tablet
##########################
# Changed ButtonRequests to Requests to make it more general
class Requests(Enum):
##########################
    TOGGLE_TEACHING_MODE = 0
    TOGGLE_FINDING_MODE = 1
    RESET_MODE = 2
    SAVE_TEACHING_OBJECT = 3
    FIND_OBJECT = 4
    OK = 5
# Added new Request type`
    RESPONSE = 6

# Enum for each mode the system can operate in
class Mode(Enum):
    TEACHING_MODE = 0
    FINDING_MODE = 1
    DEFAULT_MODE = 2

# Save the passed in object's name
def save_teaching_object(object):
    global taught_objects
    print ('object',object)
    """
    with open(participant_folder+'object_label_dictionary.json','r') as f:
        object_label_dictionary = json.load(f)
    with open(participant_folder+'object_label_back_dictionary.json', 'r') as f:
        object_label_back_dictionary = json.load(f)
    """
    if object != "" and object!=" " and object is not None:
        # folder for saving the object class
        object_folder = session_folder + object + '/'
        if os.path.isdir(object_folder) == False:
            os.mkdir(object_folder)

        current_image_rgb = rgb.getRGBImage()
        #current_image_yolo = yd.getDetectionImage()

        id = len([name for name in os.listdir(object_folder) if
                    os.path.isfile(os.path.join(object_folder, name))])             # find number of files already in the folder
        id = id/3
        id = int(id)

        object_id = object_folder + str(id)
        is_bbox = save_object(current_image_rgb,yd,object_id)     # save object data in the folder

        # update the object label dictionary
        if object not in object_label_back_dictionary and is_bbox:
            object_label_back_dictionary[object] = len(object_label_back_dictionary)
            object_label_dictionary[len(object_label_back_dictionary)-1] = object
        taught_objects +=1
        print("Save object: ", object)

# Find the passed in object
def find_object(object):
    print("Find object: ", object)
    if object != "" and object!=" " and object is not None:
        current_image_rgb = rgb.getRGBImage()
        object_id = path_dict['cur_img_dir'] + str(0)
        save_object(current_image_rgb,yd,object_id)     # save object data in cur_img_dir

        id = len([name for name in os.listdir(session_test_folder) if
                    os.path.isfile(os.path.join(session_test_folder, name))])
        id = id/3
        id = int(id)
        save_object(current_image_rgb,yd,session_test_folder+str(id))       # save test data for later use

        socket_cl.send_string("test")
        message = socket_cl.recv()
        labels = np.genfromtxt(path_dict['cur_img_dir']+'labels.csv',delimiter=',')
        if object not in object_label_back_dictionary: #or len(labels)==0:
            print ('Object not found')
            vocalize_response('I cannot find {}'.format(object), print_to_tablet = True)
            return
        object_id = object_label_back_dictionary[object]
        print ('this is labels',labels)
        if labels.size==1:
            if labels==object_id:
                object_ind = [None]
            else:
                object_ind = []
        else:
            object_ind = np.where(object_id==labels)[0]
        if len(object_ind)==0:
            print ('object is not found')
            vocalize_response('I cannot find {}...'.format(object), print_to_tablet = True)
        else:
            suppressed_boxes = np.genfromtxt(path_dict['cur_img_dir']+'suppressed_boxes.csv',
                                            delimiter=',')
            suppressed_boxes = suppressed_boxes.astype(int)
            #bounding_box = yd.boxes[object_ind[0]]
            if object_ind[0] is None:
                bounding_box = suppressed_boxes
            else:
                bounding_box = suppressed_boxes[object_ind[0]]
            coord_3d,_ = yd.get_item_3d_coordinates(None,None,obj_box=bounding_box)
            if coord_3d is not None:
                # print ('pointing to object at: ',coord_3d)
                vocalize_response('I will point to {} now. Please make sure that you are at a safe distance.'.format(object), print_to_tablet = True)
                pointToObject(coord_3d,torso,arm,gripper)
                vocalize_response('I am done.', print_to_tablet = True)
            else:
                vocalize_response('I have found {}, but I cannot reach it.'.format(object), print_to_tablet = True)
        # move the robot arm to the desired object

def print_message_on_tablet(message):
    # type: (str) -> None
    """
    Sends message to tablet to be printed.
    """
    try:
        cs.sendall(message.encode()+b"\n")
    except Exception as e:
        print(e)
        print("print_message_on_tablet failed")

# Closes the server running on the PC.
def close_PC_server():
    # Close the server
    global ss
    ss.close()
    # global script_run
    # script_run = False
    print("Closed ports and exited script")
    # Exit script
    exit()

# Functionality replaced by prev_mode
def changing_mode_state_reset():
    global taught_objects
    if taught_objects > 0:
        print("Taught " + str(taught_objects) + " objects!")
    taught_objects = 0

##########################
# Takes in a string and sends it to the robot to be vocalized
def vocalize_response(message, print_to_tablet = False):
    # type: (str) -> None
    if print_to_tablet:
        print_message_on_tablet(message)
    message = message + '. . .'
    print (message)
    delay_speech = len(message)/11.9#12.16
    # print ('delay',delay_speech)
    speech_module.say(message)
    time.sleep(delay_speech)
    speech_module.soundhandle.stopAll()
    #speech_module.soundhandle = SoundClient()
##########################

# Takes message data passed from the tablet and calls the proper methods
def decipher_data(data):
    # match_value is the Enum request sent from the tablet
    # message is the string message that follows the request
    match_value, message = data.split("|") #split the data at the delimiter "|"
    match_value = int(match_value)
    global mode
    global prev_match_value
# Whatever request the match_value equates to perform the required method
    if match_value ==  Requests.TOGGLE_TEACHING_MODE.value:
        prev_match_value = match_value
        mode = Mode.TEACHING_MODE
        changing_mode_state_reset()
        print("TEACHING Mode")
    elif match_value ==  Requests.TOGGLE_FINDING_MODE.value:
        prev_match_value = match_value
        mode = Mode.FINDING_MODE
        changing_mode_state_reset()
        print("Finding Mode")
    elif match_value ==  Requests.RESET_MODE.value:
        if prev_match_value == Requests.TOGGLE_TEACHING_MODE.value and taught_objects!=0:
            vocalize_response("I am learning the objects. Please Wait!", print_to_tablet = True)

            # object dictionaries in memory
            with open(participant_folder+'object_label_dictionary.json', 'w') as fp:
                json.dump(object_label_dictionary,fp,indent=4,sort_keys=True)
            with open(participant_folder+'object_label_back_dictionary.json', 'w') as fp:
                json.dump(object_label_back_dictionary,fp,indent=4,sort_keys=True)

            # call a function to learn the saved objects
            print ('learning objects')
            socket_cl.send_string('learn')
            message = socket_cl.recv()

        prev_match_value = match_value
        mode = Mode.DEFAULT_MODE
        changing_mode_state_reset()
        print("Default Mode")
    elif match_value ==  Requests.SAVE_TEACHING_OBJECT.value:
        save_teaching_object(message)
    elif match_value ==  Requests.FIND_OBJECT.value:
        find_object(message)
##########################
    elif match_value == Requests.RESPONSE.value:
        vocalize_response(message)
##########################

# Connect the client to its socket
def connect_client(client_server, client_address):
    # try and connect to the client address
    try:
        client_server.connect(client_address)
    # If it fails ask the user if they would like to retry the connection
    except Exception as e:
        print(e)
        try:
            # Get response to not cause overflow
            response = raw_input(client_address[0]+" failed. Try connecting again? (yes): ")
            if response != 'yes':
                close_PC_server()
            connect_client(client_server, client_address)
        except Exception as e:
            print(e)
            close_PC_server()

# send image thread
def send_image():
    # Defines the min period of each image sending cycle
    functionPeriod = 0.05
    timeSurpassed = 0
    currentTime = 0
    while True:
        # Start timer
        currentTime = time.time()
        # Current code converts numpy array from yolo to jpg and then to bytes
        used_image = yd.detection_image
        is_success, frame_jpg = cv2.imencode(".jpg",used_image)
        if is_success:
            bytes = frame_jpg.tobytes()
            # send the length of bytes to the tablet
            size = len(bytes)
            cs_display.sendall(struct.pack(">L",size))
            # Wait for response about sizing
            buff = cs_display.recv(4)
            # Size response from the tablet
            resp = struct.unpack(">L",buff)
            # If the sizes are equal, i.e. the communication was succesful, send the image data
            if size  == resp[0]:
                cs_display.sendall(bytes)
            # Wait for a final response to know transmission is complete
            buff = cs_display.recv(4)
            resp = struct.unpack(">B",buff)

            # calculate surpased time
            timeSurpassed = time.time() - currentTime
            # Wait as long as necessary to ensure the period is atleast functionPeriod amount of seconds
            try:
                time.sleep(functionPeriod - timeSurpassed)
            except:
                pass

def message_receiving():
    # Do not allow the timeouts while waiting
    ss.settimeout(None)
    ##########################
    # Changed loop to always be true
    while True:
    ##########################
        # Get sent message
        connection, client_address = ss.accept()
        try:
            # Decode and decipher message
            data = connection.recv(150).decode("utf-8")
            decipher_data(data)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    ##########################
    # Possible delete
    # taught_objects = 0
    ##########################
     # set up servers and clients
    server_port = 5000
    client_port = 5001
    client_display_port = 5002
    server_ip_address = '192.168.1.8'#"129.97.71.122" # COMPUTER
    client_ip_address = '192.168.1.2'#"129.97.71.91"  # TABLET
    if server_ip_address is None:
        server_ip_address = input("Input the computers ip address.\nHINT: defining it in the python script will allow you to skip this step ")
    else:
        print("Ensure the correct IP address is being declared in the script.")
    # Enter IP of current computer
    server_address = (server_ip_address, server_port)
    client_address = (client_ip_address, client_port)
    client_display_address = (client_ip_address, client_display_port)
    print("Starting up on", server_ip_address, "server port", server_port)
    # Create a TCP/IP server socket
    ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # receive data from tablet
    cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # sending text data to the tablet
    cs_display = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # sending image data to the tablet
    try:
        # Bind the socket to the server port
        ss.bind(server_address)
        print("Binding Successful.")
    except Exception as e:
        print("\nServer port most likely has a process listening in on it or the IP address being used is incorrect.\nEither check the IP address being used, kill the process on your current server port 'sudo kill -9 $(sudo lsof -t -i:<server port number>)', \nor change to a different server port.")
        print("\nError message: ",e)
        ss.close()
        exit()
    print("Connecting to server...")
    connect_client(cs, client_address)
    connect_client(cs_display, client_display_address)
    print("Connected.")
    print("Listening...")
    ss.listen(5)
    message_thread = threading.Thread(target=message_receiving) # sending and receiving messages
    message_thread.start()
    image_thread = threading.Thread(target = send_image)    # sending images
    image_thread.start()

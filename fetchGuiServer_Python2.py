import socket
import os
import time
from enum import Enum
from PIL import Image
import threading
import struct
import cv2

# import pandas as pd
# import numpy as np

# server socket
global ss
# client socket
global cs
global cs_display

global script_run
global mode
global taught_objects
global prev_match_value

from fetchAPI import YoloDetector, RGBCamera

import rospy
rospy.init_node('a')

yd = YoloDetector()



# Enum for each available request by the tablet
class ButtonRequests(Enum):
    TOGGLE_TEACHING_MODE = 0
    TOGGLE_FINDING_MODE = 1
    RESET_MODE = 2
    SAVE_TEACHING_OBJECT = 3
    FIND_OBJECT = 4
    OK = 5

# Enum for each mode the system can operate in
class Mode(Enum):
    TEACHING_MODE = 0
    FINDING_MODE = 1
    DEFAULT_MODE = 2

# Save the passed in object's name
def save_teaching_object(object):
    # type: (str) -> None
    global taught_objects
    taught_objects = taught_objects + 1
    print("Save object: ", object)

# Find the passed in object
# type: (str) -> None
def find_object(object):
    print("Find object: ", object)
    # move the robot arm to the desired object

# Not used
# Sends a list of items back to the tablet to be displayed
def upload_list():
    try:
        cs.sendall(input("Enter List: ").encode()+b"\n")
    except Exception as e:
        print(e)
        print("upload_list failed")

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


# Takes message data passed from the tablet and calls the proper methods
def decipher_data(data):
    # match_value is the Enum request sent from the tablet
    # message is the string message that follows the request
    match_value, message = data.split("|") #split the data at the delimiter "|"
    match_value = int(match_value)
    global mode
    # Whatever request the match_value equates to perform the required method
    global prev_match_value
    if match_value ==  ButtonRequests.TOGGLE_TEACHING_MODE.value:
        prev_match_value = match_value
        mode = Mode.TEACHING_MODE
        changing_mode_state_reset()
        print("TEACHING Mode")
    elif match_value ==  ButtonRequests.TOGGLE_FINDING_MODE.value:
        prev_match_value = match_value
        mode = Mode.FINDING_MODE
        changing_mode_state_reset()
        print("Finding Mode")
    elif match_value ==  ButtonRequests.RESET_MODE.value:
        if prev_match_value == ButtonRequests.TOGGLE_TEACHING_MODE.value:
            # call a function to learn the saved objects
            pass
        prev_match_value = match_value
        mode = Mode.DEFAULT_MODE
        changing_mode_state_reset()
        print("Default Mode")
    elif match_value ==  ButtonRequests.SAVE_TEACHING_OBJECT.value:
        save_teaching_object(message)
    elif match_value ==  ButtonRequests.FIND_OBJECT.value:
        find_object(message)

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
    # counter = 0
    timeSurpassed = 0
    currentTime = 0
    while True:
        # Start timer
        currentTime = time.time()
        # Current code converts numpy array from yolo to jpg and then to bytes
        image1 = open("res/index.png","rb")
        image2 = open("res/index2.png","rb")
        image = [image1, image2, image1]
        used_image = image[counter]
        bytes = used_image.read()
        # send the length of bytes to the tablet
        size = len(bytes)
        cs_display.sendall(struct.pack(">L",size))
        # cs_display.sendall(size.to_bytes(4, byteorder='big'))
        # Wait for response about sizing
        buff = cs_display.recv(4)
        # resp = int.from_bytes(buff, byteorder ="big")
        # Size response from the tablet
        resp = struct.unpack(">L",buff)
        # If the sizes are equal, i.e. the communication was succesful send the image data
        if size  == resp[0]:
            cs_display.sendall(bytes)
        # No longer necessary
        # counter += 1
        # counter = counter%2

        # Wait for a final response to know transmission is complete
        buff = cs_display.recv(4)
        # resp = int.from_bytes(buff, byteorder ="big")
        resp = struct.unpack(">B",buff)

        # calculate surpased time
        timeSurpassed = time.time() - currentTime
        # Wait as long as necessary to ensure the period is atleast functionPeriod amount of seconds
        try:
            time.sleep(functionPeriod - timeSurpassed)
        except:
            pass



def message_receiving():
    # Switch while loop value to just True
    # Receive message from tablet, decode it and decipher it
    while True:
        used_image = yd.detection_image
        #image1 = open("res/index.png","rb")
        #image2 = open("res/index2.png","rb")
        #image = [image1, image2, image1]
        #used_image = image[counter]
        is_success, frame_jpg = cv2.imencode(".jpg",used_image)
        if is_success:
            bytes = frame_jpg.tobytes()
            size = len(bytes)
            cs_display.sendall(struct.pack(">L",size))
            # cs_display.sendall(size.to_bytes(4, byteorder='big'))
            buff = cs_display.recv(4)
            # resp = int.from_bytes(buff, byteorder ="big")
            resp = struct.unpack(">L",buff)
            if size  == resp[0]:
                cs_display.sendall(bytes)
            counter += 1
            counter = counter%2
            buff = cs_display.recv(4)
            # resp = int.from_bytes(buff, byteorder ="big")
            resp = struct.unpack(">B",buff)
            timeSurpassed = time.time() - currentTime
            try:
                time.sleep(functionPeriod - timeSurpassed)
            except:
                pass

def message_receiving():
    value = "1"
    ss.settimeout(None)
    while value != "N":
        connection, client_address = ss.accept()
        try:
            data = connection.recv(16).decode("utf-8")
            decipher_data(data)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    taught_objects = 0
    # set up servers and clients
    server_port = 5000
    client_port = 5001
    client_display_port = 5002
    server_ip_address = "192.168.1.8" # COMPUTER
    client_ip_address = "192.168.1.4"  # TABLET
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
    # run threads
    message_thread = threading.Thread(target=message_receiving) # sending and receiving messages
    message_thread.start()
    image_thread = threading.Thread(target = send_image)    # sending images
    image_thread.start()

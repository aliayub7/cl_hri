import socket
import os
import cv2
import time
from enum import Enum
from PIL import Image
import threading

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

class ButtonRequests(Enum):
    TOGGLE_TEACHING_MODE = 0
    TOGGLE_FINDING_MODE = 1
    RESET_MODE = 2
    SAVE_TEACHING_OBJECT = 3
    FIND_OBJECT = 4
    OK = 5

class Mode(Enum):
    TEACHING_MODE = 0
    FINDING_MODE = 1
    DEFAULT_MODE = 2

def save_teaching_object(object):
    global taught_objects
    taught_objects = taught_objects + 1
    print("Save object: ", object)

def find_object(object):
    print("Find object: ", object)

def upload_list():
    try:
        cs.sendall(input("Enter List: ").encode()+b"\n")
    except Exception as e:  
        print(e) 
        print("upload_list failed")

def close_PC_server():
    global ss
    ss.close()
    global script_run
    script_run = False
    print("Closed ports and exited script")
    exit()

def changing_mode_state_reset():
    global taught_objects
    if taught_objects > 0:
        print("Taught " + str(taught_objects) + " objects!")
    taught_objects = 0

def decipher_data(data):
    match_value, message = data.split("|")
    match_value = int(match_value)
    global mode
    if match_value ==  ButtonRequests.TOGGLE_TEACHING_MODE.value:
        mode = Mode.TEACHING_MODE
        changing_mode_state_reset()
        print("TEACHING Mode")
    elif match_value ==  ButtonRequests.TOGGLE_FINDING_MODE.value:
        mode = Mode.FINDING_MODE
        changing_mode_state_reset()
        print("Finding Mode")
    elif match_value ==  ButtonRequests.RESET_MODE.value:
        mode = Mode.DEFAULT_MODE
        changing_mode_state_reset()
        print("Default Mode")
    elif match_value ==  ButtonRequests.SAVE_TEACHING_OBJECT.value:
        save_teaching_object(message)
    elif match_value ==  ButtonRequests.FIND_OBJECT.value:
        find_object(message)

def connect_client(client_server, client_address):
    try:
        client_server.connect(client_address)
    except Exception as e:
        print(e)
        try:
            # Get response to not cause overflow
            response = input(client_address[0]+" failed. Try connecting again? (yes): ")
            if response != "yes":
                close_PC_server()
            connect_client(client_server, client_address)
        except Exception as e:
            print(e)
            close_PC_server()

def send_image():
    functionPeriod = 0.05
    counter = 0
    timeSurpassed = 0
    currentTime = 0
    while True:
        currentTime = time.time()
        image1 = open("res/index.png","rb")
        image2 = open("res/index2.png","rb")
        image = [image1, image2, image1]
        used_image = image[counter]
        bytes = used_image.read()
        size = len(bytes)
        cs_display.sendall(size.to_bytes(4, byteorder='big'))
        buff = cs_display.recv(4)
        resp = int.from_bytes(buff, byteorder ="big")
        if size  == resp:
            cs_display.sendall(bytes)
        counter += 1
        counter = counter%2
        buff = cs_display.recv(4)
        resp = int.from_bytes(buff, byteorder ="big")
        timeSurpassed = time.time() - currentTime
        try:
            time.sleep(functionPeriod - timeSurpassed)
        except:
            pass

        

def message_recieving():
    value = "1"
    while value != "N":
        connection, client_address = ss.accept()
        try:
            data = connection.recv(16).decode("utf-8")
            decipher_data(data)
        except Exception as e:
            print(e)
        


if __name__ == "__main__":
    taught_objects = 0
    server_port = 5000
    client_port = 5001
    client_display_port = 5002
    server_ip_address = "192.168.10.101"
    client_ip_address = "192.168.10.102"
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
    ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cs = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    cs_display = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
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
    ss.listen()
    message_thread = threading.Thread(target=message_recieving)
    message_thread.start()
    image_thread = threading.Thread(target = send_image)
    image_thread.start()
        
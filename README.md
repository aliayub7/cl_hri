# Continual Learning through Human-Robot Interaction
Complete code for implementing our socially guided continual learning system (SGCL): Integration of continual learning models with an android tablet and a mobile manipulator robt using ROS. 

### Requirements
* torch (Currently working with 1.3.1)
* Scipy (Currently working with 1.2.1)
* Scikit Learn (Currently working with 0.21.2)
* Use requirements.txt to install all the required libraries
* Download the datasets in */data directory
## Usage
* Create ```checkpoint```, ```data``` and ```previous_classes``` folders.
* Run ```multiple_auto_decay.py``` to run EEC with multiple autoencoders without using pseudorehearsal.
* Run ```multiple_pseudo.py``` to run EEC with multiple autoencoders with pseudorehearsal.
* The code currently has parameters set for ImageNet-50. Just change the appropriate parameters to run it on other datasets.
* Label smoothing was used from this repo: [Link](https://github.com/weiaicunzai/Bag_of_Tricks_for_Image_Classification_with_Convolutional_Neural_Networks)
## If you consider citing us
```
@inproceedings{
ayub2023clhri,
title={Human Perceptions of Task Load and Trust when Interactively Teaching a Continual Learning Robot},
author={Ali Ayub, Zachary De Francesco, Patrick Holthaus, Chrystopher L. Nehaniv, Kerstin Dautenhahn},
booktitle={IEEE/CVF CVPR 2023 (4th Workshop on Continual Learning in Computer Vision)},
year={2023}
}
```

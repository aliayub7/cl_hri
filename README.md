# Continual Learning through Human-Robot Interaction
Complete code for implementing our socially guided continual learning system (SGCL): Integration of continual learning models with an android tablet and a mobile manipulator robt using ROS. 

### Requirements
* ROS with a real Fetch robot or a simulator
* Android tablet
* android studio
* torch 
* Scipy 
* Scikit Learn
## Usage
* Create ```learned_objects```, ```learned_models```, ```test_objects``` and ```cur_img_dir``` folders.
* Run ```lifelong_model_server.py``` from ```lifelong_learning``` folder.
* Connect tablet with the server. Run ```MainActivity.java``` from ```/home/sirrlab1/cl_hri/app/src/main/java/com/example/fetchgui_learning_testing``` in android studio to load the GUI on the tablet.
* Run ```fsilGUI.py``` to connect the GUI with the server and the robot.
* Use the GUI to continually teach and test the robot. 
## If you consider citing us
```
@inproceedings{
ayub2023clhri,
title={Human Perceptions of Task Load and Trust when Interactively Teaching a Continual Learning Robot},
author={Ali Ayub, Zachary De Francesco, Patrick Holthaus, Chrystopher L. Nehaniv, Kerstin Dautenhahn},
booktitle={IEEE/CVF CVPR 2023 (4th Workshop on Continual Learning in Computer Vision)},
year={2023}
}
@inproceedings{
ayub2023roman_continual,
title={How Do Human Users Teach a Continual Learning Robot in Repeated Interactions?},
author={Ali Ayub, Jainish Mehta, Zachary De Francesco, Patrick Holthaus, Kerstin Dautenhahn, Chrystopher L. Nehaniv},
booktitle={IEEE International Conference on Robot and Human Interactive Communication (ROMAN)},
year={2023}
}
@misc{ayub2023continual,
      title={Continual Learning through Human-Robot Interaction -- Human Perceptions of a Continual Learning Robot in Repeated Interactions}, 
      author={Ali Ayub and Zachary De Francesco and Patrick Holthaus and Chrystopher L. Nehaniv and Kerstin Dautenhahn},
      year={2023},
      eprint={2305.16332},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

from gripper_control import GripperControl
from speech_control import SpeechControl
from camera_modules import *
from torso_control import TorsoControl
from base_control import BaseControl
from head_control import HeadControl
from arm_control import ArmControl
from pick_place import PickPlace
from perception import Perception

import moveit_commander
import rospy

class Robot():

    def __init__(self):
        self.robot = moveit_commander.RobotCommander()
        self.gripper = GripperControl()
        self.speaker = SpeechControl()
        self.rgbCamera = RGBCamera()
        self.yoloDetector = YoloDetector()
        self.torso = TorsoControl()
        self.base = BaseControl()
        self.head = HeadControl()
        self.arm = ArmControl()
        #self.pickplace = PickPlace()           # not needed yet
        self.perception = Perception()
        rospy.loginfo("Robot initialised!")

    def getRobotState(self):
        return self.robot.get_current_state().joint_state.name, self.robot.get_current_state().joint_state.position

    def getBasePosition(self):
        return self.base.get_pose()

    def goto(self, x, y, theta ):
        self.base.goto(x, y, theta, "map")

    def getRGBImage(self):
        return self.rgbCamera.curr_image

    def getDetectionImage(self):
        return self.yoloDetector.detection_image

if __name__ == '__main__':
    rospy.init_node("robot")

    robot = Robot()
    print("Base position: ", robot.getBasePosition())
    print("Joint positions: ", robot.getRobotState())

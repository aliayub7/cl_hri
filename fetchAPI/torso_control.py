import rospy
import controller_manager_msgs.srv

import actionlib
import control_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal

ACTION_SERVER = "torso_controller/follow_joint_trajectory"
MIN, MAX = 0, 0.4
from socket import *
IP="kd-pc29.local"
PORT=8080

s = socket(AF_INET, SOCK_STREAM)
try:
    s.connect((IP, PORT))
except error:
    s = None

class TorsoControl(object):

    def __init__(self):

        ## Create publisher
        self.client = actionlib.SimpleActionClient(ACTION_SERVER,
                                                   FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for torso controller....")
        if self.client.wait_for_server():
            rospy.loginfo("Got torso controller")
        else:
            rospy.logerr("Couldn't connect to torso controller....")

        self.joint_names = ["torso_lift_joint"]
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_to(self, position, duration=5.0):
        if len(self.joint_names) != 1:
            print("Invalid trajectory position")
            return False
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names
        trajectory.points.append(JointTrajectoryPoint())
        trajectory.points[0].positions = [position]
        trajectory.points[0].velocities = [0.0]
        trajectory.points[0].accelerations = [0.0]
        trajectory.points[0].time_from_start = rospy.Duration(duration)
        follow_goal = FollowJointTrajectoryGoal()
        follow_goal.trajectory = trajectory

        self.client.send_goal(follow_goal)
        self.client.wait_for_result()
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def lower(self):
        self.move_to(0.0)
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

if __name__ == '__main__':
    rospy.init_node("test_torso")
    from torso_control import TorsoControl
    torso = TorsoControl()
    torso.move_to(.1)
    # time.sleep(5)
    # torso.lower()
    # time.sleep(5)

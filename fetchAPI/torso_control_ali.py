import rospy
import controller_manager_msgs.srv

import actionlib
import control_msgs.msg
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
import moveit_commander

ACTION_SERVER = "torso_controller/follow_joint_trajectory"
MIN, MAX = 0, 0.4


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
        self.robot = moveit_commander.RobotCommander()

    def move_to(self, position, duration=5.0):  # position between 0 and 0.4
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

    def lower(self):
        self.move_to(MIN)

    def get_pose(self):
        self.actual_positions = self.robot.get_current_state().joint_state.position[2]
        # print(self.actual_positions)
        return self.actual_positions

    def is_there(self, h_torso, tolerance=0.02):
        cur_h = self.get_pose()
        if abs(cur_h - h_torso) < tolerance:
            return True
        else:
            return False


if __name__ == '__main__':
    rospy.init_node("test_torso")
    from torso_control import TorsoControl
    import time

    torso = TorsoControl()
    if not torso.is_there(0.0):
        torso.move_to(.0)
        time.sleep(5)
    else:
        print('already there')

    # torso.lower()
    # time.sleep(5)

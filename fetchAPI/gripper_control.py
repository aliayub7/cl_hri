import rospy
import controller_manager_msgs.srv
import trajectory_msgs.msg
import time
import actionlib
import control_msgs.msg

CLOSED_POS = 0.0
OPENED_POS = 0.10
ACTION_SERVER = 'gripper_controller/gripper_action'

class GripperControl(object):
    """ Gripper interface """
    MIN_EFFORT, MAX_EFFORT = 35, 100
    def __init__(self):
        ## Connect to action server
        self._client = actionlib.SimpleActionClient(ACTION_SERVER, control_msgs.msg.GripperCommandAction)
        print("waiting for gripper server...")
        res = self._client.wait_for_server()
        if res:
            rospy.loginfo("Got gripper controller....")
        else:
            rospy.loginfo("Couldn't connect to gripper controller....")

    def open(self, pos=OPENED_POS):
        # fill ROS message
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = pos
        self._client.send_goal_and_wait(goal, rospy.Duration(10))

    def close(self, pos=CLOSED_POS, max_effort = MAX_EFFORT):
        # fill ROS message
        goal = control_msgs.msg.GripperCommandGoal()
        goal.command.position = pos
        goal.command.max_effort = max_effort
        self._client.send_goal_and_wait(goal, rospy.Duration(10))

if __name__ == '__main__':
    rospy.init_node("test_gripper")
    gripper_module = GripperControl()
    print("Testing gripper")
    gripper_module.close()
    time.sleep(5)
    gripper_module.open()
    time.sleep(5)

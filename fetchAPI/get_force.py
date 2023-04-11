
import rospy
import moveit_commander
from fetch_driver_msgs.msg import RobotState

def cb(data):
    print(data.joints)
    print(data.motors)
    print(data.boards)`         
    `
rospy.init_node("force_test")
_sub = rospy.Subscriber("/robot_state", RobotState, cb)
# robot = moveit_commander.RobotCommander()
# print(robot.get_current_state())

rospy.spin()

# Author: H J Kashyap, T Hwu
import rospy
import trajectory_msgs.msg
import control_msgs.msg
import time
import actionlib
import moveit_commander
from socket import *
IP="kd-pc29.local"
PORT=8080

s = socket(AF_INET, SOCK_STREAM)
try:
    s.connect((IP, PORT))
except error:
    s = None

class HeadControl(object):
    """ Head control interface """

    def __init__(self):
        self.actual_positions = (0, 0)

        self.robot = moveit_commander.RobotCommander()

        ## initialize head action client
        self.cli_head = actionlib.SimpleActionClient(
            '/head_controller/follow_joint_trajectory', control_msgs.msg.FollowJointTrajectoryAction)
        ## wait for the action server to establish connection
        rospy.loginfo("Waiting for head controller....")
        if not self.cli_head.wait_for_server():
            rospy.logerr("Couldn't connect to head controller... ")
        else:
            rospy.loginfo("Got head controller ")
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_head(self, pan_pos, tilt_pos, time=rospy.Time(1.0)):
        # fill ROS message
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        traj = trajectory_msgs.msg.JointTrajectory()
        traj.joint_names = ["head_pan_joint", "head_tilt_joint"]
        p = trajectory_msgs.msg.JointTrajectoryPoint()
        p.positions = [pan_pos, tilt_pos]
        p.velocities = [0, 0]
        p.time_from_start = time
        traj.points = [p]
        goal.trajectory = traj
        # send message to the action server
        self.cli_head.send_goal(goal)
        # wait for the action server to complete the order
        self.cli_head.wait_for_result()
        rospy.sleep(1)
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def head_sweep(self,pan_pos):
        self.move_head(pan_pos, 0.0, rospy.Time(3))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_up(self, offset=0.5):
        self.get_pose()
        self.move_head(self.actual_positions[0], self.actual_positions[1]-abs(offset), rospy.Time(.4))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_down(self, offset=0.5):
        self.get_pose()
        self.move_head(self.actual_positions[0], self.actual_positions[1]+abs(offset), rospy.Time(.4))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_left(self, offset=0.5):
        self.get_pose()
        self.move_head(self.actual_positions[0]+abs(offset), self.actual_positions[1], rospy.Time(.4))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_right(self, offset=0.5):
        self.get_pose()
        self.move_head(self.actual_positions[0]-abs(offset), self.actual_positions[1], rospy.Time(.4))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_home(self):
        self.move_head(0.0, 0.0, rospy.Time(1.5))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_left_45(self):
        self.get_pose()
        self.move_head(self.actual_positions[0]+0.775, self.actual_positions[1], rospy.Time(2.5))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_right_45(self):
        self.get_pose()
        self.move_head(self.actual_positions[0]-0.775, self.actual_positions[1], rospy.Time(2.5))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def get_pose(self):
        # print(self.robot.get_current_state())
        self.actual_positions = self.robot.get_current_state().joint_state.position[4:6]
        # print(self.actual_positions)
        return self.actual_positions

    # def __del__(self):
        # traj = trajectory_msgs.msg.JointTrajectory()
        # goal.trajectory = traj
        # self.cli_head.send_goal(goal)
        # self.cli_head.wait_for_result()
        # rospy.sleep(1)

if __name__ == '__main__':
    import json

    rospy.init_node("test_head")
    head_control = HeadControl()
    print (head_control.get_pose())

    locs = dict()
    locs['kitchen'] = dict()
    locs['kitchen'][0] = dict()
    locs['kitchen'][0]['short'] = [0.8485648666963335, 3.762766431642447, 1.6084295952939303]
    locs['kitchen'][0]['long'] = [1.0919262102812701, 2.8111348947068384, 1.7806022475707628]
    locs['kitchen'][1] = dict()
    locs['kitchen'][1]['short'] = [1.545694222170105, 3.760435892828146, 1.5721870366409374]
    locs['kitchen'][1]['long'] = [1.540568738675812, 3.172157269251085, 1.5771886462558253]
    locs['kitchen'][2] = dict()
    locs['kitchen'][2]['short'] = [2.2372463803757294, 3.76104417125752, 1.4509776190025]
    locs['kitchen'][2]['long'] = [2.163141398695406, 3.25236043706042, 1.4932653595642869]
    locs['kitchen'][3] = dict()
    locs['kitchen'][3]['short'] = [2.797692478054923, 3.743869940244645, 1.5493808040578205]
    locs['kitchen'][3]['long'] = [2.7751962192020545, 3.241791558730422, 1.5544040558288263]
    locs['office'] = dict()
    locs['office'][0] = dict()
    locs['office'][0]['short'] = [2.0658155195587664, -0.1611813998500562, -3.138425178625453]
    locs['office'][0]['long'] = [2.866313594831272, -0.19238173347203147, 3.09875621194747]
    locs['dining'] = dict()
    locs['dining'][0] = dict()
    locs['dining'][0]['short'] = [3.7676210689418697, 2.2480682047432365, -0.061708119125181655]
    locs['dining'][0]['long'] = [2.96394772359932, 2.2366879175411483, 0.04813706520035312]
    locs['dining'][1] = dict()
    locs['dining'][1]['short'] = [3.7335562501844857, 2.864651713033702, 0.0527104350369489]
    locs['dining'][1]['long'] = [2.931276235615739, 2.7911127228087533, 0.06019387024331343]
    locs['trash'] = dict()
    locs['trash'][0] = dict()
    locs['trash'][0]['short'] = [3.9323742665364554, -2.085278954845096, -1.569207655388622]
    locs['trash'][0]['long'] = [3.9133990315098153, -1.285090554631736, -1.5928405231375944]
    locs['start'] = [3.6051376130464234, -0.6889106314112116, 2.3810142327988593]
    with open('./locations.json', 'w') as fp:
        json.dump(locs,fp,indent=4,sort_keys=True)

    """
    -0.06238687038421631, 0.3576021287124634
    """
    # for kitchen tables: -0.0024718046188354492, 0.2394857499282837
    #head_control.move_head(0,0.239485)
    #head_control.move_head(-0.03529655933380127,0.48990798923187256)


    """
    head_control.move_down()
    time.sleep(3)
    head_control.move_left()
    time.sleep(2)
    head_control.move_home()
    time.sleep(2)
    head_control.move_right()
    time.sleep(3)
    head_control.move_home()
    """

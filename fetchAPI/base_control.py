import rospy
import geometry_msgs.msg
import trajectory_msgs.msg
import control_msgs.msg
import yaml
import time
import actionlib
import moveit_commander
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from math import sin, cos
import tf

import time

from head_control import HeadControl

from socket import *
IP="kd-pc29.local"
PORT=8080

s = socket(AF_INET, SOCK_STREAM)
try:
    s.connect((IP, PORT))
except error:
    s = None

class BaseControl(object):
    """ Move base and navigation """

    def __init__(self):
        ## Create publisher to move the base
        self._pub = rospy.Publisher('/cmd_vel', geometry_msgs.msg.Twist, queue_size=10)
        self.tf_listener = tf.TransformListener()

        ##action client for navigation
        self.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        rospy.loginfo("Waiting for move_base...")
        if not self.client.wait_for_server():
            rospy.logerr("Could not connect to move_base... Did you roslaunch fetch_navigation fetch_nav.launch map_file:=/home/fetch_admin/map_directory/5427_updated.yaml?")
        else:
            rospy.loginfo("Got move_base")

        self.actual_positions = (0, 0, 0)
        self.actual_vel = (0, 0, 0)

        self.pose = None
        #self.orientation = (0,0,0,0)

        ## subscribe to odom to get the robot current position
        def callback(msg):
            p = msg.pose.pose.position
            #print ('this is orientation',msg.pose.orientation)
            # just checking
            self.pose = msg.pose.pose
            #self.pose = msg.pose.pose

            self.actual_positions = (p.x, p.y, p.z)
        self._sub = rospy.Subscriber("/odom", Odometry, callback)
        self.actual_positions = (0, 0, 0)

        while self._pub.get_num_connections() == 0:
            rospy.sleep(0.1)

        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_forward(self, speed=0.2):
        tw = geometry_msgs.msg.Twist()
        tw.linear.x =abs(speed)
        self._pub.publish(tw)
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_backward_new(self, second=None, distance=None, speed=-0.2):
        tw = geometry_msgs.msg.Twist()
        if (second is None) and (distance is None):
            tw.linear.x = -abs(speed)
            self._pub.publish(tw)
        elif (second is not None) and (distance is None):
            cur_time = time.time()
            while time.time() - cur_time < second:
                tw.linear.x = -abs(speed)
                self._pub.publish(tw)
            # self.get_pose()

        elif (second is None) and (distance is not None):
            cur_time = time.time()
            while time.time() - cur_time < 1:
                tw.linear.x = -abs(0)
                self._pub.publish(tw)
            init_pos = self.get_pose()
            print(init_pos)
            dx = 0
            dy = 0
            while max(dx, dy) < distance:
                tw.linear.x = -abs(speed)
                self._pub.publish(tw)
                cur_pos = self.get_pose()
                dx = abs(cur_pos[0] - init_pos[0])
                dy = abs(cur_pos[1] - init_pos[1])
                # old_pos = copy.copy(cur_pos)
            print(cur_pos)
        elif (second is not None) and (distance is not None):
            print("only one of second or distance!")

    def move_backward(self, speed=-0.2):
         tw = geometry_msgs.msg.Twist()
         tw.linear.x = -abs(speed)
         self._pub.publish(tw)
         if s:
             s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_left(self, angle=0.2):
        tw = geometry_msgs.msg.Twist()
        tw.angular.z = abs(angle)
        self._pub.publish(tw)
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_right(self, angle=-0.2):
        tw = geometry_msgs.msg.Twist()
        tw.angular.z = -abs(angle)
        self._pub.publish(tw)
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    ## get 3D position of the robot
    def get_pose(self):
        (trans, rot) = self.tf_listener.lookupTransform('/map', '/base_link', rospy.Time(0))
        new_rot = tf.transformations.euler_from_quaternion(rot)
        #print(trans, new_rot)
        self.actual_positions = [trans[0], trans[1], new_rot[2]]
        return self.actual_positions#,self.pose

    def __del__(self):
        self._pub.publish(geometry_msgs.msg.Twist())
        move_goal = MoveBaseGoal()
        self.client.send_goal(move_goal)
        self.client.wait_for_result()

    ##### Navigation
    def goto(self, x, y, theta=None,orientation=None, frame="map"):
        move_goal = MoveBaseGoal()
        move_goal.target_pose.pose.position.x = x
        move_goal.target_pose.pose.position.y = y
        if theta is not None:
            move_goal.target_pose.pose.orientation.z = sin(theta/2.0)
            move_goal.target_pose.pose.orientation.w = cos(theta/2.0)
        else:
            move_goal.target_pose.pose.orientation.z = orientation[0]
            move_goal.target_pose.pose.orientation.w = orientation[1]
        move_goal.target_pose.header.frame_id = frame
        move_goal.target_pose.header.stamp = rospy.Time.now()
        self.client.send_goal(move_goal)
        self.client.wait_for_result()
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def goto_relative(self, dx, dy, dtheta, frame="map"):
        move_goal = MoveBaseGoal()
        move_goal.target_pose.pose.position.x = self.actual_positions[0] + dx
        move_goal.target_pose.pose.position.y = self.actual_positions[1] + dy
        move_goal.target_pose.pose.orientation.z = self.actual_positions[2] + sin(dtheta/2.0)
        move_goal.target_pose.pose.orientation.w = self.actual_positions[2] + cos(dtheta/2.0)
        move_goal.target_pose.header.frame_id = frame
        move_goal.target_pose.header.stamp = rospy.Time.now()
        self.client.send_goal(move_goal)
        self.client.wait_for_result()
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

if __name__ == '__main__':
    rospy.init_node("test_base")
    base_control = BaseControl()
    head_control = HeadControl()


    # start location
    x = -2.579#-5.23203065379
    y = -2.528#-0.009
    orientation = [0.901, 0.433] #[0.726619513435,0.687040088129]
    angle = 2.245#1.586

    # approach kitchen
    x = -4.521#-5.23203065379
    y = -.778#-0.009
    orientation = [0.716, 0.698] #[0.726619513435,0.687040088129]
    angle = 1.597#1.586

    #base_control.goto(x,y,theta=angle,orientation=orientation)

    #loc 1 kitchen
    x = -5.420#-5.23203065379
    y = 0.049#-0.009
    orientation = [0.716, 0.698] #[0.726619513435,0.687040088129]
    angle = 1.58#1.586

    # loc 2 kitchen
    x = -4.966#-5.23203065379
    y = 0.084#-0.009
    orientation = [0.718, 0.696] #[0.726619513435,0.687040088129]
    angle = 1.602#1.586

    # loc 3 kitchen
    x = -4.460#-5.23203065379
    y = 0.158#-0.009
    orientation = [0.716, 0.698] #[0.726619513435,0.687040088129]
    angle = 1.597#1.586

    # loc 4 kitchen
    x = -4.0#-5.23203065379
    y = 0.18#0.084
    orientation = [0.724, 0.690] #[0.726619513435,0.687040088129]
    angle = 1.7#1.625

    # loc 5 kitchen
    x = -3.324#-5.23203065379
    y = 1.223#-0.009
    orientation = [1.000, 0.006] #[0.726619513435,0.687040088129]
    angle = 3.131#1.586

    """
    # loc 6 approach1 kitchen
    x = -6.596#-5.23203065379
    y = -1.352#-0.009
    orientation = [0.724, 0.690] #[0.726619513435,0.687040088129]
    angle = 1.619#1.586
    base_control.goto(x,y,theta=angle,orientation=orientation)

    # loc 6 approach2 kitchen
    x = -6.976#-5.23203065379
    y = 0.984#-0.009
    orientation = [-0.008, 1.000] #[0.726619513435,0.687040088129]
    angle = -0.016#1.586
    base_control.goto(x,y,theta=angle,orientation=orientation)

    # loc 6 kitchen
    x = -6.285#-5.23203065379
    y = 1.021#-0.009
    orientation = [-0.007, 1.000] #[0.726619513435,0.687040088129]
    angle = -0.013#1.586
    """

    # office approach
    x = -5.693#-5.23203065379
    y = -1.594#-0.009
    orientation = [-0.635, 0.773] #[0.726619513435,0.687040088129]
    angle = -1.375#1.586

    #base_control.goto(x,y,theta=angle,orientation=orientation)

    # loc 1 office
    x = -5.249#-5.349#-5.23203065379
    y = -2.379#-2.331
    orientation = [-0.667, 0.745] #[0.726619513435,0.687040088129]
    angle = -1.561#1.586

    #base_control.goto(x,y,theta=angle,orientation=orientation)

    # loc 2 office
    x = -5.8#-5.23203065379
    y = -2.331#-0.009
    orientation = [-0.675, 0.738] #[0.726619513435,0.687040088129]
    angle = -1.461#1.586


    # loc 3 office
    x = -6.2#-6.051
    y = -2.4#-2.447
    orientation = [-0.675, 0.738] #[-.691,.723]
    angle = -1.461#-1.525

    # loc 4 office
    x = -6.6#-6.376#-5.23203065379
    y = -2.379#-2.379
    orientation = [-0.675, 0.738] #[-.689,.725]
    angle = -1.461#-1.521

    # dining approach
    x = -3.355
    y = -.589
    orientation = [.005, 1.000]
    angle = .010

    #base_control.goto(x,y,theta=angle,orientation=orientation)

    # dining location
    x = -2.249
    y = -.619
    orientation = [.005, 1.000]
    angle = .010

    #base_control.goto(x,y,theta=angle,orientation=orientation)

    time.sleep(1.0)
    #head_control.move_head(0,0.239485)

    #base_control.goto(3.2245382138551877,1.0301291047770063,theta=1.2176060776996844,orientation=None)
    head_pose = head_control.get_pose()
    print ('head_pose',head_pose)
    pose = base_control.get_pose()
    print ('first pose',pose)
    #base_control.move_backward_new(distance=0.8)
    #print (base_control.get_pose())
    #print (base_control.pose)
    #a,b = base_control.get_pose()
    #print (a)
    #print (b)
    """NOTE!!!"""
    """rostopic echo /odom will give the current pose and orientation"""

    """
    base_control = BaseControl()
    table_1 = [[-3.896,-.156,1.578],[-4.239,.792,1.566]]
    table_1_mf = 18
    table_2 = [[-5.468,.540,1.547],[-5.338,.929,1.611]]
    table_2_mf = 14
    dining_table = [[-2.185,1.020,-.004]]
    dining_table_mf = 15
    initial_location = [-3.686,-2.244,1.566]
    base_control.goto(table_1[0][0],table_1[0][1],table_1[0][2])
    time.sleep(2)
    base_control.goto(table_1[1][0],table_1[1][1],table_1[1][2])
    #base_control.goto(x,y,theta)
    for i in range(0,table_1_mf):
        base_control.move_forward()
        time.sleep(2)
    #for i in range(0,table_1_mf):
    #    base_control.move_backward()
    #    time.sleep(2)
    base_control.goto(initial_location[0],initial_location[1],initial_location[2])
    """
    """
    print (base_control.get_pose())
    base_control.goto(table_1[0][0],table_1[0][1],table_1[0][2])
    time.sleep(2)
    base_control.goto(table_1[1][0],table_1[1][1],table_1[1][2])
    #base_control.goto(x,y,theta)
    for i in range(0,table_1_mf):
        base_control.move_forward()
        time.sleep(2)
    for i in range(0,table_1_mf):
        base_control.move_backward()
        time.sleep(2)
    base_control.goto(initial_location[0],initial_location[1],initial_location[2])
    """
    # base_control.move_right(45)
    #base_control.move_forward(1.0)
    #time.sleep(2)
    #base_control.move_forward()
    # base_control.move_right()

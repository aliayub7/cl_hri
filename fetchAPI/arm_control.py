import rospy
from control_msgs.msg import JointTrajectoryControllerState, FollowJointTrajectoryAction
from moveit_python import MoveGroupInterface
import actionlib
from geometry_msgs.msg import Pose
import moveit_commander
from moveit_msgs.msg import MoveItErrorCodes, OrientationConstraint, Constraints
from scene import *
from tf.transformations import quaternion_from_euler
from math import pi
import thread
import time
from socket import *
IP="kd-pc29.local"
PORT=8080

s = socket(AF_INET, SOCK_STREAM)
try:
    s.connect((IP, PORT))
except error:
    s = None

class ArmControl(object):
    """ Move arm """

    def __init__(self):
        self.client = actionlib.SimpleActionClient("arm_controller/follow_joint_trajectory",
                                                   FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for arm controller....")
        self.client.wait_for_server()
        if self.client.wait_for_server():
            rospy.loginfo("Got arm controller")
        else:
            rospy.logerr("Couldn't connect to arm controller.... Did you roslaunch fetch_moveit_config move_group.launch?")

        self.actual_positions = [0, 0, 0, 0, 0, 0, 0]
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint",
                            "upperarm_roll_joint", "elbow_flex_joint",
                            "forearm_roll_joint", "wrist_flex_joint",
                            "wrist_roll_joint"]

        self.tuck_joint_value = {}
        self.stow_joint_value = {}
        self.inter_stow_joint_value = {}
        self.zero_joint_value = {}
        self.tuck_values = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        self.stow_values = [1.6, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        self.inter_stow_values = [0.7, -0.3, 0.0, -0.3, 0.0, -0.57, 0.0]
        self.zero_values = [0, 0, 0, 0, 0, 0, 0]
        for j, joint in enumerate(self.joint_names):
            self.tuck_joint_value[joint] = self.tuck_values[j]
            self.stow_joint_value[joint] = self.stow_values[j]
            self.inter_stow_joint_value[joint] = self.inter_stow_values[j]
            self.zero_joint_value[joint] = self.zero_values[j]

        self.robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("arm")
        self.move_group = MoveGroupInterface("arm", "base_link")

        print(self.arm_group.has_end_effector_link(), self.arm_group.get_end_effector_link(), self.arm_group.get_joints())
        self.arm_group.set_planning_time(20)
        self.arm_group.set_planner_id("RRTConnectkConfigDefault")
        self.arm_group.allow_replanning(True)
        self.arm_group.clear_path_constraints()

        self.pause = [True]
        """
        NOT NEEDED RIGHT NOW
        self.scene = Scene(PERCEPTION_MODE)
        self.scene.clear()
        self.thread = thread.start_new_thread(self.scene.update, (self.pause, ))
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))
        """

    def move_joint_position_commander(self,position):
        self.pause[0] = False
        self.arm_group.clear_pose_targets()
        self.arm_group.set_start_state_to_current_state()
        self.arm_group.set_joint_value_target(position)
        plan1 = self.arm_group.plan()
        #print ('this is the plan',plan1)
        self.arm_group.execute(plan1,wait=True)
        self.arm_group.stop()
        self.pause[0] = True
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def move_joint_positions(self, positions):
        self.pause[0] = False
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, positions, 0.02)
            if s:
                s.send(",".join([str(d) for d in list(self.get_pose())]))
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.pause = [True]
                return

    def move_joint_position(self, joint_name, position):
        self.pause[0] = False
        positions = list(self.get_pose())
        positions[self.joint_names.index(joint_name)] = position
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, positions, 0.02)
            if s:
                s.send(",".join([str(d) for d in list(self.get_pose())]))
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.pause[0] = True
                return

    def move_cartesian_position(self, position, orientation=[0.0, 0.0, 0.0]):
        ''' position: xyz
            orientation: rpy'''
        self.pause[0] = False
        #self.arm_group.set_start_state(self.robot.get_current_state())
        self.arm_group.set_start_state_to_current_state()
        quaternion = quaternion_from_euler(orientation[0], orientation[1], orientation[2])

        pose_target = Pose()
        pose_target.position.x = position[0]
        pose_target.position.y = position[1]
        pose_target.position.z = position[2]
        pose_target.orientation.w = quaternion[3]
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]

        # constraints = Constraints()
        # constraints.name = "Keep gripper horizontal"
        #
        # orientation_constraint = OrientationConstraint()
        # orientation_constraint.header = self.arm_group.get_current_pose(self.arm_group.get_end_effector_link()).header
        # orientation_constraint.link_name = self.arm_group.get_end_effector_link()
        #

        # orientation_constraint.orientation.w = quaternion[3]
        # orientation_constraint.orientation.x = quaternion[0]
        # orientation_constraint.orientation.y = quaternion[1]
        # orientation_constraint.orientation.z = quaternion[2]
        #
        # orientation_constraint.absolute_x_axis_tolerance = 0.2
        # orientation_constraint.absolute_y_axis_tolerance = 0.2
        # orientation_constraint.absolute_z_axis_tolerance = 0.2
        # orientation_constraint.weight = 1.0
        # constraints.orientation_constraints.append(orientation_constraint)
        # self.arm_group.set_path_constraints(constraints)

        result = self.arm_group.set_pose_target(pose_target)
        """
        now = time.time()
        plan1 = self.arm_group.plan()
        print ('this is the plan',plan1)
        print ('timing for planning',time.time()-now)
        now = time.time()
        self.arm_group.execute(plan1,wait=True)
        print ('timing for movement',time.time()-now)
        """
        now = time.time()
        plan = self.arm_group.go(wait=True)


        #print ('timing for movement',time.time()-now)
        # result = self.arm_group.set_rpy_target([.0,.0,.0])
        # plan = self.arm_group.go(wait=True)

        self.arm_group.stop()
        # self.arm_group.clear_path_constraints()
        self.pause[0] = True
        if s:
            s.send(",".join([str(d) for d in list(self.get_pose())]))

    def tuck_planning(self):
        my_dict = {}
        joing_values = [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0]
        for i in range(len(self.joint_names)):
            my_dict[self.joint_names[i]] = joing_values[i]
        self.move_joint_position_commander(my_dict)

    def stow_planning(self):
        my_dict = {}
        joing_values = [1.55, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        #joing_values = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        for i in range(len(self.joint_names)):
            my_dict[self.joint_names[i]] = joing_values[i]
        self.move_joint_position_commander(my_dict)

    def stow_safe(self, planning=True, **kwargs):
        p1 = self.add_box()
        p2 = self.attach_box()
        if planning:
            result = self.plan_joint_position(self.stow_joint_value)
        else:
            if kwargs is not None:
                if "max_velocity_scaling_factor" not in kwargs:
                    kwargs["max_velocity_scaling_factor"] = 0.05
            else:
                kwargs = {"max_velocity_scaling_factor": 0.05}
            # while not rospy.is_shutdown():
            plan = self.move_group.moveToJointPosition(self.joint_names, self.stow_values,
                                                       0.02, **kwargs)
            result = plan.error_code
        p3 = self.detach_box()
        p4 = self.remove_box()

    def tuck(self):
        self.pause[0] = False
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, [1.32, 1.40, -0.2, 1.72, 0.0, 1.66, 0.0], 0.02)
            if s:
                s.send(",".join([str(d) for d in list(self.get_pose())]))
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.pause[0] = True
                return

    def zero(self):
        self.pause[0] = False
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.02)
            if s:
                s.send(",".join([str(d) for d in list(self.get_pose())]))
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.pause[0] = True
                return

    def stow(self):
        self.pause[0] = False
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0], 0.02)
            if s:
                s.send(",".join([str(d) for d in list(self.get_pose())]))
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.pause[0] = True
                return

    def intermediate_stow(self):
        self.pause[0] = False
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, [0.7, -0.3, 0.0, -0.3, 0.0, -0.57, 0.0], 0.02)
            if s:
                s.send(",".join([str(d) for d in list(self.get_pose())]))
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                self.pause[0] = True
                return

    def get_pose(self):
        self.actual_positions = self.robot.get_current_state().joint_state.position[6:13]
        return self.actual_positions

    def add_box(self, timeout=4):
        box_pose = geometry_msgs.msg.PoseStamped()
        box_pose.header.frame_id = "wrist_roll_link"
        box_pose.pose.orientation.w = 1.0
        box_pose.pose.position.x = 1.5
        self.box_name = 'box'
        self.scene.add_box(self.box_name, box_pose, size=(0.22, 0.19, 0.15))  # (0.22, 0.19, 0.15)
        return self.wait_for_state_update(box_is_known=True, timeout=timeout)

    def attach_box(self, timeout=4):
        grasping_group = 'gripper'
        touch_links = self.robot.get_link_names(group=grasping_group)
        self.scene.attach_box(self.eef_link, self.box_name, touch_links=touch_links)
        return self.wait_for_state_update(box_is_attached=True, box_is_known=False, timeout=timeout)

    def _del_(self):
        self.tuck()

if __name__ == '__main__':
    import time
    rospy.init_node("test_arm")
    from torso_control import TorsoControl

    torso = TorsoControl()
    arm_module = ArmControl()
    #
    # pose = list(arm_module.get_pose())
    # pose[-2] += 0.5
    # s.send(",".join([str(d) for d in list(pose)]))
    # arm_module.stow()
    # torso.move_to(0.4)

    should_tuck = raw_input('Should I tuck?')
    torso.move_to(0.4)
    if should_tuck=='y':
        arm_module.tuck_planning()
    else:
        arm_module.stow_planning()
    torso.move_to(0.0)

    # arm_module.tuck()
    # time.sleep(5)
    # arm_module.move_joint_positions( [0,0,0,0,0,0,0])
    # time.sleep(5)
    # arm_module.move_joint_position("shoulder_pan_joint", .5)
    # time.sleep(5)
    # coord_3d =  [0.6225116569524847, -0.06502431886462116, 0.8501489263014902] #[0.6478926483367018, -0.16955347545496427, 0.8426119154023802]
    # arm_module.move_cartesian_position([coord_3d[0]-0.015, coord_3d[1]+0.01, coord_3d[2]+.3], [0.0, pi/2, 0.0])
    # print("somehow reached goal even though it makes no sense")
    # time.sleep(15)
    # arm_module.tuck()
    # time.sleep(5)

    #torso.move_to(0.0)

    # arm_module.scene.clear()

    print (arm_module.get_pose())

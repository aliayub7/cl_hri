import rospy
from control_msgs.msg import JointTrajectoryControllerState, FollowJointTrajectoryAction
from moveit_python import MoveGroupInterface
import actionlib
import geometry_msgs.msg
import sys
# from motion_planning_msg.msg import SimplePoseConstraint
import moveit_commander
from moveit_commander.conversions import pose_to_list
import moveit_msgs.msg
from scene import *
from tf.transformations import quaternion_from_euler
from math import pi


class ArmControl(object):
    # tutorial location: https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html
    # reference to moveitcommander: http://docs.ros.org/en/jade/api/moveit_commander/html/classmoveit__commander_1_1robot_1_1RobotCommander.html
    # reference to movegropupinterface: http://docs.ros.org/en/melodic/api/moveit_ros_planning_interface/html/classmoveit_1_1planning__interface_1_1MoveGroupInterface.html

    """ Move arm """

    def __init__(self):
        self.box_name = None
        self.client = actionlib.SimpleActionClient("arm_controller/follow_joint_trajectory",
                                                   FollowJointTrajectoryAction)
        rospy.loginfo("Waiting for arm controller....")
        self.client.wait_for_server()
        if self.client.wait_for_server():
            rospy.loginfo("Got arm controller")
        else:
            rospy.logerr(
                "Couldn't connect to arm controller.... Did you roslaunch fetch_moveit_config move_group.launch?")

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
        self.stow_values = [1.32, 0.7, 0.0, -2.0, 0.0, -0.57, 0.0]
        self.inter_stow_values = [0.7, -0.3, 0.0, -0.3, 0.0, -0.57, 0.0]
        self.zero_values = [0, 0, 0, 0, 0, 0, 0]
        for j, joint in enumerate(self.joint_names):
            self.tuck_joint_value[joint] = self.tuck_values[j]
            self.stow_joint_value[joint] = self.stow_values[j]
            self.inter_stow_joint_value[joint] = self.inter_stow_values[j]
            self.zero_joint_value[joint] = self.zero_values[j]

        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.arm_group = moveit_commander.MoveGroupCommander("arm")
        self.move_group = MoveGroupInterface("arm", "base_link")
        self.eef_link = self.arm_group.get_end_effector_link()
        self.group_names = self.robot.get_group_names()

        print(self.arm_group.has_end_effector_link(), self.arm_group.get_end_effector_link(),
              self.arm_group.get_joints())
        self.arm_group.set_planning_time(20)
        self.arm_group.set_planner_id("RRTConnectkConfigDefault")
        self.arm_group.allow_replanning(True)
        self.arm_group.clear_path_constraints()
        self.scene = moveit_commander.PlanningSceneInterface()

        self.pause = [True]
        '''
        self.scene = Scene(PERCEPTION_MODE)
        self.scene.clear()
        self.thread = thread.start_new_thread(self.scene.update, (self.pause, ))
	    '''

    def move_joint_positions(self, positions):
        self.pause[0] = False
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, positions, 0.02)

    def move_joint_position(self, joint_name, position):
        self.pause[0] = False
        positions = list(self.get_pose())
        positions[self.joint_names.index(joint_name)] = position
        while not rospy.is_shutdown():
            result = self.move_group.moveToJointPosition(self.joint_names, positions, 0.02)

    def plan_joint_position(self, joint_angles):
        self.arm_group.clear_pose_targets()
        self.arm_group.set_joint_value_target(joint_angles)
        plan = self.arm_group.go(wait=True)
        self.arm_group.stop()
        return plan

    def move_cartesian_position(self, position, orientation=[0.0, 0.0, 0.0], constraints=None):
        ''' position: xyz
            orientation: rpy'''
        self.pause[0] = False
        self.arm_group.set_start_state(self.robot.get_current_state())
        quaternion = quaternion_from_euler(orientation[0], orientation[1], orientation[2])

        pose_target = geometry_msgs.msg.Pose()
        pose_target.position.x = position[0]
        pose_target.position.y = position[1]
        pose_target.position.z = position[2]
        pose_target.orientation.w = quaternion[3]
        pose_target.orientation.x = quaternion[0]
        pose_target.orientation.y = quaternion[1]
        pose_target.orientation.z = quaternion[2]
        if constraints is not None:
            self.arm_group.set_path_constraints(constraints)

        self.arm_group.set_pose_target(pose_target)
        plan = self.arm_group.go(wait=True)
        print('plan ', plan)

        self.arm_group.stop()
        # self.arm_group.clear_path_constraints()
        return plan

    def tuck(self, planning=True, **kwargs):  # local method
        p1 = self.add_box()
        print('p1', p1)
        time.sleep(1)
        p2 = self.attach_box()
        print('p2', p2)
        # time.sleep(1)
        if planning:
            result = self.plan_joint_position(self.tuck_joint_value)
        else:
            if kwargs is not None:
                if "max_velocity_scaling_factor" not in kwargs:
                    kwargs["max_velocity_scaling_factor"] = 0.05
            else:
                kwargs = {"max_velocity_scaling_factor": 0.05}
            # while not rospy.is_shutdown():
            plan = self.move_group.moveToJointPosition(self.joint_names, self.tuck_values,
                                                       0.02, **kwargs)
            result = plan.error_code

        p3 = self.detach_box()
        print('p3', p3)
        p4 = self.remove_box()
        print('p4', p4)
        print('plan', result)

    def zero(self, planning=True, **kwargs):
        if planning:
            result = self.plan_joint_position(self.zero_joint_value)
        else:
            if kwargs is not None:
                if "max_velocity_scaling_factor" not in kwargs:
                    kwargs["max_velocity_scaling_factor"] = 0.05
            else:
                kwargs = {"max_velocity_scaling_factor": 0.05}
            # while not rospy.is_shutdown():
            plan = self.move_group.moveToJointPosition(self.joint_names, self.zero_values,
                                                       0.02, **kwargs)
            result = plan.error_code
        print(result)

    def stow(self, planning=True, **kwargs):
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
        print(result)

    def stow_safe(self, planning=True, **kwargs):
        p1 = self.add_box()
        print('p1', p1)
        # time.sleep(1)
        p2 = self.attach_box()
        print('p2', p2)
        # time.sleep(1)
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
        print('p3', p3)
        p4 = self.remove_box()
        print('p4', p4)
        print(result)

    def intermediate_stow(self, planning=True, **kwargs):
        if planning:
            result = self.plan_joint_position(self.inter_stow_joint_value)
        else:
            if kwargs is not None:
                if "max_velocity_scaling_factor" not in kwargs:
                    kwargs["max_velocity_scaling_factor"] = 0.05
            else:
                kwargs = {"max_velocity_scaling_factor": 0.05}
            # while not rospy.is_shutdown():
            plan = self.move_group.moveToJointPosition(self.joint_names, self.inter_stow_values,
                                                       0.02, **kwargs)
            result = plan.error_code
        print(result)

    def get_pose(self):
        self.actual_positions = self.robot.get_current_state().joint_state.position[6:13]
        return self.actual_positions

    def is_there(self, goal, tolerance=0.01):
        actual = self.get_pose()
        all_equal = True
        if type(goal) is list:
            for index in range(len(goal)):
                if abs(actual[index] - goal[index]) > tolerance:
                    return False

        elif type(goal) is geometry_msgs.msg.PoseStamped:
            return self.is_there(goal.pose, actual.pose, tolerance)

        elif type(goal) is geometry_msgs.msg.Pose:
            return self.is_there(pose_to_list(goal), pose_to_list(actual), tolerance)

        return True

    def _del_(self):
        self.tuck()

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

    def detach_box(self, timeout=4):
        self.scene.remove_attached_object(self.eef_link, name=self.box_name)
        return self.wait_for_state_update(box_is_known=True, box_is_attached=False, timeout=timeout)

    def remove_box(self, timeout=4):
        self.scene.remove_world_object(self.box_name)
        return self.wait_for_state_update(box_is_attached=False, box_is_known=False, timeout=timeout)

    def wait_for_state_update(self, box_is_known=False, box_is_attached=False, timeout=8):

        start = rospy.get_time()
        seconds = rospy.get_time()
        while (seconds - start < timeout) and not rospy.is_shutdown():
            attached_objects = self.scene.get_attached_objects([self.box_name])
            is_attached = len(attached_objects.keys()) > 0

            is_known = self.box_name in self.scene.get_known_object_names()

            if (box_is_attached == is_attached) and (box_is_known == is_known):
                return True

            rospy.sleep(0.1)
            seconds = rospy.get_time()

        return False


def to_radian(degree):  # helper function does not work
    rad = round((degree / 180) * pi, 2)
    return rad


if __name__ == '__main__':
    import time
    from torso_control import TorsoControl

    initial_time = time.time()
    rospy.init_node("test_arm")
    torso = TorsoControl()
    arm_module = ArmControl()
    if not torso.move_to(0.3):
        torso.move_to(0.3)
        time.sleep(4)
    if not arm_module.is_there(arm_module.stow_values):
        arm_module.stow_safe()
        time.sleep(4)
    if not arm_module.is_there(arm_module.tuck_values):
        arm_module.tuck()
        time.sleep(2)
    else:
        print('already there')

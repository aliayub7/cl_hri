
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from moveit_python import (MoveGroupInterface,
                           PlanningSceneInterface)
import rospy
import actionlib
import thread
import time
from head_control import HeadControl
from torso_control import TorsoControl

COLLISON_MODE, PERCEPTION_MODE = 0, 1

class Scene():

    def __init__(self, mode=COLLISON_MODE):
        self.mode = mode
        self.scene = PlanningSceneInterface("base_link")
        self.scene.removeCollisionObject("my_front_ground")
        self.scene.removeCollisionObject("my_back_ground")
        self.scene.removeCollisionObject("my_right_ground")
        self.scene.removeCollisionObject("my_left_ground")
        self.scene.addCube("my_front_ground", 2, 1.1, 0.0, -1.0)
        self.scene.addCube("my_back_ground", 2, -1.2, 0.0, -1.0)
        self.scene.addCube("my_left_ground", 2, 0.0, 1.2, -1.0)
        self.scene.addCube("my_right_ground", 2, 0.0, -1.2, -1.0)
        self.head_control = HeadControl()
        self.torso_control = TorsoControl()

        self.perception_results = None
        find_topic = "basic_grasping_perception/find_objects"
        rospy.loginfo("Waiting for %s..." % find_topic)
        self.find_client = actionlib.SimpleActionClient(find_topic, FindGraspableObjectsAction)
        if not self.find_client.wait_for_server():
            rospy.logerr("Could not connect to basic_grasping_perception. Did you rosrun simple_grasping basic_grasping_perception?")
        else:
            rospy.loginfo("Got %s" %find_topic)

    def update(self, pause=[True]):
        # self.explore()
        while 1:
            if pause[0]:
                continue

            ## Triggers the basic perception program and gets the results, i.e graspable objects and their grasp planning
            self.perception_results = None
            # self.explore()
            goal = FindGraspableObjectsGoal()
            goal.plan_grasps = True
            self.find_client.send_goal(goal)
            self.find_client.wait_for_result(rospy.Duration(5.0))
            find_result = self.find_client.get_result()
            if not find_result:
                rospy.logwarn("Basic perception detected no objects :(....")
                continue

            self.clear()
            # self.scene.waitForSync()

            self.scene.addCube("my_front_ground", 2, 1.1, 0.0, -1.0)
            self.scene.addCube("my_back_ground", 2, -1.2, 0.0, -1.0)
            self.scene.addCube("my_left_ground", 2, 0.0, 1.2, -1.0)
            self.scene.addCube("my_right_ground", 2, 0.0, -1.2, -1.0)
            self.scene.addCube("my_base", 0.52, 0.02, 0., .1)
            # self.scene.addCube("my_head", 0.29, 0.02, 0., 1.13 + torso_lift)
            # self.scene.addCube("my_head", 0.29, 0.0, -0.05, -0.15, frame_id="head_camera_rgb_optical_frame")

            if self.mode == COLLISON_MODE:
                k = 0
                for obj in find_result.objects:
                    obj.object.name = "obj_" + str(k)
                    self.scene.addSolidPrimitive(obj.object.name,
                                                 obj.object.primitives[0],
                                                 obj.object.primitive_poses[0],
                                                 use_service=False)
                    k += 1
                    # self.graspable_objects.append(obj)
            k = 0
            for obj in find_result.support_surfaces:
                # extend surface to floor, and make wider since we have narrow field of view
                height = obj.primitive_poses[0].position.z
                obj.primitives[0].dimensions = [obj.primitives[0].dimensions[0],
                                                1.5,  # wider
                                                obj.primitives[0].dimensions[2] + height]
                obj.primitive_poses[0].position.z += -height/2.0
                obj.name = "surf_" + str(k)
                self.scene.addSolidPrimitive(obj.name,
                                             obj.primitives[0],
                                             obj.primitive_poses[0],
                                             use_service=False)
                k += 1

            self.perception_results = find_result
            self.scene.waitForSync()

    def clear(self):
        for name in self.scene.getKnownCollisionObjects():
            self.scene.removeCollisionObject(name, False)
        for name in self.scene.getKnownAttachedObjects():
            self.scene.removeAttachedObject(name, False)

    def explore(self):
        self.head_control.move_home()
        for t_lift in [0.0, 0.05, 0.1, 0.15, 0.2]:
            self.torso_control.move_to(t_lift)
            self.head_control.move_down(.15)
            self.head_control.move_left(.15)
            self.head_control.move_right(.3)
            self.head_control.move_left(.15)
            time.sleep(0.1)

if __name__ == '__main__':
    rospy.init_node("test_scene")
    scene = Scene()
    scene.clear()
    pause = [False]
    thread = thread.start_new_thread(scene.update, (pause, ))
    # time.sleep(5)
    # pause[0] = False
    # time.sleep(5)
    # pause[0] = True
    # time.sleep(5)
    rospy.spin()


from moveit_python import (MoveGroupInterface,
                           PlanningSceneInterface,
                           PickPlaceInterface)
from moveit_msgs.msg import PlaceLocation
import rospy
import numpy as np
from math import pi
from scene import *

class PickPlace():

    def __init__(self):
        rospy.loginfo("Just checking... did you launch move_group?")
        """
        Not needed right now. The following are following
        """
        self.pickplace = PickPlaceInterface("arm", "gripper", verbose=True)
        self.move_group = MoveGroupInterface("arm", "base_link")

        """Not used right now. Will be needed for basic_grasping_perception"""
        self.scene = Scene(PERCEPTION_MODE)

    def pick(self, cube, grasps):
        success, pick_result = self.pickplace.pick_with_retry(cube.name,
                                                              grasps,
                                                              support_name=cube.support_surface,
                                                              scene=self.scene.scene)
        self.pick_result = pick_result
        return success

    def place(self, cube, pose_stamped):
        places = list()
        l = PlaceLocation()
        l.place_pose.pose = pose_stamped.pose
        l.place_pose.header.frame_id = pose_stamped.header.frame_id

        ## copy the posture, approach and retreat from the grasp used
        l.post_place_posture = self.pick_result.grasp.pre_grasp_posture
        l.pre_place_approach = self.pick_result.grasp.pre_grasp_approach
        l.post_place_retreat = self.pick_result.grasp.post_grasp_retreat
        places.append(copy.deepcopy(l))
        ## create another several places, rotate each by 360/m degrees in yaw direction
        m = 16 # number of possible place poses - WHY?
        pi = 3.141592653589
        for i in range(0, m-1):
            l.place_pose.pose = rotate_pose_msg_by_euler_angles(l.place_pose.pose, 0, 0, 2 * pi / m)
            places.append(copy.deepcopy(l))

        success, place_result = self.pickplace.place_with_retry(cube.name,
                                                                places,
                                                                scene=self.scene)
        return success

def picking_func(object_name):
    if object_name is None:
        return
    else:
        from arm_control import ArmControl
        from camera_modules import *
        from torso_control import TorsoControl
        from gripper_control import GripperControl, CLOSED_POS
        import thread
        import sys
        from perception import Perception
        # pp = PickPlace()
        arm = ArmControl()
        yolo = YoloDetector()
        rgb = RGBCamera()
        gripper = GripperControl()
        torso = TorsoControl()
        # perception = Perception()
        # pp = PickPlace()

        torso.move_to(.2)
        arm.stow()
        gripper.open()
        print (object_name)

        while 1:
            raw_input('in front of the table?')
            print(yolo.get_item_list())
            if object_name in yolo.get_item_list():
                print ('YES')
            obj_name = raw_input("Which object do you want to pick?")
            coord_3d, up = yolo.get_item_3d_coordinates(obj_name, rgb.getRGBImage())
            print("Picking " + obj_name + ": ", coord_3d, up)
            raw_input('Good to go?')

            torso.move_to(.4)
            time.sleep(2.0)

            arm.move_cartesian_position([coord_3d[0], coord_3d[1], coord_3d[2] + 0.3], [0, pi/2, 0])
            time.sleep(3.0)
            whatever = raw_input("input to pick")
            arm.move_cartesian_position([coord_3d[0], coord_3d[1], coord_3d[2]+.19], [0, pi/2, 0])
            time.sleep(3.0)
            gripper.close(CLOSED_POS, 60)
            time.sleep(3.0)
            # arm.tuck()
            # #### do stuff or whatever
            arm.stow()
            torso.move_to(.0)
            whatever = raw_input("input to place")
            torso.move_to(.4)
            time.sleep(3.0)
            arm.move_cartesian_position([coord_3d[0]-0.01, coord_3d[1]-0.01, coord_3d[2] + 0.2], [0, pi/2, 0])
            time.sleep(2.0)
            gripper.open()
            time.sleep(2.0)
            arm.stow()
            torso.move_to(.2)

if __name__ == '__main__':
    rospy.init_node("test_pick_place")

    new_dictionary = {0: 'milk', 1: 'apple', 2: 'banana', 3: 'cereal', 4: 'bowl',
    5: 'cup', 6: 'plate', 7: 'fork', 8: 'spoon', 9: 'mug',
    10: 'orange', 11: 'honey'}
    #x_train = np.genfromtxt('/home/fetch/Documents/fetchRobotHighLevelAPI/groceryreminder/objects_picked.csv', delimiter=',')
    #objects_list = []
    #for i in range(0,len(x_train)):
    #    objects_list.append(new_dictionary[x_train[i]])
    #print (objects_list)
    from arm_control import ArmControl
    from camera_modules import *
    from torso_control import TorsoControl
    from gripper_control import GripperControl, CLOSED_POS
    import thread
    import sys
    from perception import Perception
    # pp = PickPlace()
    arm = ArmControl()
    yolo = YoloDetector()
    rgb = RGBCamera()
    gripper = GripperControl()
    torso = TorsoControl()
    perception = Perception()
    # pp = PickPlace()


    #torso.move_to(.4)
    #arm.tuck()
    #torso.move_to(0.1)
    #raw_input('cc')

    torso.move_to(.1)
    arm.stow()
    gripper.open()
    #raw_input('ccc')


    raw_input('in front of the table?')
    #print(yolo.get_item_list())

    #obj_name = raw_input("Which object do you want to pick?")
    #coord_3d, up = yolo.get_item_3d_coordinates(obj_name, rgb.getRGBImage())
    #raw_input('ccc')

    while 1:
        current_image_rgb = rgb.curr_image
        path = '/home/fetch/cognitive_architecture_fetch/grocery_reminder/cur_img_dir'
        cv2.imwrite(path +"/current_image_rgb.png", cv2.cvtColor(current_image_rgb, cv2.COLOR_RGB2BGR)) #robot.getRGBImage()
        cv2.imwrite(path +"/current_image_rgb_detect.png", cv2.cvtColor(yolo.detection_image, cv2.COLOR_RGB2BGR)) #robot.getRGBImage()
        np.savetxt(path+"/current_image_bounding_boxes.csv", yolo.boxes,delimiter=',')

        #np.savetxt(path +"/current_image_bounding_boxes.txt", [arr], delimiter=',', fmt='%d')
        #cv2.imwrite(path +"/current_image_yolo.png", current_image_yolo)
        f = open(path +"/current_image_bounding_boxes.txt", "w")
        print (yolo.boxes)
        my_boxes = yolo.boxes
        my_class_names = yolo.class_names
        for k, box in enumerate(my_boxes):
            obj, _ = perception.get_object(my_class_names[k])
            if obj:
                f.write(my_class_names[k] + "," + ",".join([str(b) for b in box]) + "\n", obj.object.primitive_poses[0].position.x, obj.object.primitive_poses[0].position.y, obj.object.primitive_poses[0].position.z)
            else:
                f.write(my_class_names[k] + "," + ",".join([str(b) for b in box]) + "\n")
        f.close()
        raw_input('images_processed?')

        x_train = np.genfromtxt('/home/fetch/cognitive_architecture_fetch/grocery_reminder/cur_img_lv.csv', delimiter=',')
        object_list = []
        for i in range(0,len(x_train)):
            object_list.append(new_dictionary[x_train[i]])
        print ('this is object list',object_list)

        raw_input('in front of the table?')
        print(yolo.get_item_list())
        obj_name = raw_input("Which object do you want to pick?")
        index = object_list.index(obj_name)
        obj_bbox = my_boxes[index]
        coord_3d, up = yolo.get_item_3d_coordinates(obj_name, rgb.getRGBImage(),obj_box=obj_bbox)
        print("Picking " + obj_name + ": ", coord_3d, up)
        raw_input('Good to go?')

        torso.move_to(.4)
        time.sleep(2.0)

        arm.move_cartesian_position([coord_3d[0], coord_3d[1], coord_3d[2] + 0.3], [0, pi/2, 0])
        time.sleep(3.0)
        whatever = raw_input("input to pick")
        arm.move_cartesian_position([coord_3d[0], coord_3d[1], coord_3d[2]+.19], [0, pi/2, 0])
        time.sleep(3.0)
        gripper.close(CLOSED_POS, 60)
        time.sleep(3.0)
        # arm.tuck()
        # #### do stuff or whatever
        arm.stow()
        torso.move_to(.0)
        whatever = raw_input("input to place")
        torso.move_to(.4)
        time.sleep(3.0)
        arm.move_cartesian_position([coord_3d[0]-0.01, coord_3d[1]-0.01, coord_3d[2] + 0.2], [0, pi/2, 0])
        time.sleep(2.0)
        gripper.open()
        time.sleep(2.0)
        arm.stow()
        torso.move_to(.2)

    # perception.pause[0] = False
    # while 1:
    #     print(perception.get_graspable_objects())
    #     try:
    #         obj_name = perception.get_graspable_objects()[0] #raw_input("Which object do you want to pick?")
    #     except IndexError:
    #         continue
    #     print(obj_name)
    #     obj, grasps = perception.get_object(obj_name)
    #     print(obj)
    #     pp.pick(obj, grasps)
    #     gripper.close()
    #     arm.stow()
    #
    #     whatever = raw_input("input to continue")
    #     pose = PoseStamped()
    #     pose.pose = obj.primitive_poses[0]
    #     pose.pose.position.y *= -1.0
    #     pose.pose.position.z += 0.02
    #     pose.header.frame_id = cube.header.frame_id
    #     pp.place(pose)

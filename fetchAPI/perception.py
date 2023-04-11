
from grasping_msgs.msg import FindGraspableObjectsAction, FindGraspableObjectsGoal
from camera_modules import YoloDetector, Get3Dcoordinates
from moveit_python import (MoveGroupInterface,
                           PlanningSceneInterface,
                           PickPlaceInterface)
import rospy
import actionlib
from scene import *
import thread

## computes distance between two bounding boxes
def distance(c1, c2):
    return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2

## computes distance between two objects
def distance_obj(obj1, obj2):
    return (obj1.object.primitive_poses[0].position.x - obj2.object.primitive_poses[0].position.x)**2 + (obj1.object.primitive_poses[0].position.z - obj2.object.primitive_poses[0].position.z)**2
## computes distance between two objects
def distance_obj3(obj1, obj2):
    return (obj1.object.primitive_poses[0].position.x - obj2.object.primitive_poses[0].position.x)**2 + (obj1.object.primitive_poses[0].position.y - obj2.object.primitive_poses[0].position.y)**2 + (obj1.object.primitive_poses[0].position.z - obj2.object.primitive_poses[0].position.z)**2

## computes intersection between two bounding boxes
def intersection(bb1, bb2):
    x1 = max(min(bb1[0], bb1[2]), min(bb2[0], bb2[2]))
    y1 = max(min(bb1[1], bb1[3]), min(bb2[1], bb2[3]))
    x2 = min(max(bb1[0], bb1[2]), max(bb2[0], bb2[2]))
    y2 = min(max(bb1[1], bb1[3]), max(bb2[1], bb2[3]))
    if (x2 - x1) * (y2 - y1) > 20:
        return True
    return False

## remove overlapping bounding boxes
def overlapping_bb(boxes, names):
    to_remove = []
    for i, b1 in enumerate(boxes):
        for j, b2 in enumerate(boxes):
            if j <= i:
                continue
            inter = intersection(b1, b2)
            if inter:
                to_remove.append(j)
    to_remove = list(set(to_remove))
    # print("ovelap BB", to_remove)
    for k in to_remove[::-1]:
        boxes.pop(k)
        names.pop(k)
    # print(names)
    return boxes, names

## remove objects that are detected twice
def redundant_object(obj):
    # print("started with ", len(obj), " objects")
    to_remove = []
    for i in range(len(obj)):
        for j in range(i+1, len(obj)):
            dist = distance_obj(obj[i], obj[j])
            # print(dist)
            if dist < 0.001:
                to_remove.append(j)
    to_remove = list(set(to_remove))
    # print(to_remove)
    for k in to_remove[::-1]:
        obj.pop(k)
    # print("ended with ", len(obj), " objects")
    return obj

def merge(objs, index, obj2):
    pass
    # new_obj = objs[index]
    # new_obj.object.primitive_poses[0].position.x = (new_obj.object.primitive_poses[0].position.x + obj2.object.primitive_poses[0].position.x) / 2.0
    # new_obj.object.primitive_poses[0].position.x = (new_obj.object.primitive_poses[0].position.y + obj2.object.primitive_poses[0].position.y) / 2.0
    # new_obj.object.primitive_poses[0].position.x = (new_obj.object.primitive_poses[0].position.z + obj2.object.primitive_poses[0].position.z) / 2.0
    # new_obj.primitives[0].dimensions[0] = max()
    # objs[index] = new_obj

## remove objects that are already in the scene
def redundant_objects(current, detected):
    to_merge, to_remove = [], []
    for i, obj1 in enumerate(current):
        for j, obj2 in enumerate(detected):
            dist = distance_obj3(obj1, obj2)
            if dist < 0.01:
                to_merge.append((i,j))
                to_remove.append(j)
    # for i,j in to_merge:
    #     current.append(merge(current, i, detected[j]))
    for j in to_remove[::-1]:
        detected.pop(j)
    for d in detected:
        current.append(d)
    return current

class Perception():

    def __init__(self):
        self.scene = Scene(PERCEPTION_MODE)

        self.object_detector = YoloDetector()
        self._2dto3d = Get3Dcoordinates()
        self.graspable_objects = []
        self.graspable_objects_dict = {}
        self.pause = [True]
        self.thread = thread.start_new_thread(self.scene.update, (self.pause, ))
        self.thread2 = thread.start_new_thread(self.loop, ())

    def loop(self):
        while 1:
            if self.pause[0]:
                continue
            self.updateScene()

    def updateScene(self):
        self.graspable_objects = []
        find_result = self.scene.perception_results



        ## get Yolo results, remove overlapping boxes
        boxes, names, img = self.object_detector.detect()
        # boxes, names = overlapping_bb(boxes, names)
        boxes_3d = {}
        for k, b in enumerate(boxes):
            box = self._2dto3d.pixelTo3DPoint((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            if box:
                boxes_3d[names[k]] = box
        # print(boxes_3d)

        if not find_result:
            return
        print("objects detected!!")

        final_objects = []
        for obj in find_result.objects:
            if obj.object.primitive_poses[0].position.x < 1.1:
                final_objects.append(obj)

        ## remove false detection
        # final_objects = redundant_object(final_objects)

        # Assign names from Yolo to the basic_grasping_perception objects and display
        for k, obj in enumerate(final_objects):
            dist = 50000
            for key in boxes_3d:
                _dist = distance(boxes_3d[key], [obj.object.primitive_poses[0].position.x, obj.object.primitive_poses[0].position.y, obj.object.primitive_poses[0].position.z])
                # print(key, _dist)
                if _dist < dist:
                    dist = _dist
                    obj.object.name = key
            if obj.object.name == '':
                obj.object.name = "unknown"
            print(obj.object.name, obj.object.primitive_poses[0].position)
            self.scene.scene.addSolidPrimitive(obj.object.name,
                                         obj.object.primitives[0],
                                         obj.object.primitive_poses[0],
                                         use_service=False)
            self.graspable_objects.append(obj)
            self.graspable_objects_dict[obj.object.name] = obj

        self.scene.scene.waitForSync()

    def get_object(self, name):
        if name in self.graspable_objects_dict:
            return self.graspable_objects_dict[name].object, self.graspable_objects_dict[name].grasps
        else:
            print("No object detected with name: " + name)
            return None, None

        # obj = None
        # for o in self.graspable_objects:
        #     if o.object.name == name:
        #         obj = o
        #         break
        # if not obj:
        #     print("No object detected with name: " + name)
        #     return None, None
        # return obj.object, obj.grasps

    def get_graspable_objects(self):
        return self.graspable_objects_dict.keys()
        # objs = []
        # for obj in self.graspable_objects:
        #     objs.append(obj.object.name)
        # return objs


if __name__ == '__main__':
    rospy.init_node("test_perception")
    perception = Perception()
    perception.pause[0] = False
    rospy.spin()

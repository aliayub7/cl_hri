import cv2
import sys
import rospy
from sensor_msgs.msg import Image
from PyQt5 import QtGui, QtCore
import time
import math
from darknet_ros_msgs.msg import BoundingBoxes
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
import struct
import numpy as np
from geometry_msgs.msg import PointStamped
#
import tf2_py as tf2
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud, transform_to_kdl
import PyKDL

def do_transform_point(point, transform):
    p = transform_to_kdl(transform) * PyKDL.Vector(point.point.x, point.point.y, point.point.z)
    res = PointStamped()
    res.point.x = p[0]
    res.point.z = p[2]
    res.point.y = p[1]
    res.header = transform.header
    return res
    # tf2_py.TransformRegistration().add(PointStamped, do_transform_point)


def segmentation(img, bb):
    _img = img[max(0, bb[1]-10):bb[3]+10, max(0,bb[0]-10):bb[2]+10]

    gray = cv2.cvtColor(_img,cv2.COLOR_RGB2GRAY)
    edged=cv2.Canny(gray,150,900)

    edges = np.zeros((480,640), np.uint8)
    edges[max(0, bb[1]-10):bb[3]+10, max(0,bb[0]-10):bb[2]+10] = edged
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[-2], key=cv2.contourArea)
    list_of_pts = []
    for ctr in cnt:
        list_of_pts += [pt[0] for pt in ctr]

    ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
    ctr = cv2.convexHull(ctr) # done.

    mask = np.zeros((480,640), np.uint8)
    masked = cv2.drawContours(mask, [ctr],-1, 255, -1)
    segmented = cv2.bitwise_and(img, img, mask=mask)
    return segmented

from numpy import concatenate
class YoloDetector():
    def __init__(self):
        ##Subscribe to Yolo topics.
        ## IMPORTANT: run "roslaunch darknet_ros darknet_ros" before!!
        if "/darknet_ros/check_for_objects/feedback" not in concatenate(rospy.get_published_topics()):
            rospy.logerr("darknet topics do not exist. run roslaunch darknet_ros darknet_ros before!")
            sys.exit(1)
        self._sub_bb = rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.cb)
        self._sub_img = rospy.Subscriber("/darknet_ros/detection_image", Image, self.cb)
        self.boxes = []
        self.items = {}
        self.class_names = []
        self.detection_image = None
        self._2dto3d = Get3Dcoordinates()
        # self.trans = None
        rospy.loginfo("YOLO initialised!!")

    ## callback function
    def cb(self, data):
        if type(data) is Image:
            self.detection_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        elif type(data) is BoundingBoxes:
            self.boxes = []
            self.class_names = []
            for box in data.bounding_boxes:
                self.boxes.append([box.xmin, box.ymin, box.xmax, box.ymax, box.probability])
                self.class_names.append(box.Class)
                # self.items[box.Class] = [int(box.xmin+box.xmax)//2, int(box.ymin+box.ymax)//2, box.probability]
                self.items[box.Class] = [box.xmin, box.xmax, box.ymin, box.ymax, box.probability]

    ## some stuff that may be useful in the future....
    def detect(self):
        # for i, b in enumerate(self.boxes):
        #     if b[4] > .75:
        #         self.items[self.class_names[i]] = b[0],b[1],time.time()
        #
        # time_now = time.time()
        # for item in self.items.keys():
        #     if abs(time_now-self.items[item][2]) > 60:
        #         del self.items[item]
        #
        # try:
        #     self.schemaNetwork.update_context_items(self.items.keys())
        # except:
        #     pass
        return self.boxes, self.class_names, self.detection_image

    def clear(self):
        self.boxes = None
        self.class_names = None
        self.detection_image = None

    def getDetectionImage(self):
        return self.detection_image

    def get_item_coordinates(self,item_name):
        if item_name in self.items:
            return self.curr_image.shape,(self.items[item_name][0],self.items[item_name][2]),self.items[item_name][2]
        else:
            return None,None,None

    def get_item_3d_coordinates(self,i_name, rgb_img,obj_box=None):

        for k, [item_name, box] in enumerate(zip(self.class_names, self.boxes)):
            if not (i_name==item_name):
                continue

            box = obj_box
            mask = segmentation(rgb_img, box)
            pc = []
            object_loc = np.argwhere(mask[:,:,0] > 0)
            w, h = box[2] - box[0], box[3] - box[1]

            """
            #for pix in object_loc:
            #    c = self._2dto3d.pixelTo3DPoint(pix[1], pix[0])
            #    if c:
            #        pc.append(c)
            #pc = np.array(pc)
            #coord_3d = np.median(pc, axis = 0)
            """

            #up_3d = [coord_3d[0], coord_3d[1], np.percentile(pc[:,2], 95)]
            print(self._2dto3d.pixelTo3DPoint(int(box[0] + box[2])//2,
                                                int(box[1] + box[3])//2))
            coord_3d = self._2dto3d.pixelTo3DPoint(int(box[0] + box[2])//2,
                                                int(box[1] + box[3])//2)
            up_3d = self._2dto3d.pixelTo3DPoint(int(box[0] + box[2])//2 + w*.25,int(box[1] + box[3])//2 + h *.25)
            if coord_3d[0] > 0.9:
                continue
            else:
                return coord_3d, up_3d

        return None,None


    def clicked(self, x, y):
        # print(x, y)
        for k, box in enumerate(self.boxes):
            # print(box)
            if x > box[0] and x < box[2] and y > box[1] and y < box[3]:
                print("clicked: ", self.class_names[k])
                return self.class_names[k]
        return ""

    def get_item_list(self):
        return self.class_names


    ## some stuff that may be useful in the future....
    # def process_grabbable_objects(self):
    #     for item in self.items.keys():
    #             if item in self.schemaNetwork.possible_grabbable_items and item in self.items and (time.time()-self.items[item][2]) < .1:
    #                 print ("finding " + item)
    #                 finder3D = Get3Dcoordinates(int(round(640*self.items[item][0])),int(round(480*self.items[item][1])))
    #                 while True:
    #                     print("processing")
    #                     if finder3D.found_3d:
    #                         break
    #                 target_x = finder3D.map_point.point.x
    #                 target_y = finder3D.map_point.point.y
    #                 target_z = finder3D.map_point.point.z
    #
    #                 del finder3D
    #
    #                 if not math.isnan(target_x) and not math.isnan(target_y):
    #                     self.schemaNetwork.update_current_schema(item,[target_x,target_y])
    #                 else:
    #                     print("not detected")

class RGBCamera():
    def __init__(self):
        ## Subscribe to the camera of the robot
        topic_name = '/head_camera/rgb/image_raw'
        self.curr_image = None
        self._image_sub = rospy.Subscriber(topic_name, Image, self._color_image_cb)

    def _color_image_cb(self, data):
        self.curr_image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

    def save_image(self):
        if self.curr_image is not None:
            cv2.imwrite( "./trainingset/"+str(time.time())+".jpg", self.curr_image);

    def getRGBImage(self):
        try:
            return cv2.cvtColor(self.curr_image, cv2.COLOR_BGR2RGB)
        except:
            return None


class Get3Dcoordinates(object):
    """Supposed to read a point cloud snapshot and returns the XYZ coordinates of a corresponding pixel location
    Doesn't work....."""

    def __init__(self):
        ## Subscribe point cloud
        self.cloud = PointCloud2()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        sub_once = rospy.Subscriber('/head_camera/depth_registered/points', PointCloud2,
                                    self.callback)
        ## Wait until connection
        rospy.wait_for_message('/head_camera/depth_registered/points', PointCloud2, timeout=100.0)

    def callback(self, msg):
        self.cloud = msg

        self.trans = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id,
                                           #msg.header.stamp,
                                           rospy.Time().now(),
                                           rospy.Duration(1.0))

    ## transform 2D point into 3D coordinates in the "base_link" frame
    def pixelTo3DPoint(self, u, v, cloud=None):
        gen = pc2.read_points(self.cloud, field_names=("x", "y", "z"), skip_nans=True, uvs=[[int(u), int(v)]])
        try:
            target_xyz_cam = list(gen)
        except:
            return None

        # do conversion to global coordinate here
        rgbd_point = PointStamped()
        rgbd_point.header.frame_id = "head_camera_rgb_optical_frame"
        rgbd_point.header.stamp = rospy.Time(0)
        try:
            rgbd_point.point.x = target_xyz_cam[0][0]
        except IndexError:
            return None
        rgbd_point.point.y = target_xyz_cam[0][1]
        rgbd_point.point.z = target_xyz_cam[0][2]

        map_point = do_transform_point(rgbd_point, self.trans)

        return [map_point.point.x, map_point.point.y, map_point.point.z]

if __name__ == '__main__':
    rospy.init_node("test_cameras")
    rgb = RGBCamera()
    yd = YoloDetector()

    print("Testing cameras")
    while 1:
        try:
            cv2.imshow("window_rgb", rgb.getRGBImage())
        except:
            continue
        try:
            cv2.imshow("window_dn", yd.getDetectionImage())
        except:
            continue
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

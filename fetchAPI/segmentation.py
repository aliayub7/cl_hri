import cv2
import numpy as np
import glob

# roi = {"cup_front_rgb": [90:-90, 90:-110],
#         "all_front_rgb_banana": [200:-10, 60:-100],
#         "all_front_rgb_apple": [140:-70, 160:-50],
#         "all_front_rgb_plate": 20:-60, 60:-100],
#         "all_front_rgb_bowl": [100:-90, 10:-180],
#         "all_front_rgb_cup": [60:-140, 75:-145],
#         "all_front_rgb_cup2": [80:-100, 160:-30],
#         "all_front_rgb_milk": [10:-140, 130:-60]
#           "marker_0_0_rgb": [90:-30, 170:-30]
# }

def segmentation(path, bb_path):
    print(path)
    img = cv2.imread(path)
    file = open(bb_path, "r")
    bb = file.readline().split(",")[1:5]
    bb = [int(b) for b in bb]
    print(bb)
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

    # cv2.imshow("segmented", segmented)
    # cv2.waitKey(0)

path = "/home/furhat/Documents/perception/FetchHighLevelAPI/data/"
# objects = os.listdir(path)

files = glob.glob(path + '/**/*_rgb.png', recursive=True)
np.random.shuffle(files)

for object in files:
    bb_file = "_".join(object.split("_")[:-1]) + "_bounding_boxes.txt"
    # print(bb_file)
    segmentation(object, bb_file)

# object = []
# for u,v in np.argwhere(masked):
#     object.append(lookup3dcoordinates(u,v))
# object = np.array(object)
#
# average_coord = np.mean(object, axis=1)
# px_5, px_95 = np.percentile(object[0,:], [5, 95])
# py_5, py_95 = np.percentile(object[1,:], [5, 95])
# pz_5, pz_95 = np.percentile(object[2,:], [5, 95])

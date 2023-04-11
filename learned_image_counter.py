from genericpath import isfile
import os
directory = os.getcwd()
IMAGE_FOLDER = "/fetchGUI_learning_testing/learned_objects"
image_directory = directory+IMAGE_FOLDER
print(os.listdir(os.path.join(image_directory)))

images_per_part = [0]*60
for part in os.listdir(os.path.join(image_directory)):
    part_path = os.path.join(image_directory,part)
    images = 0
    if os.path.isdir(part_path):
        for session in os.listdir(part_path):
            if session.isdigit():
                session_path = os.path.join(part_path,session)
                for object_images in os.listdir(session_path):
                    image_files = os.listdir(os.path.join(session_path,object_images))
                    images += len(image_files)//3
    images_per_part[int(part)] = images
    images = 0

for images,part in zip(images_per_part, range(len(images_per_part))):
    print("Part: {} Num: {}".format(part + 1, images))



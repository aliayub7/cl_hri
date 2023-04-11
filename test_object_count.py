from genericpath import isfile
import os
directory = os.getcwd()
TEST_FOLDER = "/fetchGUI_learning_testing/test_objects"
test_directory = directory+TEST_FOLDER
print(os.listdir(os.path.join(test_directory)))

tests_per_part_per_session = [0]*60
for part in os.listdir(os.path.join(test_directory)):
    part_path = os.path.join(test_directory,part)
    images = [0]*5
    if os.path.isdir(part_path):
        for session in os.listdir(part_path):
            if session.isdigit():
                session_path = os.path.join(part_path,session)
                images[int(session)] = len(os.listdir(session_path))//3
    tests_per_part_per_session[int(part)] = images

for images,part in zip(tests_per_part_per_session, range(len(tests_per_part_per_session))):
    try:
        print("Part: {} Num: {} | {} | {} | {} | {}".format(part + 1, images[0], images[1], images[2], images[3], images[4]))
    except:
        pass




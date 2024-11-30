import os
import random
import shutil

"""This code only run once to change bounding box formating 
from normal format: [x_min, y_min, x_max, y_max]
to yolo format: [x_center, y_center, width, height]
"""


list = os.listdir("./Detection/Labels/")
list = [value.replace(".txt", "") for value in list]


def split_array_by_percentage(arr, percentage):
    # Ensure the percentage is between 0 and 100
    percentage = max(0, min(100, percentage))

    # Calculate the split index
    split_index = int(len(arr) * (percentage / 100))

    # Create a copy of the array and shuffle it
    shuffled_arr = arr.copy()
    random.shuffle(shuffled_arr)

    # Split the shuffled array
    first_part = shuffled_arr[:split_index]
    second_part = shuffled_arr[split_index:]

    return first_part, second_part


train, second = split_array_by_percentage(list, 70)
valid, test = split_array_by_percentage(second, 60)


def move_file(source, destination):
    try:
        shutil.move(source, destination)
        print(f"File moved successfully from {source} to {destination}")
    except FileNotFoundError:
        print(f"The file {source} does not exist")
    except PermissionError:
        print("Permission denied.")
    except shutil.Error as e:
        print(f"Error occurred while moving file: {e}")


for i in range(len(train)):
    image_in_dir = os.path.join("./Detection/Images/", train[i] + ".jpg")
    image_out_dir = os.path.join("./Detection/Images/train", train[i] + ".jpg")
    move_file(image_in_dir, image_out_dir)
    label_in_dir = os.path.join("./Detection/Labels/", train[i] + ".txt")
    label_out_dir = os.path.join("./Detection/Labels/train", train[i] + ".txt")
    move_file(label_in_dir, label_out_dir)

for i in range(len(test)):
    image_in_dir = os.path.join("./Detection/Images/", test[i] + ".jpg")
    image_out_dir = os.path.join("./Detection/Images/test", test[i] + ".jpg")
    move_file(image_in_dir, image_out_dir)
    label_in_dir = os.path.join("./Detection/Labels/", test[i] + ".txt")
    label_out_dir = os.path.join("./Detection/Labels/test", test[i] + ".txt")
    move_file(label_in_dir, label_out_dir)

for i in range(len(valid)):
    image_in_dir = os.path.join("./Detection/Images/", valid[i] + ".jpg")
    image_out_dir = os.path.join("./Detection/Images/valid", valid[i] + ".jpg")
    move_file(image_in_dir, image_out_dir)
    label_in_dir = os.path.join("./Detection/Labels/", valid[i] + ".txt")
    label_out_dir = os.path.join("./Detection/Labels/valid", valid[i] + ".txt")
    move_file(label_in_dir, label_out_dir)

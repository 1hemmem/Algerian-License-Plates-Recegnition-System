import os
from PIL import Image


def create_yolo_txt(file_path, image_path, yolo_text_path):
    image = Image.open(image_path)
    width = image.width
    height = image.height
    try:
        with open(file_path, "r") as file:
            num_boxes = int(file.readline().strip())
            for i in range(num_boxes):
                values = file.readline().split()
                x_min = int(values[0])
                y_min = int(values[1])
                x_max = int(values[2])
                y_max = int(values[3])
                box_width = float((x_max - x_min) / width)
                x_center = float((box_width / 2) + (x_min / width))
                box_height = float((y_max - y_min) / height)
                y_center = float((box_height / 2) + (y_min / height))
                try:
                    with open(yolo_text_path, "a") as yolo:
                        yolo.write(
                            f"0 {x_center:.4f} {y_center:.4f} {box_width:.4f} {box_height:.4f}\n"
                        )
                except KeyError:
                    print(KeyError)
    except KeyError:
        print(KeyError)


textfiles_dir = "./Detection/Labels/010/"
images_dir = "./Detection/Images/010/"
output_dir = "./Detection/Labels0/010"
textfiles_list = os.listdir(textfiles_dir)
images_list = os.listdir(images_dir)



for i in range(len(textfiles_list)):
    # textfile path
    textfile = os.path.join(textfiles_dir,textfiles_list[i])
    # image path
    image_path = os.path.join(images_dir,images_list[i])
    # yolo filetext path
    yolo_textfile = os.path.join(output_dir,textfiles_list[i])    
    create_yolo_txt(textfile,image_path,yolo_textfile)

import cv2

"""
I have made this script to test the yolo bounding
boxes of the yolo formatt and see if it is correct
"""


# Read the bounding box coordinates from the text file
with open(
    "/home/bahaeddine09/Programming/License_rec/Detectionv1/Labels/valid/0_8.txt", "r"
) as file:
    data = file.readline().strip().split()

# Extract values (assuming the format: class_id, x_center, y_center, width, height)
class_id, x_center, y_center, width, height = map(float, data)

# Load the image
image = cv2.imread(
    "/home/bahaeddine09/Programming/License_rec/Detectionv1/Images/valid/0_8.jpg"
)
img_height, img_width = image.shape[:2]

# Convert YOLO format (normalized) to pixel coordinates
x_center *= img_width
y_center *= img_height
width *= img_width
height *= img_height

# Calculate the top-left and bottom-right corners of the bounding box
x1 = int(x_center - width / 2)
y1 = int(y_center - height / 2)
x2 = int(x_center + width / 2)
y2 = int(y_center + height / 2)

# Draw the bounding box on the image
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the image
cv2.imshow("Bounding Box", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

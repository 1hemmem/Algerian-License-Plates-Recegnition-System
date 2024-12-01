# Algerian License Plate Recegnition System

[Watch the demo video](https://youtu.be/HvMSLZRUBi8)
## Project Description
The goal of this project is to develop a comprehensive end-to-end system that detects cars and their license plates, converts the license plate information into text using Optical Character Recognition (OCR), and stores both the high-quality images of the cars and license plates along with the extracted text and a timestamp indicating when the car passed the camera.

### Challenges
- The object detection model (in this case YOLO) works per frame and do not care if the same (car/license plate) is detected twice, **where in our case we care so much!!**
- Working with the object tracking and re-identification (DeepSORT)
- Making the OCR work well required us to ensure that we feed it the best quality image that have a readable license plate during it's path in the frame.

### Lessons

- Simplicity is key: keeping solutions as simple as possible reduces complexity and improves maintainability.
- Embracing Object-Oriented Programming (OOP) can save significant time and effort when implemented correctly.      
- RTFM: reading the flipping manual also will save much time, WALLAH !!
  
### Dataset
You can find the dataset I have worked with in this link:
https://github.com/mouadb0101/License_Plates_of_Algeria_Dataset

## Setup

clone the repository:
```
git clone git@github.com:1hemmem/Algerian-License-Plates-Recegnition-System.git
```
install the required packages:
```
cd src/
pip install -r requirements.txt
```

Run the application on your video:
```
cd prod/ && python app.py --path="your/path/to/video.mp4"
```
  



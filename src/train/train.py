from ultralytics import YOLO


print("#### loading model ####")
model = YOLO("../../models/yolov10n.pt")
print("#### model loaded ####")
model.train(
    data="/home/bahaeddine09/Programming/License_rec/Detection/Data.yaml",
    epochs=10,
    batch=1,
)

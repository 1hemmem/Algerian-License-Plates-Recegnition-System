import logging
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from GeminiOcr import GeminiOCR 
from prod.Utils import Utils
from prod.Filter import ImageProcessor, LicensePlateDetector


def main():
    # Initialize logger
    logger = logging.getLogger(__name__)

    # Initialize the ocr model
    ocr = GeminiOCR(
        api_key=os.environ["GEMINI_API_KEY"], model_name="gemini-1.5-pro"
    )

    # Initialize modes
    model = YOLO("../../models/yolov10n.pt")
    lpdetector = LicensePlateDetector(
        "../../models/best.pt"
    )
    
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        nn_budget=100,
    )

    cap = cv2.VideoCapture("../../../Iphone_data/output1.mp4")
    if not cap.isOpened():
        logger.error("Error: Could not open video file")
        return


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("../../../output.mp4", fourcc, 30, (frame_width, frame_height))

    min_area_threshold = 10000
    utils = Utils()
    processor = ImageProcessor()

    # Dictionary to store tracked vehicles and their status
    sharpness = {}
    completed_tracks = (
        set()
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.7)
            detections = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf.cpu().numpy()[0])
                    xyxy = box.xyxy.cpu().numpy()[0]
                    cls_id = int(box.cls.cpu().numpy()[0])

                    if cls_id in [2, 4, 6, 8]:  # Vehicle class IDs
                        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                        area = (x2 - x1) * (y2 - y1)

                        if area >= min_area_threshold:
                            detection = [[x1, y1, x2 - x1, y2 - y1], conf, cls_id]
                            detections.append(detection)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if detections:
                tracks = tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id

                    # Skip if this track has already reached the bottom
                    if track_id in completed_tracks:
                        plate_number = ocr.extract_text_from_image(image_path=completed_tracks[track_id]["plate_path"])
                        path,_ = os.path.split(completed_tracks[track_id]["plate_path"])
                        os.rename(completed_tracks[track_id]["plate_path"],os.path.join(path,plate_number+".jpg"))
                        continue

                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]

                    # Check if vehicle has reached the bottom of the frame
                    if y2 >= frame_height - 50:  # Adding a small buffer of 10 pixels
                        completed_tracks.add(track_id)
                        continue

                    # Draw tracked box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2,
                    )

                    # Extract vehicle image and calculate quality metrics
                    vehicle_image = frame[y1:y2, x1:x2]
                    detection_sucess, plate_image, coord_plate = (
                        lpdetector.detect_plate(vehicle_image)
                    )

                    if detection_sucess:
                        if vehicle_image.size == 0:
                            logger.warning("empty vehicule image detected")
                            continue
                        if plate_image.size == 0:
                            logger.warning("empty plate image detected")
                            continue

                        metrics = processor.calculate_quality_metrics(plate_image)
                        current_score = metrics.total_score
                        # print(current_score)

                        if (
                            track_id not in sharpness
                            or current_score > sharpness[track_id]["score"]
                        ):
                            if track_id in sharpness:
                                os.remove(sharpness[track_id]["image_path"])
                                os.remove(sharpness[track_id]["plate_path"])

                            image_path = utils.save_bounding_box_image(
                                frame,
                                [x1, y1, x2, y2],
                                track_id,
                                track.age,
                                isPlate=False,
                            )

                            plate_path = utils.save_bounding_box_image(
                                vehicle_image,
                                coord_plate,
                                track_id,
                                track.age,
                                isPlate=True,
                            )

                            sharpness[track_id] = {
                                "score": current_score,
                                "image_path": image_path,
                                "plate_path": plate_path,
                            }

            out.write(frame)
            cv2.imshow("Vehicle Detection and Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import argparse
import logging
import cv2
import numpy as np
import asyncio
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
from GeminiOcr import GeminiOCR
from Utils import Utils
from Filter import ImageProcessor, LicensePlateDetector
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(video_path):
    # Initialize the OCR model
    ocr = GeminiOCR(api_key=os.environ["GEMINI_API_KEY"], model_name="gemini-1.5-pro")

    # Initialize YOLO and license plate detector models
    model = YOLO("../../models/yolov10n.pt")
    lpdetector = LicensePlateDetector("../../models/best.pt")

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        max_iou_distance=0.7,
        max_cosine_distance=0.3,
        nn_budget=100,
    )
    if video_path is None:
        video_path = "../../../Iphone_data/output1.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Error: Could not open video file")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        "../../../output.mp4", fourcc, 60, (frame_width, frame_height)
    )

    min_area_threshold = 10000
    utils = Utils()
    processor = ImageProcessor()

    # Dictionaries to store tracked vehicles and their status
    sharpness = {}
    completed_tracks = {}

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
                            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if detections:
                tracks = tracker.update_tracks(detections, frame=frame)
                for track in tracks:
                    if not track.is_confirmed():
                        continue

                    track_id = track.track_id

                    # Process completed tracks
                    if (track_id in completed_tracks) and (
                        completed_tracks[track_id]["ocred"] == False
                    ):
                        try:
                            plate_path = completed_tracks[track_id]["plate_path"]

                            # First check if the file exists
                            if not os.path.exists(plate_path):
                                logger.warning(f"Plate image not found: {plate_path}")
                                completed_tracks[track_id]["ocred"] = True
                                continue

                            plate_number = await ocr.extract_text_from_image(
                                image_path=plate_path
                            )

                            # Only try to rename if we got a valid plate number
                            if plate_number and isinstance(plate_number, str):
                                path, _ = os.path.split(plate_path)
                                now = datetime.datetime.now()
                                current_hour = now.hour
                                current_minute = now.minute
                                current_second = now.second
                                current_date = now.date()
                                new_path = os.path.join(
                                    path,
                                    f"time: {current_hour}:{current_minute}:{current_second}.{current_date}_plate_number: {plate_number}.jpg",
                                )

                                os.rename(plate_path, new_path)
                            else:
                                logger.warning(
                                    f"Invalid plate number returned for track {track_id}: {plate_number}"
                                )

                            completed_tracks[track_id]["ocred"] = True

                        except FileNotFoundError as e:
                            logger.error(f"Error processing file {plate_path}: {e}")
                            completed_tracks[track_id]["ocred"] = True
                        except Exception as e:
                            logger.error(
                                f"Unexpected error processing track {track_id}: {e}"
                            )
                            completed_tracks[track_id]["ocred"] = True

                        continue

                    bbox = track.to_tlbr()
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]

                    # Check if vehicle has reached the bottom of the frame
                    if y2 >= frame_height - 50 and track_id in sharpness:
                        completed_tracks[track_id] = sharpness[track_id]
                        continue

                    # Draw tracked box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cv2.putText(
                    #     frame,
                    #     f"ID: {track_id}",
                    #     (x1, y1 - 10),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     1,
                    #     (255, 255, 255),
                    #     2,
                    # )

                    # Extract vehicle image and calculate quality metrics
                    vehicle_image = frame[y1:y2, x1:x2]
                    if vehicle_image.shape[0] != 0:
                        detection_success, plate_image, coord_plate = (
                            lpdetector.detect_plate(vehicle_image)
                        )

                    if detection_success:
                        if vehicle_image.size == 0:
                            logger.warning("Empty vehicle image detected")
                            continue
                        if plate_image.size == 0:
                            logger.warning("Empty plate image detected")
                            continue

                        metrics = processor.calculate_quality_metrics(plate_image)
                        current_score = metrics.total_score

                        # Update the sharpness dictionary for the best-quality image
                        if (
                            track_id not in sharpness
                            or current_score > sharpness[track_id]["score"]
                        ):

                            if track_id in sharpness:
                                os.remove(sharpness[track_id]["image_path"])
                                try:
                                    os.remove(sharpness[track_id]["plate_path"])
                                except FileNotFoundError as e:
                                    logger.error(f"Error deleting file {e}")

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
                                "ocred": False,
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
    parser = argparse.ArgumentParser(description="Vehicle detection and tracking.")
    parser.add_argument(
        "--path", type=str, required=False, help="Path to the input video file."
    )
    args = parser.parse_args()

    asyncio.run(main(args.path))

import cv2
import os

class Utils:
    @staticmethod
    def save_bounding_box_image(frame, bbox, track_id, age, isPlate=False):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within frame boundaries
        height, width = frame.shape[:2]
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(0, min(x2, width-1))
        y2 = max(0, min(y2, height-1))
        
        # Check if box has valid dimensions
        if x2 <= x1 or y2 <= y1:
            print(f"Invalid box dimensions for track {track_id}: [{x1}, {y1}, {x2}, {y2}]")
            return
            
        # Extract region of interest
        roi = frame[y1:y2, x1:x2]
        
        # Verify ROI is not empty
        if roi.size == 0:
            print(f"Empty ROI for track {track_id}: [{x1}, {y1}, {x2}, {y2}]")
            return

        # Create output directory if it doesn't exist
        output_dir = "../../../official_output/"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save image with track ID and age
        if isPlate:
            filename = f"{output_dir}/plate_track_{track_id}_frame_{age}.jpg"
        else:
            filename = f"{output_dir}/car_track_{track_id}_frame_{age}.jpg"
        try:
            cv2.imwrite(filename, roi)
        except Exception as e:
            print(f"Error saving image for track {track_id}: {e}")
        return filename

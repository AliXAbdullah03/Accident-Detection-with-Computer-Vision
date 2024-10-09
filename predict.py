import os
import cv2
from ultralytics import YOLO
import time

# Define directories and file paths
VIDEOS_DIR = "videos\\Accident Detection videos"
OUTPUT_DIR = "output_frames"  # Folder to save frames with detected incidents
model_path = 'runs/detect/train8/weights/best.pt'

# Check for model existence before proceeding
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found.")
    exit()

# Load the YOLO model
model = YOLO(model_path)

# Rate limiting and highest confidence tracking
notification_interval = 60  # in seconds, adjust as needed
highest_confidence = 0
best_frame_info = None
interval_started = False
start_time = 0  # Initialize to zero

# Function to process a specific video
def process_video(video_path):
    global highest_confidence, best_frame_info, interval_started, start_time
    video_name = os.path.basename(video_path)  # Extract video name from the path
    output_dir = os.path.join(OUTPUT_DIR, video_name[:-4])  # Folder for each video

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        return

    frame_count = 0  # Initialize frame counter

    # Process each frame
    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        # Perform object detection on the frame
        results = model(frame, show=False, conf=0.4, save=False)
        result = results[0]  # Get the first result since we are processing frame by frame

        # Evaluate detected incidents
        for box in result.boxes:
            cls_index = int(box.cls.item())  # Convert tensor to integer
            detected_class = model.names[cls_index]  # Use the integer index to access the class name
            conf = box.conf.item()  # Get the confidence level of the detection
            if detected_class in ["severe", "moderate"] and conf > highest_confidence:
                if not interval_started:
                    start_time = current_time  # Start the interval upon first valid detection
                    interval_started = True
                highest_confidence = conf
                best_frame_info = (frame_count, frame, detected_class)

        # Check if the interval has started and time limit is reached
        if interval_started and (current_time - start_time >= notification_interval):
            # Save the best frame if there's a valid one
            if best_frame_info:
                frame_output_path = os.path.join(output_dir, f"{best_frame_info[0]}_{best_frame_info[2]}.jpg")
                cv2.imwrite(frame_output_path, best_frame_info[1])
            # Reset variables for the next interval
            highest_confidence = 0
            best_frame_info = None
            interval_started = False

        frame_count += 1  # Increment frame counter

    # Check if there's any remaining best frame information when the video ends
    if best_frame_info:
        frame_output_path = os.path.join(output_dir, f"{best_frame_info[0]}_{best_frame_info[2]}.jpg")
        cv2.imwrite(frame_output_path, best_frame_info[1])

    # Release the video capture
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Process all videos in the directory
def process_all_videos(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".avi"):  # Add other video file extensions if needed
            video_path = os.path.join(directory, filename)
            process_video(video_path)

# Call the function with the videos directory path
process_all_videos(VIDEOS_DIR)

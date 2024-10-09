# Accident-Detection-with-Computer-Vision
This project implements an accident detection system using a custom-trained YOLOv8 model. The model processes input frames to accurately detect accidents in real-time and generates notifications accordingly. This innovative solution enhances road safety by providing timely alerts and situational awareness.

Here's a detailed description for your GitHub repository based on the provided code:

# Accident Detection Using YOLO
This project implements a real-time accident detection system using the YOLO (You Only Look Once) model. The system processes video files to identify and save frames where accidents occur, providing an efficient method for monitoring road safety. The model can distinguish between different severities of incidents, ensuring that relevant data is captured and stored for further analysis.

# Features
**Video Processing:** The code can process multiple video files from a specified directory, making it easy to analyze extensive datasets.

**Real-time Detection:** Utilizes the YOLO model for fast and accurate detection of accidents in video frames.
**Notification System:** Implements a rate-limiting mechanism that captures and saves the highest confidence frames where accidents are detected, preventing redundant notifications within a specified time interval.
**Dynamic Output Handling:** Automatically creates an output directory for each video, saving frames that contain detected incidents.
**Custom Model Support:** The system supports custom-trained YOLO models, allowing users to tailor the detection capabilities to their specific needs.
# Getting Started
Prerequisites
To run this project, ensure you have the following installed:

**Python 3.x
OpenCV (cv2)
YOLOv8 (ultralytics library)
Installation**
# Clone the repository:
**git clone https://github.com/your_username/accident-detection-yolo.git**
**cd accident-detection-yolo**
**Install the required libraries:**

**pip install opencv-python ultralytics**
Download or train your own YOLO model and place the model weights in the directory: runs/detect/train8/weights/best.pt.

# Usage
Place the video files you wish to analyze in the videos/Accident Detection videos directory.
**Run the script:**
**python accident_detection.py**
The processed frames containing detected accidents will be saved in the output_frames directory under separate folders for each video.
Customization
You can modify the notification_interval variable to change the time limit for notifications.
The confidence threshold for detection can be adjusted by changing the conf parameter in the model call.
Code Overview
# The main script (accident_detection.py) follows these steps:

**Model Loading:** It checks for the existence of the YOLO model file and loads it for inference.
Video Processing: For each video, frames are captured and analyzed for detected incidents.
**Detection Logic:** The system identifies frames containing incidents classified as "severe" or "moderate" based on confidence scores.
**Frame Saving:** Detected frames are saved in a designated output directory with a filename indicating the frame number and incident type.
**Cleanup:** Releases the video capture object and closes any OpenCV windows.
Contribution
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

# License
**Creative Commons License (CC)**

# Acknowledgments
**YOLO - For the powerful object detection framework.
OpenCV - For image and video processing capabilities.**

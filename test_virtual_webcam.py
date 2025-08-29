import cv2
import numpy as np
import pyvirtualcam
import os

# Path to the video file
video_path = r"D:\test.webm"

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at {video_path}")
    exit(1)

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file {video_path}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")

with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
    print(f'Using virtual camera: {cam.device}')
    print(f'Streaming video from: {video_path}')
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            # End of video reached, restart from beginning
            print("End of video reached, restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            continue
        
        # Convert BGR (OpenCV format) to RGB (pyvirtualcam format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Send frame to virtual camera
        cam.send(frame_rgb)
        cam.sleep_until_next_frame()
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Streamed {frame_count}/{total_frames} frames...")

# Release the video capture object
cap.release()

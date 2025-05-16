"""
Automatically download and extract the kth dataset for handclapping action.
Author: Paul Verhoeven
Date: 13-05-2025 (dd-mm-yyyy)
"""

import os
import zipfile
import urllib.request
import cv2

def download_and_extract_handclapping(action: str) -> None:
    """
    Automatically download and extract the kth dataset for a specific action.
    """
    url = f"http://www.csc.kth.se/cvap/actions/{action}.zip"
    dest_folder = os.path.join("dataset", action)
    zip_path = os.path.join(dest_folder, f"{action}.zip")

    os.makedirs(dest_folder, exist_ok=True)
    if not os.path.exists(zip_path):
        print(f"Downloading {action}.zip...")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print(f"{action}.zip file already exists.")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    print("Extraction complete.")

    os.remove(zip_path)
    print(f"{action+".zipfile removed.":50}")
    print()

def extract_and_save_frames(action: str) -> None:
    """
    Extract the frames from the kth dataset and save them as images. This is done as the dataset is too large to be used as a video.
    """
    folder = os.path.join("dataset", action)


    for video_file in os.listdir(folder):
        if video_file.endswith(".avi"):
            cap = cv2.VideoCapture(os.path.join(folder, video_file))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save the frame as an image
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_file_name = video_file.replace('uncomp.avi', f"frame_{frame_number}.jpg")
                frame_path = os.path.join(folder, frame_file_name)
                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, frame)
            cap.release()
    
            # remove the video file
            os.remove(os.path.join(folder, video_file))
            print(f"Removed {video_file:50}", end="\r")



# All the kth action/movement types
movement_types = ['handclapping', 'handwaving', 'walking', 'running', 'jogging', 'boxing']
for action in movement_types:
    download_and_extract_handclapping(action)
    extract_and_save_frames(action)
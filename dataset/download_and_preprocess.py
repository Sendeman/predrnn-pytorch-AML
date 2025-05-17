"""
Automatically download and extract the kth dataset for handclapping action. 
And preprocess the data by extracting the frames and saving them as single-channel normalized (divided by 255) torch tensors.
Author: Paul Verhoeven
Date: 13-05-2025 (dd-mm-yyyy)
"""

import zipfile
import urllib.request
import cv2
from torch import Tensor, save
import numpy as np
from pathlib import Path

def download_and_extract(action: str, overwrite: bool = False) -> bool:
    """
    Automatically download and extract the kth dataset for a specific action.
    """
    url = f"http://www.csc.kth.se/cvap/actions/{action}.zip"
    dest_folder = Path("dataset") / "KTH_data" / action
    zip_path = dest_folder / f"{action}.zip"
    
    if dest_folder.exists() and not overwrite:  # "Dataset already exists at dest_folder. Use overwrite=True to re-download.
        return False

    dest_folder.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    zip_path.unlink()

def extract_and_save_frames(action: str, extraction: bool) -> None:
    """
    Extract the frames from the kth dataset and save them as images. This is done as the dataset is too large to be used as a video.
    """
    folder = Path("dataset") / "KTH_data" / action

    for video_file in folder.glob("*.avi"):
        cap = cv2.VideoCapture(str(video_file))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[:, :, 0]  # Convert to single-channel grayscale
            frame = np.array(frame) / 255.0  # Normalize the pixel values to [0, 1]
            frame_tensor = Tensor(frame)

            # Save the frame as a .pt file
            frame_file_name = video_file.stem.replace('uncomp', f"frame_{frame_idx}")
            frame_path = folder / f"{frame_file_name}.pt"
            if not frame_path.exists():
                save(frame_tensor, frame_path)
            frame_idx += 1
        cap.release()

        # remove the video file
        video_file.unlink()

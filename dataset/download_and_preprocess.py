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

def download_and_extract(overwrite: bool = False) -> bool:
    """
    Automatically download and extract the kth dataset for a specific action.
    Args:
        action (str): The action to download (e.g., "handclapping").
        overwrite (bool): If True, overwrite the existing dataset. Default is False.
    """
    url = f"http://www.csc.kth.se/cvap/actions/running.zip"
    dest_folder = Path("dataset") / "kth" / "running"
    zip_path = dest_folder / f"running.zip"
    
    if dest_folder.exists() and not overwrite:  # "Dataset already exists at dest_folder. Use overwrite=True to re-download.
        return False

    dest_folder.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)

    zip_path.unlink()

    return True

def extract_and_save_frames() -> None:
    """
    Extract the frames from the kth dataset and save them as images. This is done as the dataset is too large to be used as a video.
    Args:
        action (str): The action to process (e.g., "handclapping").
        extraction (bool): If True, extract the frames from the videos.
    """
    folder = Path("dataset") / "kth/running"

    for video_file in folder.glob("*.avi"):
        frames_target_folder = folder / video_file.name[:19]
        frames_target_folder.mkdir(exist_ok=True)
                                
        vidcap = cv2.VideoCapture(video_file)
        frame_idx = 0
        
        while vidcap.isOpened():
            success, image = vidcap.read()
            if success:
                frame_file_name = f'image-{frame_idx:03}.png'
                frame_path = frames_target_folder / frame_file_name

                # Resize to 128x128
                image = cv2.resize(image, (128, 128), image) 
                cv2.imwrite(f'./{str(frame_path)}', image)
                frame_idx += 1
            else:
                break
        cv2.destroyAllWindows()
        vidcap.release()

        # Remove the video file
        video_file.unlink()
        
        
def process_frames() -> None:
    """
    Turns the KTH images into tensors and saves them.
    """
    source_folder = Path("dataset") / "kth/running"

    target_folder = Path("dataset") / "kth_processed/running"
    target_folder.mkdir(parents=True, exist_ok=True)

    for folder in source_folder.glob("*"):
        if not folder.is_dir():
            break
        
        person_target_folder = target_folder / folder.name
        person_target_folder.mkdir(exist_ok=True)
        
        for file_path in folder.glob("*.png"):
            frame = cv2.imread(str(file_path))
            
            frame = frame[:, :, 0]          # Convert to single-channel grayscale
            frame = np.array(frame) / 255.0 # Normalize the pixel values to [0, 1]
            frame_tensor = Tensor(frame)    # Convert to tensor

            frame_path = person_target_folder / file_path.with_suffix('.pt').name
            if not frame_path.exists():
                save(frame_tensor, str(frame_path))

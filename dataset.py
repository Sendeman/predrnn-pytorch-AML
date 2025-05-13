import os
import zipfile


import urllib.request

def download_and_extract_handclapping(action: str):
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
    print(f"{action}.zip file removed.")



movement_types = ['handclapping', 'handwaving', 'walking', 'running', 'jogging', 'boxing']
for action in movement_types:
    download_and_extract_handclapping(action)
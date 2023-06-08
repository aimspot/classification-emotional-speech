import yadisk
import zipfile
import os 
import time

y = yadisk.YaDisk(token="y0_AgAAAABdcI3zAAn0RwAAAADj3zGiww39YKFIQ1G2HN-WOGrI5pYEEFk")

def folder_to_zip(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname=relative_path)


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def remove_model(name_model):
    try:
        y.remove(f"save_models/{name_model}.zip", permanently=True)
    except:
        print("Remove cancel")


def upload_model(name_model):
    folder_to_zip(f"save_models/{name_model}", f"save_models/{name_model}.zip")
    try:
        y.upload(f"save_models/{name_model}.zip", f"/models/{name_model}.zip")
    except:
        print("Model loaded")


def download_model(name_model):
    y.download(f"/models/{name_model}.zip", f"{name_model}.zip")
    unzip_file(f"{name_model}.zip", f"{name_model}")
    print("Model downloaded")

def download_sound(name_sound):
    y.download(f"/sound/{name_sound}.wav", f"{name_sound}.wav")
    print("Sound downloaded")

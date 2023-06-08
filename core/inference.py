import time
import psutil
from keras.models import load_model
from utils.database import Database
from utils.yandex_cloud import download_model, download_sound
from utils.extract_features import get_features
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def predict_with_memory_and_time_measurement(model, path_sound):
    emotion = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
   # path_sound = "03-01-01-01-01-01-05.wav"
    try:
        download_sound(path_sound.split('.')[0])
    except:
        print("")

    test = get_features(path_sound)
    testX = []
    for ele in test:
        testX.append(ele)

    scaler = StandardScaler()
    testX = scaler.fit_transform(testX)
    testX = np.expand_dims(testX, axis=2)

    start_time = time.time()
    predictions = model.predict(testX)
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)
    counts = np.sum(predicted_labels == 1, axis=0)

    max_index = np.argmax(counts)

    print("Emotion: ", emotion[max_index])
    predict_time = time.time() - start_time
    print(f"Time: {predict_time} sec")
    return predict_time, emotion[max_index]

# if __name__ == "__main__":
#     predict_with_memory_and_time_measurement()



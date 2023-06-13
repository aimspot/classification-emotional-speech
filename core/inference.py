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

def predict_with_memory_and_time_measurement(model, path_sound, tfl = False):
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
    if tfl:
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        np_features = np.array(testX[0])
        # print(np_features.shape)

        # If the expected input type is int8 (quantized model), rescale data
        input_type = input_details[0]['dtype']
        if input_type == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            np_features = (np_features / input_scale) + input_zero_point
            
        # Convert features to NumPy array of expected type
        np_features = np_features.astype(input_type)

        # Add dimension to input sample (TFLite model expects (# samples, data))
        np_features = np.expand_dims(np_features, axis=0)

        # Create input tensor out of raw features
        model.set_tensor(input_details[0]['index'], np_features)
        model.invoke()
        predictions = model.get_tensor(output_details[0]['index'])
    else:
        predictions = model.predict(testX)
    threshold = 0.5
    predicted_labels = (predictions > threshold).astype(int)
    print(predicted_labels)
    counts = np.sum(predicted_labels == 1, axis=0)

    max_index = np.argmax(counts)

    print("Emotion: ", emotion[max_index])
    predict_time = time.time() - start_time
    print(f"Time: {predict_time} sec")
    return predict_time, emotion[max_index]

# if __name__ == "__main__":
#     predict_with_memory_and_time_measurement()



import time
import psutil
from keras.models import load_model
from utils.database import Database
from utils.yandex_cloud import download_model
import tensorflow as tf

def predict_with_memory_and_time_measurement():
    db = Database()
    name_model = "2023-06-08-00-15-24"
    path_sound = "03-01-01-01-01-01-05.wav"

    try:
        download_model(name_model)
    except:
        print("Model is ready")
    model = tf.keras.models.load_model(f'{name_model}')

    # Загрузка модели
    start_memory = psutil.virtual_memory().used
    model = tf.keras.models.load_model(f'{name_model}')
    load_memory = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)

    # Предсказание
    #start_memory = psutil.virtual_memory().used
    start_time = time.time()
    predictions = model.predict(input_data)
    #predict_memory = psutil.virtual_memory().used - start_memory
    predict_time = time.time() - start_time
    print(load_memory)
    print(predict_time)

    return load_memory, predict_time

if __name__ == "__main__":
    predict_with_memory_and_time_measurement()



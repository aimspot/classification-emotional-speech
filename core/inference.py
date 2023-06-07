import time
import psutil
from keras.models import load_model

def predict_with_memory_and_time_measurement(model_path, input_data):

    # Загрузка модели
    start_memory = psutil.virtual_memory().used
    model = load_model(model_path)
    load_memory = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)

    # Предсказание
    #start_memory = psutil.virtual_memory().used
    start_time = time.time()
    predictions = model.predict(input_data)
    #predict_memory = psutil.virtual_memory().used - start_memory
    predict_time = time.time() - start_time
    print()

    return load_memory, predict_time



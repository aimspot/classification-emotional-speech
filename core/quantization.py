import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_model_optimization as tfmot
from metrics import metrics_model, get_split_dataset
from train import get_name_model,save_model

from utils.database import Database
from utils.yandex_cloud import download_model, upload_model_tfl



def model_optimization():
    """
    Квантизация модели включает в себя замену чисел с плавающей точкой на целочисленные представления с фиксированной точностью. 
    Это позволяет снизить требования к памяти и вычислительным ресурсам,
    а также повысить эффективность работы модели на аппаратных устройствах, 
    таких как микроконтроллеры или графические процессоры.
    """
    db = Database()
    name_model = db.get_best_model()
    try:
        download_model(name_model)
    except:
        print("Model")

    model = tf.keras.models.load_model(f'{name_model}')

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    name = db.get_model_by_name(name_model)

    name_model = get_name_model()

    with open(f'{name_model}.tflite', 'wb') as f:
        f.write(tflite_model)
    
    db.insert_model_name(f'tfl_{name}', name_model)
    upload_model_tfl(name_model)



if __name__ == '__main__':
    model_optimization()
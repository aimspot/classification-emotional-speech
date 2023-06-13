from flask import Flask, jsonify, request
from utils.database import Database
from utils.yandex_cloud import download_model, download_model_tfl
from inference import predict_with_memory_and_time_measurement
import tensorflow as tf
import numpy as np

app = Flask(__name__)

db = Database()
name_model = db.get_best_model()
try:
    download_model(name_model)
except:
    print("Model")
model = tf.keras.models.load_model(f'{name_model}')

tfl = False


@app.route('/model', methods=['POST'])
def model():
    global model
    global tfl
    name_model = request.json['name_model']
    name = db.get_model_by_name(name_model)
    if name.split('_')[0] =='tfl':
        try:
            download_model_tfl(name_model)
        except:
            print("Model is ready")

        model = tf.lite.Interpreter(model_path=f'{name_model}.tflite')
        model.allocate_tensors()
        tfl = True

        result = {
            'Answer': "New model is ready"
        }
    else:
        try:
            download_model(name_model)
        except:
            print("Model")
        model = tf.keras.models.load_model(f'{name_model}')
        tfl = False
        result = {
            'Answer': "New model is ready"
        }
    return jsonify(result)



@app.route('/predict', methods=['POST'])
def predict():
    audio_path = request.json['audio_path']
    _, answer = predict_with_memory_and_time_measurement(model, audio_path, tfl)

    result = {
        'prediction': answer
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
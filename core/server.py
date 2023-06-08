from flask import Flask, jsonify, request
from utils.database import Database
from utils.yandex_cloud import download_model
from inference import predict_with_memory_and_time_measurement
import tensorflow as tf
import numpy as np

app = Flask(__name__)

db = Database()
name_model = db.get_best_model()
try:
    download_model(name_model)
except:
    print("Model is ready")
model = tf.keras.models.load_model(f'{name_model}')



@app.route('/predict', methods=['POST'])
def predict():
    audio_path = request.json['audio_path']
    _, answer = predict_with_memory_and_time_measurement(model, audio_path)

    result = {
        'prediction': answer
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
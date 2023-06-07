from flask import Flask, jsonify, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    # Получение пути к аудио файлу из запроса
    audio_path = request.json['audio_path']

    # Загрузка аудио файла (ваш код загрузки)

    # Предобработка аудио файла (ваш код предобработки)

    # Выполнение предсказания модели
    predictions = model.predict(np.expand_dims(audio_data, axis=0))

    # Возвращение результата предсказания
    result = {
        'prediction': predictions.tolist()
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
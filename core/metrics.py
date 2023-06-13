import argparse
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils.yandex_cloud import download_model, download_model_tfl
from utils.database import Database
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import psutil
import numpy as np
import pandas as pd
from inference import predict_with_memory_and_time_measurement

def get_split_dataset(df):
    #df=pd.read_csv(path_to_csv)
    X = df.iloc[: ,:-1].values
    try:
        Y = df['s_Labels'].values
    except:
        Y = df['s_labels'].values
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    return x_train, x_test, y_train, y_test


def select_best_model(db):
    name_model_list, accuracy_list, f1_list = db.get_model_metrics()
    best_accuracy = 0.0
    best_f1 = 0.0
    best_model_name = ""

    for name_model, accuracy, f1 in zip(name_model_list, accuracy_list, f1_list):
        if accuracy > best_accuracy and f1 > best_f1:
            best_accuracy = accuracy
            best_f1 = f1
            best_model_name = name_model
    db.delete_best_model()
    db.insert_best_model(best_model_name)
    print(best_model_name)


def metrics_model():
    db = Database()
    df = db.getting_data()
    predict_tfl = []
    tfl = False
    predicted_labels = None
    x_train, x_test, y_train, y_test = get_split_dataset(df)
    names, name_models = db.get_empty_metrics()
    model = None
    for name, name_model in zip(names, name_models):
        if name.split('_')[0] != 'tfl':
            try:
                download_model(name_model)
            except:
                print("Model is ready")
            start_memory = psutil.virtual_memory().used
            model = tf.keras.models.load_model(f'{name_model}')
            load_memory = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)
            predictions = model.predict(x_test)
            threshold = 0.5
            predicted_labels = (predictions > threshold).astype(int)

        else:
            tfl = True
            try:
                download_model_tfl(name_model)
            except:
                print("Model is ready")
            for i in range(0, x_test.shape[0]):
                start_memory = psutil.virtual_memory().used
                model = tf.lite.Interpreter(model_path=f'{name_model}.tflite')
                model.allocate_tensors()
                # Получение информации о входных и выходных тензорах
                input_details = model.get_input_details()
                output_details = model.get_output_details()
                load_memory = (psutil.virtual_memory().used - start_memory) / (1024 * 1024)
                np_features = np.array(x_test[i])
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
                pred = model.get_tensor(output_details[0]['index'])
                threshold = 0.5
                prediction = (pred > threshold).astype(int)
                predict_tfl.append(prediction[0])
                predicted_labels = predict_tfl

        true_labels = y_test

        precision = precision_score(true_labels, predicted_labels, pos_label='positive', average='micro')
        recall = recall_score(true_labels, predicted_labels, pos_label='positive', average='micro')
        f1 = f1_score(true_labels, predicted_labels, pos_label='positive', average='micro')
        accuracy = accuracy_score(true_labels, predicted_labels)
        time, _ = predict_with_memory_and_time_measurement(model, "03-01-01-01-01-01-05.wav", tfl)

        print("Precision:", precision)
        print("Recall:", recall)
        print("Accuracy:", accuracy)
        print("F1-score:", f1)
        print(f"Time: {time} sec")
        print(f"Ram: {load_memory} mb")

        db.delete_null_metrics(name, name_model)
        db.insert_metrics(name, name_model, precision, recall, accuracy, f1, time, load_memory)
    select_best_model(db)

if __name__ == "__main__":
    metrics_model()
    #interpreter = tf.lite.Interpreter(model_path='/media/farm/ssd_1_tb_evo_sumsung/classification-emotional-speech/core/2023-06-13-17-39-27.tflite')
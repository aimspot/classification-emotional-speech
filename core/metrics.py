import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils.yandex_cloud import download_model
from utils.database import Database
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd


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



def metrics_model():
    #x_train, x_test, y_train, y_test = get_split_dataset(opt.data)
    db = Database()
    df = db.getting_data()
    x_train, x_test, y_train, y_test = get_split_dataset(df)
    names, name_models = db.get_empty_metrics()
    for name, name_model in zip(names, name_models):
        #Запустить цикл по моделям без метрик
        download_model(name_model)
        model = tf.keras.models.load_model(f'save_models/{name_model}')
        predictions = model.predict(x_test)

        threshold = 0.5
        predicted_labels = (predictions > threshold).astype(int)
        true_labels = y_test

        precision = precision_score(true_labels, predicted_labels, pos_label='positive', average='micro')
        recall = recall_score(true_labels, predicted_labels, pos_label='positive', average='micro')
        f1 = f1_score(true_labels, predicted_labels, pos_label='positive', average='micro')
        accuracy = accuracy_score(true_labels, predicted_labels)
        print("Precision:", precision)
        print("Recall:", recall)
        print("Accuracy:", accuracy)
        print("F1-score:", f1)
        db.insert_metrics(model, name_model, precision, recall, accuracy, f1)

if __name__ == "__main__":
    metrics_model()
import pandas as pd
import numpy as np
from tqdm import tqdm

from models.cnn import cnn_model
from models.dcnn import dcnn_model
from models.lstm import lstm_model
from models.rnn import rnn_model

import tensorflow as tf
from tensorflow.keras import backend as K

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

import keras
import tensorflow as tf

import argparse

from datetime import datetime

def get_split_dataset(path_to_csv):
    df=pd.read_csv(path_to_csv)
    X = df.iloc[: ,:-1].values
    Y = df['Labels'].values
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, shuffle=True)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    return x_train, x_test, y_train, y_test

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='batch - size')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--lr', type=float, default=0.0000001, help='Learning rate')
    parser.add_argument('--model', type=str, default='CNN', help='initial model')
    return parser.parse_args()

def get_date_time():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_datetime

    
    
def save_model(model, name):
    model.save(f'save_models/{name}_{get_date_time()}', save_format='tf')
    
    

def main(opt):
    #add getting data from db
    x_train, x_test, y_train, y_test = get_split_dataset('utils/final_csv_actor.csv')
    
    for name, model_init in zip(['CNN', 'DCNN', 'LSTM', 'RNN'], [cnn_model(x_train), dcnn_model(x_train), lstm_model(x_train), rnn_model(x_train)]):
        if opt.model == name:
            model = model_init
            
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=opt.lr)
    history=model.fit(x_train, y_train, batch_size=opt.bs, epochs=opt.epochs, validation_data=(x_test, y_test), callbacks=[rlrp])
    
    save_model(model, opt.model)
    

if __name__ == "__main__":
    opt = opt()
    main(opt)
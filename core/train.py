
from models.cnn import cnn_model
from models.dcnn import dcnn_model
from models.lstm import lstm_model
from models.rnn import rnn_model
import psutil
from utils.database import Database
#from eval_model import eval_model
from metrics import metrics_model, get_split_dataset

import tensorflow as tf
from tensorflow.keras import backend as K

from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf

import argparse
from utils.yandex_cloud import upload_model
from datetime import datetime


def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=16, help='batch - size')
    parser.add_argument('--epochs', type=int, default=5, help='total training epochs')
    parser.add_argument('--lr', type=float, default=0.0000001, help='Learning rate')
    parser.add_argument('--model', type=str, default='CNN', help='initial model')
    return parser.parse_args()

def get_date_time():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_datetime

    
    
def save_model(model, name):
    name_model = f'{name}_{get_date_time()}'
    tf.saved_model.save(model, f'save_models/{name_model}')
    return name_model
    
    

def main(opt):
    db = Database()
    df = db.getting_data()
    x_train, x_test, y_train, y_test = get_split_dataset(df)
    for name, model_init in zip(['CNN', 'DCNN', 'LSTM', 'RNN'], [cnn_model(x_train), dcnn_model(x_train), lstm_model(x_train), rnn_model(x_train)]):
        if opt.model == name:
            model = model_init
            
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=opt.lr)
    psutil.virtual_memory()
    history=model.fit(x_train, y_train, batch_size=opt.bs, epochs=opt.epochs, validation_data=(x_test, y_test), callbacks=[rlrp])
    memory_info = psutil.virtual_memory()
    memory_usage_mb = memory_info.used / (1024 * 1024)
    print("Memory usage:", memory_usage_mb, "MB")
    name_model = save_model(model, opt.model)
    db.insert_model_name(opt.model, name_model)
    upload_model(name_model)
    #metrics_model(opt.model, name_model)
    

if __name__ == "__main__":
    opt = opt()
    main(opt)
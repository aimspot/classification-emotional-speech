from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Flatten
import tensorflow.keras.backend as K

def lstm_model(x_train):
    model = Sequential()
    model.add(LSTM(512, input_shape=(x_train.shape[1], 1), return_sequences=True))
    model.add(BatchNormalization())

    model.add(LSTM(512, return_sequences=True))
    model.add(BatchNormalization())

    model.add(LSTM(256, return_sequences=True))
    model.add(BatchNormalization())

    model.add(LSTM(128, return_sequences=True))
    model.add(BatchNormalization())

    model.add(LSTM(64, return_sequences=True))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(8, activation="softmax"))

    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
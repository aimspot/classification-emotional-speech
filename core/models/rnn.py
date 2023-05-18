from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Flatten, Dropout

def rnn_model(x_train):
    model = Sequential()
    model.add(LSTM(256, input_shape=(x_train.shape[1], 1), return_sequences=True))

    model.add(LSTM(64))
    model.add(Dropout(0.3))

    model.add(Dense(8, activation="softmax"))

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
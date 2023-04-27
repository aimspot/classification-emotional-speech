from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, LSTM, Bidirectional
#change parametrs
class DCNN:
    def __init__(self):
        self.model.add(Conv1D(512, kernel_size=5, strides=1,
                        padding="same", activation="relu",
                        input_shape=(x_train.shape[1], 1)))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

        self.model.add(Conv1D(512, kernel_size=5, strides=1,
                                padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

        self.model.add(Conv1D(256, kernel_size=5, strides=1,
                                padding="same", activation="relu"))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=5, strides=2, padding="same"))

        self.model.add(Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        self.model.add(Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling1D(pool_size=3, strides = 2, padding = 'same'))

        self.model.add(Flatten())
        self.model.add(Dense(512, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(14, activation="softmax"))


        self.model.compile(optimizer = 'RMSprop' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

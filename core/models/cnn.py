from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense

class CNN:
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(x_train.shape[1], 1)))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        self.model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        self.model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
        self.model.add(Dropout(0.2))

        self.model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

        self.model.add(Flatten())
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(units=8, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



#my_model = MyModel()


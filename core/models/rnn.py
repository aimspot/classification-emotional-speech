import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, TimeDistributed, concatenate
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, LeakyReLU, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
#change parametrs
class MyRNN:
    def __init__(self, x_train):
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_1_MELSPECT', input_shape=(x_train.shape[1], 1)))
        self.model.add(TimeDistributed(BatchNormalization(), name='BatchNorm_1_MELSPECT'))
        self.model.add(TimeDistributed(Activation('elu'), name='Activ_1_MELSPECT'))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'), name='MaxPool_1_MELSPECT'))
        self.model.add(TimeDistributed(Dropout(0.2), name='Drop_1_MELSPECT'))     

        ## Second LFLB (local feature learning block)
        self.model.add(TimeDistributed(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_2_MELSPECT'))
        self.model.add(TimeDistributed(BatchNormalization(), name='BatchNorm_2_MELSPECT'))
        self.model.add(TimeDistributed(Activation('elu'), name='Activ_2_MELSPECT'))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_2_MELSPECT'))
        self.model.add(TimeDistributed(Dropout(0.2), name='Drop_2_MELSPECT'))

        ## Second LFLB (local feature learning block)
        self.model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_3_MELSPECT'))
        self.model.add(TimeDistributed(BatchNormalization(), name='BatchNorm_3_MELSPECT'))
        self.model.add(TimeDistributed(Activation('elu'), name='Activ_3_MELSPECT'))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_3_MELSPECT'))
        self.model.add(TimeDistributed(Dropout(0.2), name='Drop_3_MELSPECT'))

        ## Second LFLB (local feature learning block)
        self.model.add(TimeDistributed(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'), name='Conv_4_MELSPECT'))
        self.model.add(TimeDistributed(BatchNormalization(), name='BatchNorm_4_MELSPECT'))
        self.model.add(TimeDistributed(Activation('elu'), name='Activ_4_MELSPECT'))
        self.model.add(TimeDistributed(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'), name='MaxPool_4_MELSPECT'))
        self.model.add(TimeDistributed(Dropout(0.2), name='Drop_4_MELSPECT'))  

        ## Flat
        self.model.add(TimeDistributed(Flatten(), name='Flat_MELSPECT'))                      
                               
        # Apply 2 LSTM layer and one FC
        self.model.add(LSTM(256, return_sequences=False, dropout=0.2, name='LSTM_1'))

        self.model.add(Dense(units=8, activation='softmax', name='FC'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
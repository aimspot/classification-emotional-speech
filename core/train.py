import pandas as pd
import numpy as np
from tqdm import tqdm
from models.cnn import CNN

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau

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


def main():
    x_train, x_test, y_train, y_test = get_split_dataset('final_csv_actor.csv')
    model = CNN(x_train)
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=0, patience=2, min_lr=0.0000001)
    history=model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_test, y_test), callbacks=[rlrp])

if __name__ == "__main__":
    main()
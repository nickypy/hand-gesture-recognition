import pickle
import cv2
import numpy as np
import train

from data import Data
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from matplotlib import pyplot as plt

if __name__ == '__main__':
    # load data set
    dataset = pickle.load(open("dataset.pkl", "rb"))

    (X_train, y_train), (X_test, y_test) = dataset.load_data()

    num_classes = np.unique(y_train).shape[0]


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    train_mean = np.mean(X_train)
    train_std = np.std(X_train)

    test_mean = np.mean(X_test)
    test_std = np.std(X_test)

    X_train -= train_mean
    X_train /= train_std
    X_test -= test_mean
    X_test /= test_std

    encoder = LabelEncoder()
    encoder.fit(y_train)
    y_train = encoder.transform(y_train)

    encoder = LabelEncoder()
    encoder.fit(y_test)
    y_test = encoder.transform(y_test)

    Y_train = np_utils.to_categorical(y_train, num_classes)
    Y_test = np_utils.to_categorical(y_test, num_classes)

    model = load_model('model.h5')
    scores = model.evaluate(X_test, Y_test, verbose=1)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
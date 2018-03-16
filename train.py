# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import np_utils
from lenet import LeNet
from data import Data

import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 32

if __name__ == '__main__':
	# load dataset
	dataset = pickle.load(open("dataset.pkl", "rb"))

	(X_train, y_train), (X_test, y_test) = dataset.load_data()

	num_train, height, width, depth = X_train.shape
	num_test = X_test.shape[0]
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

	# initialize the model
	print("[INFO] compiling model...")
	model = LeNet.build(width=width, height=height, depth=depth, classes=num_classes)
	opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
	model.compile(loss="binary_crossentropy", optimizer=opt,
		metrics=["accuracy"])

	# train the network
	print("[INFO] training network...")
	H = model.fit(X_train, Y_train,
	    batch_size=BS, epochs=EPOCHS,
	    verbose=1, validation_split=0.1)


	# save the model to disk
	print("[INFO] serializing network...")
	model.save('model.h5')

	# plot the training loss and accuracy
	plt.style.use("ggplot")
	plt.figure()
	N = EPOCHS
	plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
	plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
	plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
	plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
	plt.title("Training Loss and Accuracy on Hand Gesture")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig("data.png")

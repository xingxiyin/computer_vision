#-*-coding:utf-8-*-
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from minigooglenet import MiniGooogleNet
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.datasets import cifar10
import numpy as np
import argparse
import os

# Define the total number of epochs to train for along with the initial learning rae
NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch, NUM_EPOCHS=70, INIT_LR=5e-3, power=1.0):
    # Compute the new learning rate based on polynomial decay
    alpha = INIT_LR * (1 - (epoch / float(NUM_EPOCHS))) ** power

    # return the new learning rate
    return alpha

# Construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-m", "--model", required=True, help="Path to output model")
# ap.add_argument("-o", "--output", required=True, help="Path to directory (logs, plots, etc.)")
# args = vars(ap.parse_args())

# Loading the training and testing data, converting the images from integers to floats
print("[INFO] Loadin CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
# print(trainX)
testX = testX.astype("float")

# Apply mean substraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# Convert the labels from intergers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         horizontal_flip=True,
                         fill_mode="nearest")

# Construct the set of callbacks
callbacks = [LearningRateScheduler(poly_decay)]

# Initialize the optimizer and model
print("[INFO] Compliling model...")
opt = SGD(lr=INIT_LR, momentum=0.9)
model = MiniGooogleNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] training network...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX)//64,
                    epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

# Evaluate the network
print("[INFO] Evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))

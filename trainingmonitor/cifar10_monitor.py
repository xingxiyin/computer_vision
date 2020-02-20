import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from model_structure.minivggnet import MiniVGGNet
from trainingmonitor import TrainingMonitor
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="Path to the output loss/accuracy plot")
args = vars(ap.parse_args())


def step_decay(epoch, initAlpha=0.01, factor=0.5, dropEvery=5):
    # The larger our factor is, the slower the learning rate will decay,
    # Conversely, the smaller the factor is te faster the learing rate will decrease
    # Compute the learning rate for the current epoch
    alpha = initAlpha * (factor ** np.float((1 + epoch)/dropEvery))

    # return the leanring rate
    return float(alpha)


# Loading the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
              "ship", "truck"]

# Initialize the optimizer and model
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Construct the set of callbacks
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callback = [TrainingMonitor(figPath, jsonPath)]

# train the network
H = model.fit(trainX, trianY,
              validation_data=(testX, testY),
              batch_size=64,
              epochs=40,
              callbacks=callback,
              verbose=1)

# Evaluae the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=labelNames))

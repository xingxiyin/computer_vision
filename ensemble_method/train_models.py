#-*-coding:utf-8-*-
import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from model_structure.minivggnet import MiniVGGNet
from keras .preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="Path to output directory")
ap.add_argument("-m", "--models", required=True,
                help="Path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5,
                help="# of models to train")
args = vars(ap.parse_args())

# Loading the training and testing data, then scale it into the range [0, 1]
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trianY)
testY = lb.transform(testY)

# Initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
              'ship', "truck"]

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

# Loop over the number of models to trian
for i in np.arange(0, args["num_models"]):
    # Initialize the optimizer and model
    print("[INFO] Training model {}/{}".format(i+1), args["num_models"])
    opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the network
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
                            validation_data=(testX, testY), epochs=40,
                            steps_per_epoch=len(trainX)//64, verbose=1)

    # Save the model to disk
    model_path = [args["models"], "model_{}.model".format(i)]
    model.save(os.path.sep.join(model_path))

    # Evaluate the network
    predictions = model.prediction(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                                   target_names=labelNames)

    # Save the classification report to file
    paths = [args["output"], "model_{}.txt".format(i)]
    with open(os.path.sep.join(p), "w") as file:
        file.write(report)
        file.close()



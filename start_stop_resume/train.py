import matplotlib
matplotlib.usr("Agg")

from epochcheckpoint import EpochCheckpoint
from trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
from keras.models import load_model
import keras.backend as K
import numpy as np
import argparse
import os
import sys


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True,
                help="path to output checkpoint directory")
ap.add_argument("-m", "--model", type=str,
                help="path to *specific* model checkpoint to load")
ap.add_argument("-s", "--start-epoch", type=int, default=0,
                help="epoch to restart training at")
args = vars(ap.parse_args())


# Download the Fashion MNIST dataset
print("[INFO] loading Fashion MNIST...")
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

# Fashion MNIST images are 28x28 but the network we will be training is expecting 32x32 images
trainX = np.array([cv2.resize(x, (32, 32)) for x in trainX])
testX = np.array([cv2.resize(x, (32, 32)) for x in testX])

# scale data to the range of [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1, horizontal_flip=True,
                         fill_mode="nearest")

# reshape the data matrices to include a channel dimension (required
# for training)
trainX = trainX.reshape((trainX.shape[0], 32, 32, 1))
testX = testX.reshape((testX.shape[0], 32, 32, 1))

# if there is no specific model checkpoint supplied, then initialize
# the network (ResNet-56) and compile the model
if args["model"] is None:
    print("[INFO] compiling model...")
    opt = SGD(lr=1e-1)
    model = ResNet.build(32, 32, 1, 10, (9, 9, 9),
                         (64, 64, 128, 256), reg=0.0001)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
else:# otherwise, we're using a checkpoint model
    # load the checkpoint from disk
    print("[INFO] loading {}...".format(args["model"]))
    model = load_model(args["model"])

    # update the learning rate
    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, 1e-2)
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

# build the path to the training plot and training history
plotPath = os.path.sep.join(["output", "resnet_fashion_mnist.png"])
jsonPath = os.path.sep.join(["output", "resnet_fashion_mnist.json"])

# construct the set of callbacks
callbacks = [EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]),
            TrainingMonitor(plotPath, jsonPath=jsonPath, startAt=args["start_epoch"])]


# train the network
print("[INFO] training network...")
model.fit_generator(
                    aug.flow(trainX, trainY, batch_size=128),
                    validation_data=(testX, testY),
                    steps_per_epoch=len(trainX) // 128,
                    epochs=80,
                    callbacks=callbacks,
                    verbose=1)


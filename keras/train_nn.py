from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.models import Sequential
from small_vggnet import SmallVGGNet
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

import matplotlib
matplotlib.use("Agg")

# Plot function for the training loss and accuracy
def plot_result(args, history, num_epoch):
    x_aixs = np.arange(0, num_epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(x_aixs, history.history["loss"], label="train_loss")
    plt.plot(x_aixs, history.history["val_loss"], label="val_loss")
    plt.plot(x_aixs, history.history["accuracy"], label="train_acc")
    plt.plot(x_aixs, history.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(args["plot"])


def main(args, seed=111):
    # Initializer the images and labels
    print("[INFO] Loading iamges....")
    images = []
    labels = []

    # Grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    random.seed(42)
    random.shuffle(imagePaths)

    # Loop over the input images
    for imagePath in imagePaths:
        # 1.Loading the image, resize the image to be 32*32 pixel
        # 2.Flatten the image into 32*32*3 pixel image into a list, and store the image in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        images.append(image)

        # Extract the class label from the image path and update the labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)


    # Scale the raw pixel intensities to the range [0, 1]
    images = np.array(images, dtype="float")/255.0
    labels = np.array(labels)

    # Partition the images into training data and testing data
    train_X, test_X, train_Y, test_Y = train_test_split(images, labels, test_size=0.25, random_state=42)

    # Convert the labels from integers to vectors
    labeler = LabelBinarizer()
    train_Y = labeler.fit_transform(train_Y)
    test_Y = labeler.transform(test_Y)

    # Construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")

    # Initialize the network
    model = SmallVGGNet.build(width=64, height=64, channel=3, num_classes=len(labeler.classes_))
    # Define the keras network architecture
    # model = Sequential()
    # model.add(Dense(1024, input_shape=(3072, ), activation="relu"))
    # model.add(Dense(512, activation="relu"))
    # model.add(Dense(len(labeler.classes_), activation="softmax"))

    # Compile keras model
    # initialize learning rate and the number of epoch for training
    init_lr = 0.01
    num_epoch = 75

    # compile the model by SGD optimizer and categorical cross-entropy loss function
    print("[INFO] Training network...")
    optimizer = SGD(lr=init_lr, decay=init_lr/num_epoch)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Training the neural network
    history = model.fit_generator(generator=aug.flow(train_X, train_Y, batch_size=32),
                                validation_data=(test_X, test_Y),
                                steps_per_epoch=len(train_X)/32,
                                epochs=num_epoch)

    # Evaluate the network
    print("[INFO] Evaluating network...")
    predictions = model.predict(test_X, batch_size=32)
    print(classification_report(test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=labeler.classes_))

    # PLot the training loss and accuracy
    plot_result(args, history, num_epoch)

    # Saving the model and label binarizer to disk
    print("[INFO] Serializing network and label binarizer...")
    model.save(args["model"])
    with open(args["label"], "wb") as file:
        file.write(pickle.dumps(labeler))



if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path of the input images")
    ap.add_argument("-m", "--model", required=True,
                    help="Path of the output trained model")
    ap.add_argument("-l", "--label", required=True,
                    help="Path of the output label binarizer")
    ap.add_argument("-p", "--plot", required=True,
                    help="Path of the output accuracy/loss plot")
    args = vars(ap.parse_args())

    main(args)


    """
    python3 train_nn.py --dataset /home/yinxx/Project/data/CV/image --model ./model/lenet.model --label ./label/lenet_lb.pickle --plot ./plot/lenet.png 

    """






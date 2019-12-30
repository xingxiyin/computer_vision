import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def main(args):
    # Initialize the number of epoches and the number of batch size
    num_epochs = 50
    batch_size = 32
    
    # Derive the path to the directories containing the training, validation,
    # and testing splits, respectively
    TRAIN_PATH = os.path.sep.join([args["dataset"], "training"])
    VAL_PATH = os.path.sep.join([args["dataset"], "validation"])
    TEST_PATH = os.path.sep.join([args["dataset"], "testing"])

    # Calculate the total number of image of training, validation and testing dataset
    totalTrain = len(list(paths.list_image(TRAIN_PATH)))
    totalVal = len(list(paths.list_image(VAL_PATH)))
    totalTest = len(list(paths.list_image(TEST_PATH)))

    # Initialize the training dataset's augmentation object
    trainAug = ImageDataGenerator(rescale=1/255.0,
                                  rotation_range=20,
                                  zoom_range=0.05,
                                  width_shift_range=0.05,
                                  height_shift_range=0.05,
                                  shear_range=0.05,
                                  horizontal_flip=True,
                                  fill_mode="nearest")

    # Initialize the validation(and testing) data augmentation object
    valAug = ImageDataGenerator(rescale=1/255.0)

    # Initialize the traning generator
    trainGen = trainAug.flow_from_directory(TRAIN_PATH,
                                            class_mode="categorical",
                                            target_size=(64, 64),
                                            color_mode = "rgb",
                                            shuffle=True,
                                            batch_size=batch_size)

    # Initialize the validation generator
    valGen = valAug.flow_from_directory(VAL_PATH,
                                        class_mode="categorical",
                                        target_size=(64, 64),
                                        color_mode="rgb",
                                        shuffle=False,
                                        batch_size=batch_size)

    # Initialize the testing generator
    testGen = valAug.flow_from_directory(TEST_PATH,
                                         class_mode="categorical",
                                         target_size=(64, 64),
                                         color_mode="rgb",
                                         shuffle=False,
                                         batch_size=batch_size)

    # Initialize the model
    model = ResNet.build(64, 64, 3, 2, (2, 2, 3), (32, 64, 128, 256), reg=0.0005)
    optimizer = SGD(lr=1e-1, momentum=0.9, decay=0.01/num_epochs)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    # Train the model
    H = model.fit_generator(trainAug,
                            steps_per_epoch=totalTrain//batch_size,
                            validation_data=valGen,
                            validation_steps=totalVal//batch_size,
                            epochs=num_epochs)

    # reset the testing generator and then use our trained model to
    # make predictions on the data
    print("[INFO] evaluating network...")
    testGen.reset()
    predIdxs = model.predict_generator(testGen,
                                       steps=(totalTest // BS) + 1)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)

    # show a nicely formatted classification report
    print(classification_report(testGen.classes, predIdxs,
                                target_names=testGen.class_indices.keys()))

    # Save the network to disk
    print("[INFO] serializing network to '{}'...".format(args["model"]))
    model.save(args["model"])

    




if __name__ == '__main__':
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path of the input images")
    ap.add_argument("-m", "--model", required=True,
                    help="Path of the output trained model")
    ap.add_argument("-p", "--plot", required=True,
                    help="Path of the output accuracy/loss plot")
    args = vars(ap.parse_args())

    main(args)





